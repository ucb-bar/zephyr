/*
 * Copyright (c) 2023 BayLibre SAS
 * Written by: Nicolas Pitre
 * Copyright (c) 2026 — Saturn-fork V/F decoupling.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * RISC-V V-extension (RVV) state save/restore.
 *
 * This file owns *all* RVV-specific kernel-side context handling on
 * this fork.  It is deliberately separated from arch/riscv/core/fpu.c
 * (which now handles only the F extension) — see
 * agents/notes/zephyr_v_decouple_design.md for the rationale.
 *
 * Pairing with the F extension: some vector instructions (vfadd.vf,
 * vfmacc.vf, vfmv.{s,f}.{f,s}, etc.) read or write a scalar F register
 * as part of their semantics, so MSTATUS.FS must also be reachable
 * (CLEAN/DIRTY) at execution time.  We do *not* couple the two save
 * paths here: F state is still managed lazily by fpu.c, V state is
 * eagerly saved/restored in switch.S via the functions defined here.
 * A thread that uses both extensions ends up paying both an F lazy
 * trap (first F access) and a V eager save (every context switch).
 *
 * Eager V is the only mode supported on this fork right now; the lazy
 * V path that exists in upstream is gated out below until the
 * decoupled trap dispatch lands (see step 5 of the implementation plan
 * in zephyr_v_decouple_design.md).
 */

#include <zephyr/kernel.h>
#include <zephyr/kernel_structs.h>
#include <kernel_arch_interface.h>
#include <zephyr/sys/atomic.h>

#ifdef CONFIG_RISCV_ISA_EXT_V

/* HAS_V() is the runtime gate: build-time CONFIG_RISCV_ISA_EXT_V
 * combined with a per-hart misa probe so an asymmetric SMP build
 * (some harts without V) still does the right thing.  Reads misa
 * inline rather than caching in a per-CPU table — misa is a M-mode
 * read-only CSR and the read is single-cycle on Rocket. */
static inline unsigned long read_csr_misa(void)
{
	unsigned long x;
	__asm__ volatile("csrr %0, misa" : "=r"(x));
	return x;
}
#define MISA_EXT_BIT(ch) (1UL << ((ch) - 'A'))

#define HAS_V() (IS_ENABLED(CONFIG_RISCV_ISA_EXT_V) && \
		 (read_csr_misa() & MISA_EXT_BIT('V')))

/* ---------------------------------------------------------------------------
 * V-preemption-race mitigations (Saturn fork).  These were part of the
 * deployed baseline (verified byte-for-byte against zephyr_ub_base.elf disasm:
 * force_vstart0 8x-retry @0x800072b0/0x800073a0, strong_drain canary
 * @0x800072c6/0x800073b6, MB_V_SAVE_FENCE @0x8000724c/0x8000739a, thread_in_ram
 * guard @0x8000722e/0x8000732e, dbg globals @0x80082c50..80). They are re-stated
 * here alongside the t0/t1 register-realloc restore fix (see the reload loop in
 * z_riscv_vstate_restore_thread). The realloc is the load-bearing fix for the
 * a4/a5 RoCC-scalar-response strand; these mitigations cover the vstart-partial
 * / stale-vtype confound and provide diagnostics.
 * ------------------------------------------------------------------------- */

/* dbg globals + drain canary (file scope; addresses tracked in the deployed
 * ELF at 80082c50..80082c80). */
volatile unsigned long z_riscv_v_dbg_wild_save,    z_riscv_v_dbg_wild_save_ptr;
volatile unsigned long z_riscv_v_dbg_wild_restore, z_riscv_v_dbg_wild_restore_ptr;
volatile unsigned long z_riscv_v_dbg_restore_entry_vstart; /* vstart at restore entry */
volatile unsigned long z_riscv_v_dbg_restore_nonzero;      /* count restores w/ vstart!=0 */
static volatile unsigned long z_riscv_v_drain_canary;

/* Wild-pointer guard: a valid k_thread lives in SRAM and is 8-aligned. Guards
 * the eager save/restore against a corrupted thread pointer (a0) so a wild
 * vse8.v/vle8.v does not fault on an out-of-range address. NOTE: this does NOT
 * catch the observed a0=0x44 mechanism reaching the same vle8.v via a stale
 * mstatus_v/is_dirty path — it only rejects pointers outside SRAM. */
static inline bool z_riscv_v_thread_in_ram(const struct k_thread *t)
{
	uintptr_t p = (uintptr_t)t;
	return (p >= (uintptr_t)CONFIG_SRAM_BASE_ADDRESS) &&
	       (p <  (uintptr_t)CONFIG_SRAM_BASE_ADDRESS +
			     ((uintptr_t)CONFIG_SRAM_SIZE << 10)) &&
	       ((p & 7UL) == 0UL);
}

/* Force vstart==0 with a bounded (8x) drain+clear+verify retry. vsetvli does
 * NOT clear vstart per the RVV spec; a leftover non-zero vstart from a preempted
 * V op makes the following vsetvli/vse8.v/vle8.v start mid-register (partial
 * op / stale vtype). Retrying with a fence between attempts covers a decoupled
 * write to vstart that has not yet retired. */
static inline void z_riscv_force_vstart0(void)
{
	unsigned long vs;
	int n = 0;

	do {
		__asm volatile("fence rw, rw" ::: "memory");
		__asm volatile(".option push\n\t"
			       ".option arch, +v\n\t"
			       "csrw	vstart, x0\n\t"
			       "csrr	%0, vstart\n\t"
			       ".option pop\n\t"
			       : "=r"(vs));
	} while (vs != 0 && ++n < 8);
}

/* Stronger drain: fence, a dependent load-after-store to a canary, then consume
 * the loaded value under a second fence. Forces the store buffer / any in-flight
 * decoupled traffic to a defined ordering point before the bulk V load/store. */
static inline void z_riscv_v_strong_drain(void)
{
	unsigned long v;

	__asm volatile("fence rw, rw" ::: "memory");
	z_riscv_v_drain_canary = 1UL;            /* store */
	v = z_riscv_v_drain_canary;              /* dependent load-after-store */
	__asm volatile("fence rw, rw" :: "r"(v) : "memory"); /* consume -> ordering */
}

/* Snapshot the four V CSRs into the per-thread save area.  Caller must
 * have ensured MSTATUS.VS != OFF — otherwise these csrr ops trap with
 * illegal-instruction.  Used by both the eager and lazy save paths. */
void z_riscv_vstate_csr_save(struct z_riscv_v_context *dest)
{
	if (!HAS_V()) {
		return;
	}
	__asm volatile(".option push\n\t"
		       ".option arch, +v\n\t"
		       "csrr %0, vstart\n\t"
		       "csrr %1, vl\n\t"
		       "csrr %2, vtype\n\t"
		       "csrr %3, vcsr\n\t"
		       ".option pop\n\t"
		       : "=r"(dest->vstart), "=r"(dest->vl),
			 "=r"(dest->vtype), "=r"(dest->vcsr));
}

/* Restore the four V CSRs from the per-thread save area.  vsetvl
 * (zero rd, %1=AVL, %2=vtype) sets vl/vtype atomically as the spec
 * requires; vstart and vcsr are restored separately.  Note that
 * vsetvl{i} does NOT clear vstart per the RVV spec, so the caller
 * must ensure no V op runs between this restore and the user code
 * resuming — otherwise the restored vstart could be silently zeroed. */
void z_riscv_vstate_csr_restore(struct z_riscv_v_context *src)
{
	if (!HAS_V()) {
		return;
	}
	__asm volatile(".option push\n\t"
		       ".option arch, +v\n\t"
		       "vsetvl x0, %1, %2\n\t"
		       "csrw   vstart, %0\n\t"
		       "csrw   vcsr, %3\n\t"
		       ".option pop\n\t"
		       :: "r"(src->vstart), "r"(src->vl),
			  "r"(src->vtype), "r"(src->vcsr));
}

#ifndef CONFIG_RISCV_ISA_EXT_V_LAZY
/*
 * Eager V save/restore — the path actually used on this fork.
 *
 * Called from arch/riscv/core/switch.S on every context switch when
 * CONFIG_RISCV_ISA_EXT_V is set (and lazy is OFF).  Correctness
 * requirement: on every switch the new thread's full V state
 * (v0..v31 + vstart/vl/vtype/vcsr) must be reloaded, because the V
 * regs and V CSRs are global per-hart resources shared across threads.
 *
 * History: an earlier implementation gated *both* save and restore on
 * a per-thread `is_dirty` flag derived from MSTATUS.VS==DIRTY at save
 * time.  That broke the restore path: a thread that had never gone
 * through the full save would early-exit out of restore_thread with
 * the hardware V CSRs still holding the previous thread's values, so
 * the next vector op on this hart used a stale vstart.  On Saturn
 * this manifested as mtval=0xcc747057, mcause=6 inside picolibc's
 * vse64.v memcpy — see agents/notes/zephyr_rvv_context_switch_bug.md
 * and agents/notes/zephyr_rvv_fix_summary.md for the full history.
 *
 * Two correctness rules this code now enforces unconditionally:
 *   (a) Always reload v0..v31 + V CSRs on restore.  is_dirty is
 *       still updated by save (for diagnostics + future lazy code)
 *       but the restore path no longer gates on it.
 *   (b) Always csrw vstart, x0 before the bulk vse8.v / vle8.v.
 *       vsetvli does NOT clear vstart per the RVV spec; a leftover
 *       non-zero vstart from a preempted V op would cause partial
 *       saves/loads and silently corrupt v0..v(vstart-1).
 */
void z_riscv_vstate_save_thread(struct k_thread *thread)
{
	if (!HAS_V()) {
		return;
	}

	/* Wild-pointer guard: reject a corrupted thread pointer before it
	 * reaches the eager vse8.v (matches deployed baseline @0x8000722e). */
	if (!z_riscv_v_thread_in_ram(thread)) {
		z_riscv_v_dbg_wild_save++;
		z_riscv_v_dbg_wild_save_ptr = (unsigned long)thread;
		return;
	}

	struct z_riscv_v_context *save_to = &thread->arch.saved_v_context;
	unsigned long mstatus_v = csr_read(mstatus) & MSTATUS_VS;

	if (mstatus_v != 0) {
		/* V access is enabled (VS != OFF): V CSRs and v0..v31
		 * are reachable without trapping.  Save BOTH CSRs and
		 * v0..v31 unconditionally — Saturn does not flip
		 * MSTATUS.VS=DIRTY for every V op, so a DIRTY-only fast
		 * path could miss real reg modifications. */
		z_riscv_vstate_csr_save(save_to);
		z_riscv_force_vstart0();
		z_riscv_v_strong_drain();
		unsigned long vl;
		__asm volatile(
			".option push\n\t"
			".option arch, +v\n\t"
			"csrw		vstart, x0\n\t"
			"vsetvli	%0, x0, e8, m8, ta, ma\n\t"
			"vse8.v		v0, (%1)\n\t"
			"add		%1, %1, %0\n\t"
			"vse8.v		v8, (%1)\n\t"
			"add		%1, %1, %0\n\t"
			"vse8.v		v16, (%1)\n\t"
			"add		%1, %1, %0\n\t"
			"vse8.v		v24, (%1)\n\t"
			".option pop\n\t"
			: "=&r"(vl)
			: "r"(save_to->vreg)
			: "memory");
	}

	/* MB_V_SAVE_FENCE (save side): order the bulk vse8.v ahead of the
	 * is_dirty publish + mstatus flip (matches deployed @0x8000724c). */
	__asm volatile("fence rw, rw" ::: "memory");
	save_to->is_dirty = (mstatus_v != 0);

	csr_clear(mstatus, MSTATUS_VS);
	csr_set(mstatus, MSTATUS_VS_CLEAN);
}

void z_riscv_vstate_save(void)
{
	if (!HAS_V()) {
		return;
	}
	z_riscv_vstate_save_thread(_current);
}

void z_riscv_vstate_restore_thread(struct k_thread *thread)
{
	if (!HAS_V()) {
		return;
	}

	/* Wild-pointer guard: reject a corrupted thread pointer before it
	 * reaches the eager vle8.v (matches deployed baseline @0x8000732e).
	 * DIAGNOSTIC ROLE for the a0 residual: if a0 is wild AT ENTRY
	 * (memory-mediated thread-ptr corruption), this catches it here and
	 * increments z_riscv_v_dbg_wild_restore instead of faulting later at
	 * `ld a5,512(a0)`. If a0 is instead clobbered mid-function (decoupled
	 * resp with rd=a0), the guard passes and the later fault persists. */
	if (!z_riscv_v_thread_in_ram(thread)) {
		z_riscv_v_dbg_wild_restore++;
		z_riscv_v_dbg_wild_restore_ptr = (unsigned long)thread;
		return;
	}

	struct z_riscv_v_context *restore_from = &thread->arch.saved_v_context;

	/* Enable V access — csr_set only ORs bits, lifting VS from
	 * OFF/INIT/CLEAN to at least CLEAN so the V loads + CSR writes
	 * below don't raise illegal-instruction. */
	csr_set(mstatus, MSTATUS_VS_CLEAN);

	/* Diagnostic: record vstart seen on entry (matches deployed
	 * @0x8000734a); a non-zero value means a preempted V op leaked its
	 * partial-completion vstart across the switch. */
	{
		unsigned long ev;

		__asm volatile(".option push\n\t"
			       ".option arch, +v\n\t"
			       "csrr %0, vstart\n\t"
			       ".option pop\n\t"
			       : "=r"(ev));
		z_riscv_v_dbg_restore_entry_vstart = ev;
		if (ev != 0) {
			z_riscv_v_dbg_restore_nonzero++;
		}
	}

	/* Always reset vstart=0 — vsetvli does NOT clear vstart per the
	 * RVV spec, so a leftover non-zero vstart from a preempted V op
	 * on the previous thread would make the next vector op start at
	 * the wrong offset and silently corrupt v0..v(vstart-1). This
	 * runs on every switch regardless of is_dirty. */
	__asm volatile(".option push\n\t"
		       ".option arch, +v\n\t"
		       "csrw vstart, x0\n\t"
		       ".option pop\n\t");

	/* Reload v0..v31 + V CSRs only if this thread has ever been
	 * saved (sticky is_dirty). For fresh K_FP_REGS threads that
	 * never use V (e.g. two gemmini threads time-sharing one hart),
	 * saved_v_context is uninitialised and the vreg pointer ends up
	 * bogus on Saturn — vle8.v then faults with a wild address.
	 * Skip the bulk reload in that case; the next switch back to a
	 * V-using thread will properly restore once it has had a chance
	 * to save its state. */
	if (restore_from->is_dirty) {
		/* MB_V_SAVE_FENCE (restore side) + vstart force + strong drain
		 * ahead of the bulk vle8.v (matches deployed @0x8000739a). */
		__asm volatile("fence rw, rw" ::: "memory");
		z_riscv_force_vstart0();
		z_riscv_v_strong_drain();

		/* V-RACE FIX (local test, saturn-rocc-sresp-reg-realloc): force the bulk vle8.v to use
		 * t-registers (t0=ptr, t1=VLMAX) instead of the compiler's default a4/a5. A co-resident RoCC
		 * accelerator scalar response (FC custom-0 fc_rocc POP rd=a4/a5, or Gemmini counter rd=a5),
		 * backpressured behind the switch's vector traffic, lands DURING this vle8.v; if the base/VLMAX
		 * are a4/a5 the add a5,a5,a4 consumes the strayed resp -> wild vle8.v (mcause 5/2). Using t0/t1
		 * (no accel/div op targets rd=5/6) makes a strayed a4/a5 resp land on DEAD a4/a5 harmlessly --
		 * TIMING-INDEPENDENT (no drain needed; SW drains fail due to wb_wxd backpressure). */
		register unsigned long _vr_ptr asm("t0") = (unsigned long)restore_from->vreg;
		register unsigned long _vr_vl  asm("t1");
		__asm volatile(
			".option push\n\t"
			".option arch, +v\n\t"
			"vsetvli	%0, x0, e8, m8, ta, ma\n\t"
			"vle8.v		v0, (%1)\n\t"
			"add		%1, %1, %0\n\t"
			"vle8.v		v8, (%1)\n\t"
			"add		%1, %1, %0\n\t"
			"vle8.v		v16, (%1)\n\t"
			"add		%1, %1, %0\n\t"
			"vle8.v		v24, (%1)\n\t"
			".option pop\n\t"
			: "=&r"(_vr_vl)
			: "r"(_vr_ptr)
			: "memory");
		z_riscv_vstate_csr_restore(restore_from);
	}

	/* Don't clear is_dirty — make it sticky so once a thread has
	 * saved real V state, every subsequent switch restores it. */

	csr_clear(mstatus, MSTATUS_VS);
	csr_set(mstatus, MSTATUS_VS_CLEAN);
}

void z_riscv_vstate_restore(void)
{
	if (!HAS_V()) {
		return;
	}
	z_riscv_vstate_restore_thread(_current);
}
#endif /* !CONFIG_RISCV_ISA_EXT_V_LAZY */

#ifdef CONFIG_RISCV_ISA_EXT_V_LAZY
/*
 * Lazy V save/restore.  Not exercised by this fork right now — the
 * eager path above is the production path on Saturn — but kept
 * available for builds that want it.  When the V trap dispatch lands
 * (step 5 in zephyr_v_decouple_design.md) these will be called from
 * z_riscv_v_trap and z_riscv_v_load instead of from the FPU trap
 * machinery in fpu.c.
 *
 * NOTE: the FPU-side glue that previously called these (z_riscv_fpu_load,
 * arch_flush_local_fpu, etc.) has been removed.  Until the V trap path
 * is implemented, building with CONFIG_RISCV_ISA_EXT_V_LAZY=y leaves
 * these functions defined but unreferenced.
 */
void z_riscv_vstate_save(struct z_riscv_v_context *save_to)
{
	if (!HAS_V()) {
		return;
	}
	unsigned long vl;

	z_riscv_vstate_csr_save(save_to);
	__asm volatile(".option push\n\t"
		       ".option arch, +v\n\t"
		       "csrw		vstart, x0\n\t"
		       "vsetvli	%0, x0, e8, m8, ta, ma\n\t"
		       "vse8.v		v0, (%1)\n\t"
		       "add		%1, %1, %0\n\t"
		       "vse8.v		v8, (%1)\n\t"
		       "add		%1, %1, %0\n\t"
		       "vse8.v		v16, (%1)\n\t"
		       "add		%1, %1, %0\n\t"
		       "vse8.v		v24, (%1)\n\t"
		       ".option pop\n\t"
		       : "=&r"(vl)
		       : "r"(save_to->vreg)
		       : "memory");
	csr_clear(mstatus, MSTATUS_VS);
	csr_set(mstatus, MSTATUS_VS_CLEAN);
}

void z_riscv_vstate_restore(struct z_riscv_v_context *restore_from)
{
	if (!HAS_V()) {
		return;
	}
	unsigned long vl;

	__asm volatile(".option push\n\t"
		       ".option arch, +v\n\t"
		       "csrw		vstart, x0\n\t"
		       "vsetvli	%0, x0, e8, m8, ta, ma\n\t"
		       "vle8.v		v0, (%1)\n\t"
		       "add		%1, %1, %0\n\t"
		       "vle8.v		v8, (%1)\n\t"
		       "add		%1, %1, %0\n\t"
		       "vle8.v		v16, (%1)\n\t"
		       "add		%1, %1, %0\n\t"
		       "vle8.v		v24, (%1)\n\t"
		       ".option pop\n\t"
		       : "=&r"(vl)
		       : "r"(restore_from->vreg)
		       : "memory");
	z_riscv_vstate_csr_restore(restore_from);
	csr_clear(mstatus, MSTATUS_VS);
	csr_set(mstatus, MSTATUS_VS_CLEAN);
}
#endif /* CONFIG_RISCV_ISA_EXT_V_LAZY */


/* =====================================================================
 * Decoupled lazy V trap path (Saturn fork — CONFIG_RISCV_V_DECOUPLED_LAZY).
 *
 * Mirrors fpu.c's lazy F machinery, but V-only.  When the new Kconfig
 * is on:
 *   - switch.S no longer eagerly saves/restores V on every switch.
 *   - Instead, MSTATUS.VS is left at OFF for any thread that has not
 *     yet exercised V on its current scheduling slot.
 *   - The first V instruction traps as "illegal instruction" with
 *     MSTATUS.VS=OFF; isr.S routes that trap to z_riscv_v_trap, which
 *     loads this thread's V state and sets MSTATUS.VS=CLEAN.
 *   - On context switch, z_riscv_v_thread_context_switch decides
 *     whether to preemptively reload (recently-used hint) or leave
 *     VS=OFF and wait for the trap.
 *
 * Pairing with F:  K_V_REGS today is K_FP_REGS, so threads that opt
 * in to F sharing also have V state allocated.  Vector-float
 * instructions (vfmv.s.f, vfadd.vf, ...) need both MSTATUS.FS != OFF
 * and MSTATUS.VS != OFF; when both are lazy, the first such
 * instruction takes a V trap, then a F trap on retry, then succeeds.
 * Two extra exceptions per thread first-use is negligible.
 *
 * SMP migration / FLUSH_V_IPI: not yet implemented.  Our microros and
 * xpurt configs use SCHED_CPU_MASK_PIN_ONLY=y so threads don't migrate;
 * the same-CPU lazy path below is sufficient for them.  Add
 * arch_flush_v_ipi (mirroring arch_flush_fpu_ipi in ipi_clint.c) when
 * an SMP migration consumer comes along.
 * =====================================================================
 */

#ifdef CONFIG_RISCV_V_DECOUPLED_LAZY

static void z_riscv_v_disable(void)
{
	unsigned long status = csr_read(mstatus);

	__ASSERT((status & MSTATUS_IEN) == 0,
		 "must be called with IRQs disabled");

	if ((status & MSTATUS_VS) != 0) {
		/* Snapshot the dirty/clean state, then flip VS off so a
		 * subsequent V op traps cleanly. */
		_current_cpu->arch.v_state = (status & MSTATUS_VS);
		csr_clear(mstatus, MSTATUS_VS);
	}
}

static void z_riscv_v_load(void)
{
	__ASSERT((csr_read(mstatus) & MSTATUS_IEN) == 0,
		 "must be called with IRQs disabled");
	__ASSERT((csr_read(mstatus) & MSTATUS_VS) == 0,
		 "must be called with V access disabled");

	/* Become the new owner of this hart's V regs. */
	atomic_ptr_set(&_current_cpu->arch.v_owner, _current);

	/* Enable V at INIT, then run the eager-style bulk reload — the
	 * existing z_riscv_vstate_restore_thread carries the vstart=0
	 * fix so we reuse it verbatim. */
	csr_set(mstatus, MSTATUS_VS_INIT);
#ifndef CONFIG_RISCV_ISA_EXT_V_LAZY
	z_riscv_vstate_restore_thread(_current);
#else
	z_riscv_vstate_restore(&_current->arch.saved_v_context);
#endif
}

/*
 * Flush this CPU's V content to memory and clear ownership.  If the
 * saved V state is "clean" (in-memory copy is up to date), skip the
 * actual save; the v_state tracking is updated on every V disable so
 * we know whether memory lags hardware.
 *
 * Called locally and from flush_v_ipi_handler() when that lands.
 */
void arch_flush_local_v(void)
{
	__ASSERT((csr_read(mstatus) & MSTATUS_IEN) == 0,
		 "must be called with IRQs disabled");
	__ASSERT((csr_read(mstatus) & MSTATUS_VS) == 0,
		 "must be called with V access disabled");

	struct k_thread *owner =
		(struct k_thread *)atomic_ptr_get(&_current_cpu->arch.v_owner);

	if (owner != NULL) {
		bool dirty = _current_cpu->arch.v_state == MSTATUS_VS_DIRTY;

		if (dirty) {
			/* Re-enable V to drain the regs out. */
			csr_set(mstatus, MSTATUS_VS_CLEAN);
#ifndef CONFIG_RISCV_ISA_EXT_V_LAZY
			z_riscv_vstate_save_thread(owner);
#else
			z_riscv_vstate_save(&owner->arch.saved_v_context);
#endif
		}

		/* Hint to the next-schedule-in to preemptively reload. */
		owner->arch.v_recently_used = dirty;

		csr_clear(mstatus, MSTATUS_VS);
		atomic_ptr_clear(&_current_cpu->arch.v_owner);
	}
}

#ifdef CONFIG_SMP
/*
 * Locate `thread` on whichever CPU currently owns its V state and pull
 * it back to memory.  Mirrors flush_owned_fpu in fpu.c.  Without an
 * arch_flush_v_ipi this only works for the local CPU (which is enough
 * for SCHED_CPU_MASK_PIN_ONLY=y configs); cross-CPU support is TODO.
 */
static void flush_owned_v(struct k_thread *thread)
{
	__ASSERT((csr_read(mstatus) & MSTATUS_IEN) == 0,
		 "must be called with IRQs disabled");

	int i;
	atomic_ptr_val_t owner;
	unsigned int num_cpus = arch_num_cpus();

	for (i = 0; i < num_cpus; i++) {
		owner = atomic_ptr_get(&_kernel.cpus[i].arch.v_owner);
		if ((struct k_thread *)owner != thread) {
			continue;
		}
		if (i == _current_cpu->id) {
			z_riscv_v_disable();
			arch_flush_local_v();
			break;
		}
		/* Cross-CPU flush would go here — needs FLUSH_V_IPI. */
		break;
	}
}
#endif

void z_riscv_v_enter_exc(void)
{
	/* On entering any exception we deny V access to the trapped
	 * code — same policy as fpu_enter_exc. */
	z_riscv_v_disable();
}

/*
 * V-access trap.  Called from isr.S when an illegal instruction
 * trap fires with opcode = OP-V (0x57) and MSTATUS.VS == OFF.  This
 * means the running thread tried to execute a V instruction without V
 * access; we save the previous owner's V state if any, then load this
 * thread's V state and re-enable VS so the instruction can retry.
 */
void z_riscv_v_trap(struct arch_esf *esf)
{
	__ASSERT((esf->mstatus & MSTATUS_VS) == 0 &&
		 (csr_read(mstatus) & MSTATUS_VS) == 0,
		 "z_riscv_v_trap called despite V being accessible");

	/* Save current owner's V state to its save area. */
	arch_flush_local_v();

	if (_current->arch.exception_depth > 0) {
		/*
		 * V op was executed while already in an exception.  Grant
		 * access to the returning context and disable IRQs so we
		 * don't recurse — the exception's V context is not
		 * preservable across nested IRQs.
		 */
		esf->mstatus &= ~MSTATUS_MPIE_EN;
		esf->mstatus |= MSTATUS_VS_INIT;
		return;
	}

#ifdef CONFIG_SMP
	flush_owned_v(_current);
#endif

	/* Make V accessible+clean to the returning context and load it. */
	esf->mstatus |= MSTATUS_VS_CLEAN;
	z_riscv_v_load();
}

/*
 * Lazy V context switch policy.  Mirror of fpu_access_allowed —
 * decides whether to preemptively grant V access on the way out of
 * a context switch (or exception exit), rather than waiting for the
 * V trap.  Returns true iff V access should be granted.
 */
static bool v_access_allowed(unsigned int exc_update_level)
{
	__ASSERT((csr_read(mstatus) & MSTATUS_IEN) == 0,
		 "must be called with IRQs disabled");

	if (_current->arch.exception_depth == exc_update_level) {
		struct k_thread *owner = (struct k_thread *)atomic_ptr_get(
			&_current_cpu->arch.v_owner);
		if (owner == _current) {
			/* V regs already hold our state. */
			return true;
		}
		if (_current->arch.v_recently_used) {
			/* Hot V user — pre-claim instead of trapping. */
			z_riscv_v_disable();
			arch_flush_local_v();
#ifdef CONFIG_SMP
			flush_owned_v(_current);
#endif
			z_riscv_v_load();
			_current_cpu->arch.v_state = MSTATUS_VS_CLEAN;
			return true;
		}
		return false;
	}
	return false;
}

void z_riscv_v_exit_exc(struct arch_esf *esf)
{
	/* Strip whatever VS bits the trapped frame had; v_access_allowed
	 * decides what to put back. */
	esf->mstatus &= ~MSTATUS_VS;
	if (v_access_allowed(1)) {
		esf->mstatus |= _current_cpu->arch.v_state;
	}
}

void z_riscv_v_thread_context_switch(void)
{
	if (v_access_allowed(0)) {
		csr_clear(mstatus, MSTATUS_VS);
		csr_set(mstatus, _current_cpu->arch.v_state);
	} else {
		z_riscv_v_disable();
	}
}

#endif /* CONFIG_RISCV_V_DECOUPLED_LAZY */

#endif /* CONFIG_RISCV_ISA_EXT_V */
