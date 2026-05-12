/*
 * Copyright (c) BayLibre SAS
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ZEPHYR_INCLUDE_RISCV_STRUCTS_H_
#define ZEPHYR_INCLUDE_RISCV_STRUCTS_H_

/* Per CPU architecture specifics */
struct _cpu_arch {
#ifdef CONFIG_USERSPACE
	unsigned long user_exc_sp;
	unsigned long user_exc_tmp0;
	unsigned long user_exc_tmp1;
#endif
#if defined(CONFIG_SMP) || (CONFIG_MP_MAX_NUM_CPUS > 1)
	unsigned long hartid;
	bool online;
#endif
#ifdef CONFIG_FPU_SHARING
	atomic_ptr_val_t fpu_owner;
	uint32_t fpu_state;
#ifdef CONFIG_RISCV_ISA_EXT_V
#ifdef CONFIG_RISCV_ISA_EXT_V_LAZY
	uint32_t vpu_state;
#endif
#endif
#endif
#ifdef CONFIG_RISCV_V_DECOUPLED_LAZY
	/* Decoupled-lazy-V per-CPU ownership tracking — parallel to
	 * fpu_owner / fpu_state, lives independently of FPU_SHARING.
	 * v_owner: thread whose V state is currently live in this hart's
	 * vector registers.  NULL if none.
	 * v_state: snapshot of MSTATUS.VS bits (VS_INIT/CLEAN/DIRTY) at
	 * the time V was disabled, used to restore the right state on
	 * the way back into the owner thread. */
	atomic_ptr_val_t v_owner;
	uint32_t v_state;
#endif
};

#endif /* ZEPHYR_INCLUDE_RISCV_STRUCTS_H_ */
