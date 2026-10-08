/*
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Zephyr port of merlin/samples/SaturnOPU/simple_embedding_ukernel/
 * model_benchmark.c. Embeds a `.vmfb` produced by `merlin compile
 * --target zephyr_rocket_rv64` and invokes module.main on a multi-hart
 * Rocket SoC running Zephyr.
 *
 * IREE is linked under IREE_PLATFORM_GENERIC: the runtime is
 * single-threaded with no internal worker pool. Multi-hart parallelism
 * is at the *application* layer -- main() spawns
 * CONFIG_MERLIN_IREE_NUM_WORKERS k_threads, each creating its own
 * iree_vm_context and invoking module.main on a private slice of the
 * batch. With CONFIG_SCHED_CPU_MASK_PIN_ONLY each worker is pinned to a
 * single hart.
 *
 * Differences from the bare-metal reference:
 *   - The .vmfb data is supplied via Zephyr's generate_inc_file_for_target
 *     instead of an objcopy-derived linker symbol pair.
 *   - Stdio writes go to printk, which is wired to the HTIF console driver
 *     in chipyard_riscv64.dts (zephyr,console = &htif).
 *   - Each k_thread maintains its own iree_vm_context (no shared state)
 *     because IREE_PLATFORM_GENERIC has no synchronization.
 *
 * On success the program prints a CSV row of per-worker cycle counts and
 * the aggregate average, then triggers Zephyr shutdown so the HTIF tohost
 * register receives 1 (FireSim runworkload exits 0).
 */

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/reboot.h>
#include <zephyr/sys/util.h>
#include <zephyr/arch/cpu.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#if defined(__has_include)
#if __has_include(<malloc.h>)
#include <malloc.h>
#endif
#endif

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/device_group.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

extern iree_status_t create_sample_device(iree_allocator_t host_allocator,
					  iree_hal_device_t **out_device);
extern iree_const_byte_span_t load_bytecode_module_data(void);

/* --- Hardware cycle counter ------------------------------------------- */

static inline uint64_t read_cycles(void)
{
	uint64_t cycles;
	__asm__ volatile("rdcycle %0" : "=r"(cycles));
	return cycles;
}

/* --- Build-time configuration ---------------------------------------- */

#ifndef MODEL_NAME
#define MODEL_NAME "MLP"
#endif
#ifndef VARIANT_NAME
#define VARIANT_NAME "RVV"
#endif
#ifndef INPUT_NDIMS
#define INPUT_NDIMS 2
#endif
#ifndef INPUT_DIM0
#define INPUT_DIM0 1
#endif
#ifndef INPUT_DIM1
#define INPUT_DIM1 784
#endif
#ifndef INPUT_DIM2
#define INPUT_DIM2 1
#endif
#ifndef INPUT_DIM3
#define INPUT_DIM3 1
#endif
#ifndef NUM_INPUTS
#define NUM_INPUTS 1
#endif
#ifndef FUNC_NAME
#define FUNC_NAME "module.main"
#endif
#ifndef BENCH_WARMUP_ITERS
#define BENCH_WARMUP_ITERS 2
#endif
#ifndef BENCH_MEASURE_ITERS
#define BENCH_MEASURE_ITERS 5
#endif

#define MAX_WORKERS 16
#if CONFIG_MERLIN_IREE_NUM_WORKERS > MAX_WORKERS
#error "CONFIG_MERLIN_IREE_NUM_WORKERS exceeds MAX_WORKERS"
#endif

#define WORKER_STACK_SIZE 16384

K_THREAD_STACK_ARRAY_DEFINE(worker_stacks, MAX_WORKERS, WORKER_STACK_SIZE);
static struct k_thread worker_threads[MAX_WORKERS];

/* --- Per-worker state ------------------------------------------------- */

struct worker_ctx {
	int worker_id;
	uint64_t cycles[BENCH_MEASURE_ITERS];
	uint64_t avg_cycles;
	int rc; /* 0 ok, nonzero = failure */
	float head_out[8]; /* first 8 elements of worker 0's output */
	int head_valid;
};

static void *zephyr_aligned_alloc(size_t alignment, size_t size)
{
	return memalign(alignment, size);
}

static void shutdown_with_status(int code)
{
	printk("[exit] tohost <- %d\n", code);
	sys_reboot(code == 0 ? SYS_REBOOT_COLD : SYS_REBOOT_WARM);
	for (;;) {
		k_cpu_idle();
	}
}

/* --- Worker entry: full IREE setup + invoke loop --------------------- */

static iree_status_t run_worker(struct worker_ctx *wctx)
{
	iree_vm_instance_t *instance = NULL;
	IREE_RETURN_IF_ERROR(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
						     iree_allocator_system(),
						     &instance));
	IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(instance));

	iree_hal_device_t *device = NULL;
	IREE_RETURN_IF_ERROR(create_sample_device(iree_allocator_system(),
						  &device));

	iree_hal_device_group_t *device_group = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_device_group_create_from_device(
		device, iree_allocator_system(), &device_group));

	iree_vm_module_t *hal_module = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_module_create(
		instance, iree_hal_module_device_policy_default(),
		device_group, IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
		iree_hal_module_debug_sink_stdio(stderr),
		iree_allocator_system(), &hal_module));
	iree_hal_device_group_release(device_group);

	const iree_const_byte_span_t module_data = load_bytecode_module_data();
	iree_vm_module_t *bytecode_module = NULL;
	IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
		instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE, module_data,
		iree_allocator_null(), iree_allocator_system(),
		&bytecode_module));

	iree_vm_context_t *context = NULL;
	iree_vm_module_t *modules[] = {hal_module, bytecode_module};
	IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
		instance, IREE_VM_CONTEXT_FLAG_NONE, ARRAY_SIZE(modules), modules,
		iree_allocator_system(), &context));
	iree_vm_module_release(hal_module);
	iree_vm_module_release(bytecode_module);

	iree_vm_function_t main_function;
	IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
		context, iree_make_cstring_view(FUNC_NAME), &main_function));

	iree_hal_dim_t shape[4] = {INPUT_DIM0, INPUT_DIM1, INPUT_DIM2, INPUT_DIM3};
	const int ndims = INPUT_NDIMS;

	iree_host_size_t num_elements = 1;
	for (int d = 0; d < ndims; ++d) {
		num_elements *= (iree_host_size_t)shape[d];
	}

	iree_hal_element_type_t elem_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
	iree_host_size_t elem_size = sizeof(float);

	iree_hal_buffer_params_t params = {
		.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
		.usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
	};

	iree_vm_list_t *inputs = NULL;
	IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
						 NUM_INPUTS,
						 iree_allocator_system(),
						 &inputs));

	for (int inp = 0; inp < NUM_INPUTS; ++inp) {
		size_t input_bytes = num_elements * elem_size;
		size_t input_bytes_aligned = (input_bytes + 127) & ~((size_t)127);
		void *input_data = zephyr_aligned_alloc(128, input_bytes_aligned);
		if (input_data == NULL) {
			return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
						"memalign(128, %zu) failed",
						input_bytes_aligned);
		}
		memset(input_data, 0, input_bytes);
		float *f32_data = (float *)input_data;
		for (iree_host_size_t i = 0; i < num_elements; ++i) {
			/* Per-worker offset so each worker exercises a different
			 * input slice of the batch (proves the contexts are
			 * independent and not stomping each other). */
			int seed = (int)(i + (iree_host_size_t)wctx->worker_id);
			f32_data[i] = (float)(seed % 256) / 256.0f;
		}
		iree_hal_buffer_view_t *bv = NULL;
		IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
			device, iree_hal_device_allocator(device), ndims, shape,
			elem_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
			iree_make_const_byte_span(input_data,
						  num_elements * elem_size),
			&bv));
		free(input_data);
		iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(bv);
		IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &ref));
	}

	iree_vm_list_t *outputs = NULL;
	IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
						 1, iree_allocator_system(),
						 &outputs));

	const int kWarmup = BENCH_WARMUP_ITERS;
	const int kIters = BENCH_MEASURE_ITERS;

	for (int i = 0; i < kWarmup; ++i) {
		iree_vm_list_resize(outputs, 0);
		iree_status_t s = iree_vm_invoke(
			context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
			NULL, inputs, outputs, iree_allocator_system());
		if (!iree_status_is_ok(s)) {
			printk("[w%d] warmup invoke FAILED at iter %d\n",
			       wctx->worker_id, i + 1);
			iree_status_fprint(stderr, s);
			iree_status_free(s);
			return iree_make_status(IREE_STATUS_INTERNAL,
						"warmup invoke failed");
		}
	}

	uint64_t sum = 0;
	for (int i = 0; i < kIters; ++i) {
		uint64_t it_start = read_cycles();
		iree_vm_list_resize(outputs, 0);
		IREE_RETURN_IF_ERROR(iree_vm_invoke(
			context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
			NULL, inputs, outputs, iree_allocator_system()));
		wctx->cycles[i] = read_cycles() - it_start;
		sum += wctx->cycles[i];
	}
	wctx->avg_cycles = sum / (uint64_t)kIters;

	/* Worker 0 dumps its first 8 output values for downstream
	 * cross-checking against the host reference. */
	if (wctx->worker_id == 0 && iree_vm_list_size(outputs) > 0) {
		iree_vm_ref_t out_ref = iree_vm_ref_null();
		iree_vm_list_get_ref_assign(outputs, 0, &out_ref);
		iree_hal_buffer_view_t *out_bv =
			iree_hal_buffer_view_deref(out_ref);
		if (out_bv != NULL) {
			iree_device_size_t byte_len =
				iree_hal_buffer_view_byte_length(out_bv);
			iree_host_size_t copy =
				MIN((iree_host_size_t)sizeof(wctx->head_out),
				    byte_len);
			iree_status_t cs = iree_hal_device_transfer_d2h(
				device, iree_hal_buffer_view_buffer(out_bv), 0,
				wctx->head_out, copy,
				IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
				iree_infinite_timeout());
			if (iree_status_is_ok(cs)) {
				wctx->head_valid = 1;
			} else {
				iree_status_free(cs);
			}
		}
	}

	iree_vm_list_release(inputs);
	iree_vm_list_release(outputs);
	iree_hal_device_release(device);
	iree_vm_context_release(context);
	iree_vm_instance_release(instance);

	return iree_ok_status();
}

static void worker_entry(void *p1, void *p2, void *p3)
{
	ARG_UNUSED(p2);
	ARG_UNUSED(p3);
	struct worker_ctx *wctx = (struct worker_ctx *)p1;

	iree_status_t s = run_worker(wctx);
	if (!iree_status_is_ok(s)) {
		iree_status_fprint(stderr, s);
		iree_status_free(s);
		wctx->rc = 1;
	} else {
		wctx->rc = 0;
	}
}

/* --- Main: spawn workers, join, summarize ---------------------------- */

int main(void)
{
	const int num_workers = CONFIG_MERLIN_IREE_NUM_WORKERS;
	const int harts = arch_num_cpus();

	printk("\n[merlin] Zephyr × Merlin × FireSim model benchmark\n");
	printk("[merlin] Model=%s Variant=%s NUM_WORKERS=%d harts=%d\n",
	       MODEL_NAME, VARIANT_NAME, num_workers, harts);
	printk("[merlin] Input shape: %dx%dx%dx%d ndims=%d num_inputs=%d\n",
	       INPUT_DIM0, INPUT_DIM1, INPUT_DIM2, INPUT_DIM3, INPUT_NDIMS,
	       NUM_INPUTS);

	static struct worker_ctx ctxs[MAX_WORKERS];
	memset(ctxs, 0, sizeof(ctxs));

	for (int i = 0; i < num_workers; ++i) {
		ctxs[i].worker_id = i;
		k_tid_t tid = k_thread_create(
			&worker_threads[i], worker_stacks[i],
			K_THREAD_STACK_SIZEOF(worker_stacks[i]), worker_entry,
			&ctxs[i], NULL, NULL,
			K_PRIO_PREEMPT(7), 0, K_NO_WAIT);
#ifdef CONFIG_SCHED_CPU_MASK_PIN_ONLY
		/* Pin worker i to hart i so the cycle counter we sample is
		 * the local hart's, not whoever last scheduled this thread. */
		if (i < harts) {
			k_thread_cpu_pin(tid, i);
		}
#else
		ARG_UNUSED(tid);
#endif
		char name[16];
		snprintf(name, sizeof(name), "merlin-w%d", i);
		k_thread_name_set(&worker_threads[i], name);
	}

	int any_fail = 0;
	for (int i = 0; i < num_workers; ++i) {
		k_thread_join(&worker_threads[i], K_FOREVER);
		if (ctxs[i].rc != 0) {
			any_fail = 1;
		}
	}

	if (any_fail) {
		printk("[merlin] one or more workers failed\n");
		shutdown_with_status(1);
	}

	/* Per-worker CSV rows then an aggregate row averaged across workers.
	 * Format intentionally matches the bare-metal model_benchmark.c so
	 * existing CSV scrapers handle both. */
	uint64_t agg_sum = 0;
	for (int i = 0; i < num_workers; ++i) {
		printk("CSV, %s, %s, %" PRIu64 ", %" PRIu64 ", %" PRIu64
		       ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "\n",
		       MODEL_NAME, VARIANT_NAME, ctxs[i].cycles[0],
		       ctxs[i].cycles[1], ctxs[i].cycles[2], ctxs[i].cycles[3],
		       ctxs[i].cycles[4], ctxs[i].avg_cycles);
		agg_sum += ctxs[i].avg_cycles;
	}
	uint64_t agg_avg = agg_sum / (uint64_t)num_workers;
	printk("CSV-AGG, %s, %s, workers=%d, avg=%" PRIu64 "\n", MODEL_NAME,
	       VARIANT_NAME, num_workers, agg_avg);

	if (ctxs[0].head_valid) {
		printk("OUT[0..7]: %f %f %f %f %f %f %f %f\n",
		       (double)ctxs[0].head_out[0], (double)ctxs[0].head_out[1],
		       (double)ctxs[0].head_out[2], (double)ctxs[0].head_out[3],
		       (double)ctxs[0].head_out[4], (double)ctxs[0].head_out[5],
		       (double)ctxs[0].head_out[6], (double)ctxs[0].head_out[7]);
	}

	printk("[merlin] benchmark OK\n");
	shutdown_with_status(0);
	return 0;
}
