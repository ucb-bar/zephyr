/*
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Zephyr-side `create_sample_device` for the Merlin model benchmark.
 *
 * IREE is linked under IREE_PLATFORM_GENERIC, so the only HAL driver
 * available is local-sync (single-threaded inline executor). Multi-hart
 * parallelism is provided at the app layer by spawning multiple k_threads,
 * each calling create_sample_device + iree_vm_invoke independently. Each
 * hart owns its own iree_vm_context.
 *
 * Layout follows merlin/samples/SaturnOPU/simple_embedding_ukernel/
 * device_embedded_sync.c so signatures stay in sync with the underlying
 * iree_bar fork (which owns iree_hal_device_create_params_t,
 * iree_async_proactor_pool_*, etc.).
 */

#include <stddef.h>
#include <zephyr/kernel.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/device.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/embedded_elf_loader.h"

iree_status_t create_sample_device(iree_allocator_t host_allocator,
				   iree_hal_device_t **out_device)
{
	iree_hal_sync_device_params_t params;
	iree_hal_sync_device_params_initialize(&params);

	iree_hal_executable_loader_t *loader = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_embedded_elf_loader_create(
		/*plugin_manager=*/NULL, host_allocator, &loader));

	iree_string_view_t identifier = iree_make_cstring_view("local-sync");
	iree_hal_allocator_t *device_allocator = NULL;
	iree_status_t status = iree_hal_allocator_create_heap(
		identifier, host_allocator, host_allocator, &device_allocator);

	iree_async_proactor_pool_t *proactor_pool = NULL;
	if (iree_status_is_ok(status)) {
		status = iree_async_proactor_pool_create(
			/*node_count=*/1, /*node_ids=*/NULL,
			iree_async_proactor_pool_options_default(),
			host_allocator, &proactor_pool);
	}

	iree_hal_device_create_params_t create_params =
		iree_hal_device_create_params_default();
	create_params.proactor_pool = proactor_pool;
	if (iree_status_is_ok(status)) {
		status = iree_hal_sync_device_create(
			identifier, &params, &create_params,
			/*loader_count=*/1, &loader, device_allocator,
			host_allocator, out_device);
	}
	iree_async_proactor_pool_release(proactor_pool);
	iree_hal_allocator_release(device_allocator);
	iree_hal_executable_loader_release(loader);
	return status;
}

/* Bare-metal stubs.
 *
 * Under IREE_PLATFORM_GENERIC (Zephyr profile + chipyard_riscv64):
 *   - IREE_ENABLE_THREADING=OFF, but iree_thread_create/release symbols
 *     are still referenced by transitive archive deps -> stub them.
 *   - iree/async/proactor_platform.c always falls through to
 *     iree_async_proactor_create_posix() in the
 *     `#elif !defined(IREE_PLATFORM_EMSCRIPTEN)` branch (since GENERIC
 *     defines neither WINDOWS nor EMSCRIPTEN). The local-sync HAL never
 *     actually instantiates a working platform proactor (it queues work
 *     inline), so we stub create_posix to return UNAVAILABLE and let the
 *     caller fall through to single-NUMA-node defaults.
 *   - iree_async_proactor_thread_{create,join,release,request_stop} are
 *     similarly never reached on the local-sync hot path. Stub them all.
 */
#if defined(IREE_PLATFORM_GENERIC)
#include "iree/base/threading/thread.h"
#include "iree/async/proactor.h"
#include "iree/async/util/proactor_thread.h"

iree_status_t iree_thread_create(iree_thread_entry_t entry, void *entry_arg,
				 iree_thread_create_params_t params,
				 iree_allocator_t allocator,
				 iree_thread_t **out_thread)
{
	(void)entry;
	(void)entry_arg;
	(void)params;
	(void)allocator;
	*out_thread = NULL;
	return iree_ok_status();
}
void iree_thread_release(iree_thread_t *thread)
{
	(void)thread;
}

iree_status_t iree_async_proactor_create_posix(
	iree_async_proactor_options_t options, iree_allocator_t allocator,
	iree_async_proactor_t **out_proactor)
{
	(void)options;
	(void)allocator;
	*out_proactor = NULL;
	return iree_status_from_code(IREE_STATUS_UNAVAILABLE);
}

iree_status_t iree_async_proactor_thread_create(
	iree_async_proactor_t *proactor,
	iree_async_proactor_thread_options_t options,
	iree_allocator_t allocator,
	iree_async_proactor_thread_t **out_thread)
{
	(void)proactor;
	(void)options;
	(void)allocator;
	*out_thread = NULL;
	return iree_status_from_code(IREE_STATUS_UNAVAILABLE);
}

iree_status_t iree_async_proactor_thread_join(
	iree_async_proactor_thread_t *thread, iree_duration_t timeout)
{
	(void)thread;
	(void)timeout;
	return iree_ok_status();
}
void iree_async_proactor_thread_release(iree_async_proactor_thread_t *thread)
{
	(void)thread;
}
void iree_async_proactor_thread_request_stop(
	iree_async_proactor_thread_t *thread)
{
	(void)thread;
}
#endif
