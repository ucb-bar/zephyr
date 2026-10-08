/*
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Wraps the embedded `.vmfb` byte array (produced at configure time by
 * Zephyr's generate_inc_file_for_target) in an iree_const_byte_span_t.
 *
 * The bare-metal reference uses a linker-script symbol pair
 * (_binary_*_start / _binary_*_end). On Zephyr the cleaner way is to embed
 * via generate_inc_file_for_target, which creates a #include-able file
 * with a `static const uint8_t` array. Both get the same end result: a
 * pointer + length to read-only flash/.rodata.
 */

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"

static const uint8_t mlp_vmfb_data[] = {
#include "mlp_vmfb.inc"
};

iree_const_byte_span_t load_bytecode_module_data(void)
{
	return iree_make_const_byte_span(mlp_vmfb_data, sizeof(mlp_vmfb_data));
}
