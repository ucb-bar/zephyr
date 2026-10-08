# SPDX-License-Identifier: Apache-2.0
#
# Zephyr sample: Merlin / IREE model benchmark on chipyard_riscv64.
#
# Embeds a `.vmfb` bytecode module (produced by `merlin compile`) into the
# application and invokes its entry function under the IREE local-sync HAL.
# IREE itself runs single-threaded; multi-hart parallelism is at the app
# layer (one k_thread per worker, each with its own iree_vm_context).
#
# Build:
#   ZEPHYR_EXTRA_MODULES=$CHIPYARD_ROOT/software/zephyrproject/modules/merlin-iree \
#   MERLIN_ROOT=$MERLIN_ROOT \
#   west build -b chipyard_riscv64 \
#     $CHIPYARD_ROOT/software/zephyrproject/zephyr/samples/merlin/model_benchmark
#
# By default the sample expects a pre-built `data/mlp.vmfb` produced by
#   ./merlin compile models/mlp/mlp.q.int8.mlir \
#       --target zephyr_rocket_rv64 \
#       --output samples/merlin/model_benchmark/data/mlp.vmfb
# The path can be overridden with -DMERLIN_VMFB=/path/to/foo.vmfb.

cmake_minimum_required(VERSION 3.20.0)

# Default to the chipyard_riscv64 board if the user did not pass -b explicitly.
if(NOT DEFINED BOARD)
  set(BOARD chipyard_riscv64)
endif()

# Pull in the merlin-iree out-of-tree module unless the user already wired it
# in via west.yml or ZEPHYR_EXTRA_MODULES.
if(NOT DEFINED ZEPHYR_EXTRA_MODULES AND DEFINED ENV{CHIPYARD_ROOT})
  list(APPEND ZEPHYR_EXTRA_MODULES
    "$ENV{CHIPYARD_ROOT}/software/zephyrproject/modules/merlin-iree")
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(merlin_model_benchmark LANGUAGES C)

# --- Resolve the .vmfb to embed -------------------------------------------

if(NOT DEFINED MERLIN_VMFB)
  set(MERLIN_VMFB "${CMAKE_CURRENT_SOURCE_DIR}/data/mlp.vmfb")
endif()

if(NOT EXISTS "${MERLIN_VMFB}")
  message(WARNING
    "merlin sample: ${MERLIN_VMFB} is missing.\n"
    "  Run `./merlin compile models/mlp/mlp.q.int8.mlir "
    "--target zephyr_rocket_rv64 --output ${MERLIN_VMFB}`\n"
    "  before building, or pass -DMERLIN_VMFB=/path/to/foo.vmfb.")
endif()

# Generate a header (`mlp_vmfb.inc`) containing a `static const uint8_t
# mlp_vmfb_data[]` byte array. Zephyr's `generate_inc_file_for_target`
# produces a single `#include`-able file from any binary input.
set(_vmfb_inc "${ZEPHYR_BINARY_DIR}/include/generated/mlp_vmfb.inc")
generate_inc_file_for_target(app "${MERLIN_VMFB}" "${_vmfb_inc}")

# --- Application sources --------------------------------------------------

target_sources(app PRIVATE
  src/main.c
  src/device_create.c
  src/bytecode_module_data.c
)

target_include_directories(app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${ZEPHYR_BINARY_DIR}/include/generated
)

target_link_libraries(app PRIVATE merlin_iree)

# Force-define _RETARGETABLE_LOCKING for libc-hooks.c. The riscv-tools
# newlib in $CHIPYARD_ROOT/.conda-env was built without retargetable
# locking (sys/lock.h falls through to dummy-lock typedefs), but
# Zephyr's libc-hooks.c carries a BUILD_ASSERT that fails without it.
# Asserting it here lies to the assert: the IREE archives keep using
# newlib's dummy locks, while libc-hooks.c emits the
# __retarget_lock_*-flavored locking that Zephyr does provide. Both
# views coexist because we never call newlib functions across the
# dual locking paths simultaneously (single-process app).
zephyr_compile_definitions(_RETARGETABLE_LOCKING=1)

# Configurable model metadata. Each is only forwarded as a compile define
# if the user passed it via -D...; otherwise main.c's #ifndef defaults
# (MLP / RVV / 1x784) take effect.
foreach(_var
    MODEL_NAME VARIANT_NAME FUNC_NAME
    INPUT_NDIMS INPUT_DIM0 INPUT_DIM1 INPUT_DIM2 INPUT_DIM3 NUM_INPUTS
    BENCH_WARMUP_ITERS BENCH_MEASURE_ITERS)
  if(DEFINED ${_var} AND NOT "${${_var}}" STREQUAL "")
    if("${_var}" MATCHES "_NAME$")
      target_compile_definitions(app PRIVATE ${_var}="${${_var}}")
    else()
      target_compile_definitions(app PRIVATE ${_var}=${${_var}})
    endif()
  endif()
endforeach()

# Emit a friendly cmake-time summary.
message(STATUS "merlin sample: vmfb = ${MERLIN_VMFB}")
