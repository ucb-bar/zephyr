Merlin / IREE model benchmark
#############################

Overview
********

This sample embeds an IREE bytecode module (``.vmfb``) compiled by
``merlin compile --target zephyr_rocket_rv64`` and invokes its entry
function under the IREE ``local-sync`` HAL on the ``chipyard_riscv64``
board. Multi-hart parallelism is at the *application* layer:
``main()`` spawns ``CONFIG_MERLIN_IREE_NUM_WORKERS`` k_threads, each
with its own ``iree_vm_context``, each pinned to a distinct hart.

It is the end-to-end validation point for the Merlin × Zephyr × FireSim
integration:

* Compiler: Merlin / IREE producing ``riscv64-unknown-elf`` bytecode with
  RVV (``+zvl128b``).
* Runtime: IREE static archives cross-compiled by
  ``./merlin build --profile zephyr`` (under ``IREE_PLATFORM_GENERIC``,
  single-thread local-sync HAL) and linked via the ``merlin-iree`` Zephyr
  module. **IREE itself is unmodified.**
* RTOS: Zephyr 4.1.99 with SMP enabled. Multi-hart parallelism comes from
  the application: ``main()`` spawns ``CONFIG_MERLIN_IREE_NUM_WORKERS``
  ``k_threads``, each with its own ``iree_vm_context``, each pinned to a
  hart via ``k_thread_cpu_pin``.
* Target: ``FireSimQuadRocketConfig`` (4 Rocket harts, RV64GC + RVV).

Building
********

The Merlin install path (``$MERLIN_ROOT``) and the Chipyard root
(``$CHIPYARD_ROOT``) must both be exported.

.. code-block:: bash

   # 1. Compile the model to a .vmfb (one-time per model).
   ./merlin compile models/mlp/mlp.q.int8.mlir \
       --target zephyr_rocket_rv64 \
       --output $CHIPYARD_ROOT/software/zephyrproject/zephyr/samples/merlin/model_benchmark/data/mlp.vmfb

   # 2. Cross-build the IREE runtime against the Zephyr toolchain.
   ./merlin build --profile zephyr --config release

   # 3. Build the Zephyr application.
   west build -b chipyard_riscv64 \
       $CHIPYARD_ROOT/software/zephyrproject/zephyr/samples/merlin/model_benchmark

Running on FireSim
******************

.. code-block:: bash

   ./merlin chipyard stage-zephyr-workload zephyr-quad-rocket \
       --elf build/zephyr/zephyr.elf
   cd $CHIPYARD_ROOT/sims/firesim/deploy
   firesim infrasetup
   firesim runworkload

Expected uartlog tail::

   [merlin] Zephyr × Merlin × FireSim model benchmark
   [merlin] Model=MLP Variant=RVV NUM_WORKERS=4 harts=4
   [merlin] Input shape: 1x784x1x1 ndims=2 num_inputs=1
   CSV, MLP, RVV, <c0>, <c1>, <c2>, <c3>, <c4>, <avg>      (worker 0)
   CSV, MLP, RVV, <c0>, <c1>, <c2>, <c3>, <c4>, <avg>      (worker 1)
   CSV, MLP, RVV, <c0>, <c1>, <c2>, <c3>, <c4>, <avg>      (worker 2)
   CSV, MLP, RVV, <c0>, <c1>, <c2>, <c3>, <c4>, <avg>      (worker 3)
   CSV-AGG, MLP, RVV, workers=4, avg=<aggregate average>
   OUT[0..7]: <8 floats>
   [merlin] benchmark OK
   [exit] tohost <- 0
