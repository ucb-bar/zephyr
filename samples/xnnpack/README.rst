.. zephyr:code-sample:: xnnpack
   :name: XNNPack Example

   Execute a linear layer using the XNNPack library


Overview
********

A XNNPack library example that can be used with spike or QEMU riscv64
targets. 

Building and Running
********************

This application can be built and executed on QEMU as follows:

.. zephyr-app-commands::
   :zephyr-app: samples/hello_world
   :host-os: unix
   :board: qemu_riscv64
   :goals: run
   :compact:

To build for another board, change "qemu_riscv64" above to that board's name.

Sample Output
=============

.. code-block:: console

   TODO 

