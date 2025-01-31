# # SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(xnnpack)

# Include XNNPack
set(XNNPACK_DIR ${CMAKE_SOURCE_DIR}/../../../modules/xnnpack)
add_subdirectory(${XNNPACK_DIR} xnnpack_build)

# Link XNNPack to the Zephyr application
target_link_libraries(app PRIVATE XNNPACK pthreadpool cpuinfo)

include_directories(${XNNPACK_DIR}/include)

# Disable JIT (not needed on RISC-V)
set(XNN_ENABLE_JIT OFF CACHE BOOL "Disable JIT compilation")

# Enable threading for XNNPack
set(XNN_ENABLE_THREADPOOL ON CACHE BOOL "Enable multithreading in XNNPack")

# Set CPU info (minimal for RISC-V, will likely stub it later)
set(XNNPACK_USE_SYSTEM_LIBS OFF CACHE BOOL "Disable system libraries for CPU detection")

# Disable tests to skip building googletest and googlebenchmark
set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "Disable building XNNPack tests")
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "Disable building XNNPack tests")

# set(XNNPACK_LIBRARY_TYPE "static" CACHE STRING "Build a Static Library")
# set(XNNPACK_BUILD_LIBRARY ON CACHE BOOL "Build XNNPack as a library")

target_sources(app PRIVATE src/main.c)

# SPDX-License-Identifier: Apache-2.0

# # SPDX-License-Identifier: Apache-2.0

# cmake_minimum_required(VERSION 3.20.0)

# find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
# project(xnnpack)

# # Include XNNPack
# set(XNNPACK_DIR ${CMAKE_SOURCE_DIR}/../../../modules/xnnpack)

# # Ensure dependencies (cpuinfo, pthreadpool, etc.) are available
# # set(CPUINFO_SOURCE_DIR "${CMAKE_BINARY_DIR}/cpuinfo-source" CACHE STRING "cpuinfo source directory")
# # set(PTHREADPOOL_SOURCE_DIR "${CMAKE_BINARY_DIR}/pthreadpool-source" CACHE STRING "pthreadpool source directory")
# # set(FXDIV_SOURCE_DIR "${CMAKE_BINARY_DIR}/FXdiv-source" CACHE STRING "FXdiv source directory")

# # Prevent XNNPack from building a standalone library, just collect its sources
# set(XNNPACK_BUILD_LIBRARY OFF CACHE BOOL "Disable XNNPack standalone build")
# add_subdirectory(${XNNPACK_DIR} xnnpack_build)

# # Ensure XNNPack's source variables are defined
# if(NOT DEFINED OPERATOR_SRCS OR NOT DEFINED REFERENCE_SRCS OR NOT DEFINED SUBGRAPH_SRCS OR
#    NOT DEFINED LOGGING_SRCS OR NOT DEFINED XNNPACK_SRCS OR NOT DEFINED TABLE_SRCS)
#     message(FATAL_ERROR "XNNPack source variables are not defined! Make sure XNNPack's CMakeLists.txt is processed first.")
# endif()

# # Collect all XNNPack sources
# set(XNNPACK_ALL_SRCS
#     ${OPERATOR_SRCS}
#     ${REFERENCE_SRCS}
#     ${SUBGRAPH_SRCS}
#     ${LOGGING_SRCS}
#     ${XNNPACK_SRCS}
#     ${TABLE_SRCS}
# )

# # Include required directories
# include_directories(
#     ${XNNPACK_DIR}/include
#     ${XNNPACK_DIR}/src
#     ${CPUINFO_SOURCE_DIR}/include
#     ${PTHREADPOOL_SOURCE_DIR}/include
# )

# # Ensure Zephyr compiles XNNPack sources
# target_sources(app PRIVATE src/main.c ${XNNPACK_ALL_SRCS})

# # Link required dependencies
# target_link_libraries(app PRIVATE pthreadpool cpuinfo)
# # target_link_libraries(app PRIVATE)