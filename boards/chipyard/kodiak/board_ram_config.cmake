# SPDX-License-Identifier: Apache-2.0
# This file is included after Kconfig is processed to generate RAM config header

# Generate a header file with RAM configuration based on Kconfig
# This header will be included by the DTS file
set(RAM_CONFIG_HEADER ${CMAKE_BINARY_DIR}/include/generated/ram_config.h)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/generated)

if(CONFIG_CHIPYARD_KODIAK_RAM_NONCOHERENT_SPAD)
  file(WRITE ${RAM_CONFIG_HEADER} "#define CHIPYARD_KODIAK_RAM_NONCOHERENT_SPAD 1\n")
elseif(CONFIG_CHIPYARD_KODIAK_RAM_COHERENT_SPAD)
  file(WRITE ${RAM_CONFIG_HEADER} "#define CHIPYARD_KODIAK_RAM_COHERENT_SPAD 1\n")
elseif(CONFIG_CHIPYARD_KODIAK_RAM_BACKING_DRAM)
  file(WRITE ${RAM_CONFIG_HEADER} "#define CHIPYARD_KODIAK_RAM_BACKING_DRAM 1\n")
endif()

# Add the generated include directory to DTS preprocessor include path
# This needs to be done via DTS_ROOT or similar mechanism
get_filename_component(GENERATED_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include/generated ABSOLUTE)
list(APPEND DTS_ROOT ${GENERATED_INCLUDE_DIR})
