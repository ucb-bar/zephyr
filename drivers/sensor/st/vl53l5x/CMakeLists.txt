# SPDX-License-Identifier: Apache-2.0

zephyr_library()

zephyr_library_sources(
  vl53l5x.c
  vl53l5x_platform.c

  st/modules/vl53l5cx_api.c
  st/modules/vl53l5cx_plugin_xtalk.c
  st/modules/vl53l5cx_plugin_detection_thresholds.c
  st/modules/vl53l5cx_plugin_motion_indicator.c

  st/porting/platform.c
)

zephyr_library_include_directories(
  .
  st
  st/modules
  st/porting
)

zephyr_library_compile_definitions(
  VL53L5CX_NB_TARGET_PER_ZONE=CONFIG_VL53L5CX_NB_TARGET_PER_ZONE
)

