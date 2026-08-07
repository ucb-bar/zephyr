/*
 * Copyright (c) 2023 Prosaris SOlutions Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ZEPHYR_INCLUDE_DRIVERS_SENSOR_VL53L1X_H_
#define ZEPHYR_INCLUDE_DRIVERS_SENSOR_VL53L1X_H_

#include <zephyr/device.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Manually initialize the VL53L1X sensor at runtime.
 *
 * This function can be used to initialize the sensor after address
 * reprogramming or power cycling. The sensor will be automatically
 * initialized on first use, but this function allows explicit control
 * over when initialization occurs.
 *
 * @param dev Pointer to the VL53L1X device
 * @return 0 on success, negative error code on failure
 */
int vl53l1x_reinit(const struct device *dev);

/**
 * @brief Run the VL53L1X calibration flow (offset + crosstalk) with a placed target.
 *
 * Place a target at the given distances (mm) in a dark, low-reflection environment, then call.
 * Crosstalk calibration also programs + enables crosstalk compensation. Results persist until the
 * next power cycle (store + reload for production). See datasheet 2.3 / UM2356.
 *
 * @param dev       Pointer to the VL53L1X device
 * @param offset_mm Offset calibration target distance (ST recommends ~140 mm)
 * @param xtalk_mm  Crosstalk calibration target distance
 * @return 0 on success, negative error code on failure
 */
int vl53l1x_calibrate(const struct device *dev, int32_t offset_mm, int32_t xtalk_mm);

#ifdef __cplusplus
}
#endif

#endif /* ZEPHYR_INCLUDE_DRIVERS_SENSOR_VL53L1X_H_ */
