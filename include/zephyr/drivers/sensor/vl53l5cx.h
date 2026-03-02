/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ZEPHYR_INCLUDE_DRIVERS_SENSOR_VL53L5CX_H_
#define ZEPHYR_INCLUDE_DRIVERS_SENSOR_VL53L5CX_H_

#include <stdint.h>

#include <zephyr/device.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Flattened (8x8) grid view of the most recent ranging frame.
 *
 * The ST ULD supports 4x4 or 8x8. For 4x4, only the first 16 entries are
 * meaningful (row-major), and the remaining entries are set but may not be
 * meaningful depending on the ULD configuration.
 *
 * This helper intentionally exposes only the first target per zone.
 */
struct vl53l5cx_grid {
	uint8_t resolution; /* 16 for 4x4, 64 for 8x8 */
	uint8_t nb_target_detected[64];
	uint8_t target_status[64];
	int16_t distance_mm[64];
};

/**
 * @brief Manually initialize the VL53L5CX at runtime (deferred init).
 */
int vl53l5cx_reinit(const struct device *dev);

/**
 * @brief Update the I2C address of the VL53L5CX.
 *
 * @param dev VL53L5CX device
 * @param new_addr_7bit New 7-bit I2C address (0x29 style)
 */
int vl53l5cx_set_i2c_address_7bit(const struct device *dev, uint8_t new_addr_7bit);

/**
 * @brief Get the latest flattened distance grid.
 *
 * Requires that at least one successful sample has been fetched.
 */
int vl53l5cx_get_grid(const struct device *dev, struct vl53l5cx_grid *out);

#ifdef __cplusplus
}
#endif

#endif /* ZEPHYR_INCLUDE_DRIVERS_SENSOR_VL53L5CX_H_ */

