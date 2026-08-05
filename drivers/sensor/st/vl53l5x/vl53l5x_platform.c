/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Zephyr platform glue for ST VL53L5CX ULD.
 */

#include <zephyr/drivers/i2c.h>
#include <zephyr/kernel.h>

#include <string.h>

#include "platform.h"

int32_t vl53l5x_platform_get_tick(void)
{
	return (int32_t)k_uptime_get_32();
}

static inline uint16_t vl53l5x_addr8_to_7(uint16_t addr8)
{
	return (uint16_t)(addr8 >> 1);
}

static int vl53l5x_i2c_write_reg16_chunked(struct i2c_dt_spec *spec,
					   uint16_t addr7,
					   uint16_t reg,
					   const uint8_t *buf,
					   uint16_t len)
{
	uint16_t offset = 0U;
	uint8_t tx[2 + CONFIG_VL53L5CX_I2C_CHUNK_SIZE];

	while (offset < len) {
		uint16_t chunk = len - offset;
		if (chunk > CONFIG_VL53L5CX_I2C_CHUNK_SIZE) {
			chunk = CONFIG_VL53L5CX_I2C_CHUNK_SIZE;
		}

		uint16_t r = reg + offset;
		tx[0] = (uint8_t)(r >> 8);
		tx[1] = (uint8_t)(r & 0xFF);
		memcpy(&tx[2], &buf[offset], chunk);

		spec->addr = (uint16_t)addr7;
		int ret = i2c_write_dt(spec, tx, (uint32_t)(2U + chunk));
		if (ret != 0) {
			return ret;
		}

		offset = (uint16_t)(offset + chunk);
	}

	return 0;
}

static int vl53l5x_i2c_read_reg16_chunked(struct i2c_dt_spec *spec,
					  uint16_t addr7,
					  uint16_t reg,
					  uint8_t *buf,
					  uint16_t len)
{
	uint16_t offset = 0U;
	uint8_t regbuf[2];

	while (offset < len) {
		uint16_t chunk = len - offset;
		if (chunk > CONFIG_VL53L5CX_I2C_CHUNK_SIZE) {
			chunk = CONFIG_VL53L5CX_I2C_CHUNK_SIZE;
		}

		uint16_t r = reg + offset;
		regbuf[0] = (uint8_t)(r >> 8);
		regbuf[1] = (uint8_t)(r & 0xFF);

		spec->addr = (uint16_t)addr7;
		int ret = i2c_write_read_dt(spec, regbuf, sizeof(regbuf), &buf[offset], chunk);
		if (ret != 0) {
			return ret;
		}

		offset = (uint16_t)(offset + chunk);
	}

	return 0;
}

int32_t vl53l5x_platform_write(VL53L5CX_Platform *p, uint16_t reg, uint8_t *buf, uint16_t len)
{
	struct i2c_dt_spec *spec = (struct i2c_dt_spec *)p->user_data;
	uint16_t addr7 = vl53l5x_addr8_to_7(p->address);
	int ret = vl53l5x_i2c_write_reg16_chunked(spec, addr7, reg, buf, len);
	p->last_error = ret;
	return (ret == 0) ? 0 : 1;
}

int32_t vl53l5x_platform_read(VL53L5CX_Platform *p, uint16_t reg, uint8_t *buf, uint16_t len)
{
	struct i2c_dt_spec *spec = (struct i2c_dt_spec *)p->user_data;
	uint16_t addr7 = vl53l5x_addr8_to_7(p->address);
	int ret = vl53l5x_i2c_read_reg16_chunked(spec, addr7, reg, buf, len);
	p->last_error = ret;
	return (ret == 0) ? 0 : 1;
}

