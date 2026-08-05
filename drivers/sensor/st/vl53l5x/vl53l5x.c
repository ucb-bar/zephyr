/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Zephyr driver wrapper for ST VL53L5CX ULD (multizone ToF).
 *
 * Notes:
 * - Devicetree `reg` uses 7-bit address (0x29 style). The ST ULD uses 8-bit
 *   address (0x52 style) internally, so we adapt in the platform layer.
 * - This driver returns millimeters in `SENSOR_CHAN_DISTANCE` for consistency
 *   with the existing `vl53l1x` driver in this tree.
 */

#define DT_DRV_COMPAT st_vl53l5cx

#include <errno.h>

#include <zephyr/device.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util_macro.h>

#include <zephyr/drivers/sensor/vl53l5cx.h>

#include <string.h>

#include "vl53l5cx_api.h"

LOG_MODULE_REGISTER(VL53L5CX, CONFIG_SENSOR_LOG_LEVEL);

struct vl53l5x_config {
	struct i2c_dt_spec i2c;
#ifdef CONFIG_VL53L5CX_LPN
	struct gpio_dt_spec lpn;
#endif
};

struct vl53l5x_data {
	struct i2c_dt_spec i2c; /* runtime copy to allow addr update */
	struct k_mutex lock;

	VL53L5CX_Configuration dev;
	VL53L5CX_ResultsData results;

	bool initialized;
	bool ranging;
	uint8_t resolution;
};

/* --- ST platform hooks (implemented in st/porting/platform.c) --- */
int32_t vl53l5x_platform_get_tick(void);
int32_t vl53l5x_platform_write(VL53L5CX_Platform *p, uint16_t reg, uint8_t *buf, uint16_t len);
int32_t vl53l5x_platform_read(VL53L5CX_Platform *p, uint16_t reg, uint8_t *buf, uint16_t len);

static int vl53l5x_initialize_locked(const struct device *dev)
{
	struct vl53l5x_data *data = dev->data;
	const struct vl53l5x_config *cfg = dev->config;
	uint8_t status;

	if (data->initialized) {
		return 0;
	}

	if (!device_is_ready(cfg->i2c.bus)) {
		LOG_ERR("[%s] I2C bus not ready", dev->name);
		return -ENODEV;
	}

#ifdef CONFIG_VL53L5CX_LPN
	if (cfg->lpn.port) {
		if (!gpio_is_ready_dt(&cfg->lpn)) {
			LOG_ERR("[%s] LPn GPIO not ready", dev->name);
			return -ENODEV;
		}
		int ret = gpio_pin_configure_dt(&cfg->lpn, GPIO_OUTPUT_INACTIVE);
		if (ret < 0) {
			LOG_ERR("[%s] Failed to configure LPn GPIO", dev->name);
			return ret;
		}

		/* Bring device out of low power. */
		(void)gpio_pin_set_dt(&cfg->lpn, 1);
		k_sleep(K_MSEC(2));
	}
#endif

	memset(&data->dev, 0, sizeof(data->dev));
	memset(&data->results, 0, sizeof(data->results));

	/* ST expects 8-bit address (0x52) internally. */
	data->dev.platform.address = (uint16_t)(data->i2c.addr << 1);
	data->dev.platform.user_data = &data->i2c;
	data->dev.platform.Write = vl53l5x_platform_write;
	data->dev.platform.Read = vl53l5x_platform_read;
	data->dev.platform.GetTick = vl53l5x_platform_get_tick;

	status = vl53l5cx_init(&data->dev);
	if (status != VL53L5CX_STATUS_OK) {
		LOG_ERR("[%s] vl53l5cx_init failed (status=%u)", dev->name, status);
		return -EIO;
	}

	/* Program default profile */
#if defined(CONFIG_VL53L5CX_DEFAULT_RESOLUTION_8X8)
	status = vl53l5cx_set_resolution(&data->dev, VL53L5CX_RESOLUTION_8X8);
#else
	status = vl53l5cx_set_resolution(&data->dev, VL53L5CX_RESOLUTION_4X4);
#endif
	if (status != VL53L5CX_STATUS_OK) {
		LOG_ERR("[%s] set_resolution failed (%u)", dev->name, status);
		return -EIO;
	}

	/* Continuous mode is typical; integration time only affects autonomous mode
	 * but we still set it for consistency with ST examples.
	 */
	status = vl53l5cx_set_ranging_mode(&data->dev, VL53L5CX_RANGING_MODE_CONTINUOUS);
	status |= vl53l5cx_set_integration_time_ms(&data->dev, CONFIG_VL53L5CX_DEFAULT_INTEGRATION_TIME_MS);
	status |= vl53l5cx_set_ranging_frequency_hz(&data->dev, CONFIG_VL53L5CX_DEFAULT_RANGING_FREQUENCY_HZ);
	if (status != VL53L5CX_STATUS_OK) {
		LOG_ERR("[%s] default profile programming failed (%u)", dev->name, status);
		return -EIO;
	}

	status = vl53l5cx_get_resolution(&data->dev, &data->resolution);
	if (status != VL53L5CX_STATUS_OK) {
		data->resolution = VL53L5CX_RESOLUTION_4X4;
	}

	data->initialized = true;
	data->ranging = false;
	return 0;
}

static void vl53l5x_setup_platform_only_locked(const struct device *dev)
{
	struct vl53l5x_data *data = dev->data;

	memset(&data->dev, 0, sizeof(data->dev));
	data->dev.platform.address = (uint16_t)(data->i2c.addr << 1);
	data->dev.platform.user_data = &data->i2c;
	data->dev.platform.Write = vl53l5x_platform_write;
	data->dev.platform.Read = vl53l5x_platform_read;
	data->dev.platform.GetTick = vl53l5x_platform_get_tick;
}

static int vl53l5x_ensure_ranging_locked(const struct device *dev)
{
	struct vl53l5x_data *data = dev->data;
	uint8_t status;

	if (!data->initialized) {
		int ret = vl53l5x_initialize_locked(dev);
		if (ret != 0) {
			return ret;
		}
	}

	if (data->ranging) {
		return 0;
	}

	status = vl53l5cx_start_ranging(&data->dev);
	if (status != VL53L5CX_STATUS_OK) {
		LOG_ERR("[%s] start_ranging failed (%u)", dev->name, status);
		return -EIO;
	}
	data->ranging = true;
	return 0;
}

static int vl53l5x_sample_fetch(const struct device *dev, enum sensor_channel chan)
{
	struct vl53l5x_data *data = dev->data;
	uint8_t ready = 0U;
	uint8_t status;
	int ret;
	int64_t start;

	ARG_UNUSED(chan);

	k_mutex_lock(&data->lock, K_FOREVER);

	ret = vl53l5x_ensure_ranging_locked(dev);
	if (ret != 0) {
		goto out;
	}

	start = k_uptime_get();
	do {
		status = vl53l5cx_check_data_ready(&data->dev, &ready);
		if (status != VL53L5CX_STATUS_OK) {
			LOG_ERR("[%s] check_data_ready failed (%u)", dev->name, status);
			ret = -EIO;
			goto out;
		}
		if (ready != 0U) {
			break;
		}
		k_sleep(K_MSEC(CONFIG_VL53L5CX_DATA_READY_POLL_MS));
	} while ((k_uptime_get() - start) < CONFIG_VL53L5CX_DATA_READY_TIMEOUT_MS);

	if (ready == 0U) {
		ret = -ETIMEDOUT;
		goto out;
	}

	status = vl53l5cx_get_ranging_data(&data->dev, &data->results);
	if (status != VL53L5CX_STATUS_OK) {
		LOG_ERR("[%s] get_ranging_data failed (%u)", dev->name, status);
		ret = -EIO;
		goto out;
	}

	status = vl53l5cx_get_resolution(&data->dev, &data->resolution);
	if (status != VL53L5CX_STATUS_OK) {
		/* Keep previous. */
	}

	ret = 0;

out:
	k_mutex_unlock(&data->lock);
	return ret;
}

static inline uint8_t vl53l5x_side_from_resolution(uint8_t resolution)
{
	return (resolution == VL53L5CX_RESOLUTION_8X8) ? 8U : 4U;
}

static int vl53l5x_channel_get(const struct device *dev,
			       enum sensor_channel chan,
			       struct sensor_value *val)
{
	struct vl53l5x_data *data = dev->data;
	uint8_t side;
	uint8_t center_idx;
	int16_t dist_mm;

	if (chan != SENSOR_CHAN_DISTANCE) {
		return -ENOTSUP;
	}

	k_mutex_lock(&data->lock, K_FOREVER);
	if (!data->initialized) {
		k_mutex_unlock(&data->lock);
		return -EAGAIN;
	}

	side = vl53l5x_side_from_resolution(data->resolution);
	center_idx = (uint8_t)((side / 2U) * side + (side / 2U));

	dist_mm = data->results.distance_mm[(uint16_t)center_idx * VL53L5CX_NB_TARGET_PER_ZONE];

	val->val1 = (int32_t)dist_mm;
	val->val2 = 0;

	k_mutex_unlock(&data->lock);
	return 0;
}

static DEVICE_API(sensor, vl53l5x_api) = {
	.sample_fetch = vl53l5x_sample_fetch,
	.channel_get = vl53l5x_channel_get,
};

int vl53l5cx_reinit(const struct device *dev)
{
	struct vl53l5x_data *data = dev->data;
	int ret;

	k_mutex_lock(&data->lock, K_FOREVER);
	data->initialized = false;
	data->ranging = false;
	ret = vl53l5x_initialize_locked(dev);
	k_mutex_unlock(&data->lock);
	return ret;
}

int vl53l5cx_set_i2c_address_7bit(const struct device *dev, uint8_t new_addr_7bit)
{
	struct vl53l5x_data *data = dev->data;
	uint8_t status;
	uint8_t alive = 0U;
	uint8_t old_addr_7bit;

	k_mutex_lock(&data->lock, K_FOREVER);
	if (!data->initialized) {
		/* Allow address programming before full firmware download init. */
		vl53l5x_setup_platform_only_locked(dev);
	}

	old_addr_7bit = (uint8_t)data->i2c.addr;

	/* First check: are we alive on the current bus address? */
	status = vl53l5cx_is_alive(&data->dev, &alive);
	if (status != VL53L5CX_STATUS_OK || alive == 0U) {
		/* If not, but the caller is asking for a new address, try that
		 * address too (common case: device already reprogrammed).
		 */
		if (new_addr_7bit != old_addr_7bit) {
			data->i2c.addr = new_addr_7bit;
			data->dev.platform.address = (uint16_t)(new_addr_7bit << 1);
			alive = 0U;
			status = vl53l5cx_is_alive(&data->dev, &alive);

			if (status == VL53L5CX_STATUS_OK && alive != 0U) {
				LOG_INF("[%s] device already responds at 0x%02x; skipping reprogram",
					dev->name, new_addr_7bit);
				k_mutex_unlock(&data->lock);
				return 0;
			}

			/* Restore old address for error logging. */
			data->i2c.addr = old_addr_7bit;
			data->dev.platform.address = (uint16_t)(old_addr_7bit << 1);
		}

		LOG_ERR("[%s] is_alive failed at 0x%02x and 0x%02x (status=%u alive=%u last_i2c_err=%d)",
			dev->name, old_addr_7bit, new_addr_7bit, status, alive,
			(int)data->dev.platform.last_error);
		k_mutex_unlock(&data->lock);
		return -EIO;
	}

	/* ST API expects 8-bit address. */
	status = vl53l5cx_set_i2c_address(&data->dev, (uint16_t)new_addr_7bit << 1);
	if (status != VL53L5CX_STATUS_OK) {
		LOG_ERR("[%s] set_i2c_address failed (status=%u last_i2c_err=%d old_addr7=0x%02x new_addr7=0x%02x)",
			dev->name, status, (int)data->dev.platform.last_error, data->i2c.addr, new_addr_7bit);
		k_mutex_unlock(&data->lock);
		return -EIO;
	}

	/* Keep the Zephyr-side address in sync for subsequent I2C transactions. */
	data->i2c.addr = new_addr_7bit;

	/* Best-effort sanity: check it responds at the new address. */
	status = vl53l5cx_is_alive(&data->dev, &alive);
	LOG_INF("[%s] address change done (alive=%u last_i2c_err=%d new_addr7=0x%02x)",
		dev->name, alive, (int)data->dev.platform.last_error, new_addr_7bit);
	k_mutex_unlock(&data->lock);
	return 0;
}

int vl53l5cx_get_grid(const struct device *dev, struct vl53l5cx_grid *out)
{
	struct vl53l5x_data *data = dev->data;
	uint8_t res;

	if (out == NULL) {
		return -EINVAL;
	}

	k_mutex_lock(&data->lock, K_FOREVER);
	if (!data->initialized) {
		k_mutex_unlock(&data->lock);
		return -EAGAIN;
	}

	res = data->resolution;
	out->resolution = res;

	/* Copy only the first target per zone into a flat 64-entry grid. */
	for (uint8_t i = 0U; i < VL53L5CX_RESOLUTION_8X8; i++) {
		out->nb_target_detected[i] = data->results.nb_target_detected[i];
		out->target_status[i] = data->results.target_status[(uint16_t)i * VL53L5CX_NB_TARGET_PER_ZONE];
		out->distance_mm[i] = data->results.distance_mm[(uint16_t)i * VL53L5CX_NB_TARGET_PER_ZONE];
	}

	k_mutex_unlock(&data->lock);
	return 0;
}

static int vl53l5x_init(const struct device *dev)
{
	struct vl53l5x_data *data = dev->data;
	const struct vl53l5x_config *cfg = dev->config;

	k_mutex_init(&data->lock);
	data->i2c = cfg->i2c;
	data->initialized = false;
	data->ranging = false;
	data->resolution = VL53L5CX_RESOLUTION_4X4;
	return 0;
}

#define VL53L5X_DEFINE(inst)									\
	static struct vl53l5x_data vl53l5x_data_##inst;						\
	static const struct vl53l5x_config vl53l5x_config_##inst = {				\
		.i2c = I2C_DT_SPEC_INST_GET(inst),							\
		IF_ENABLED(CONFIG_VL53L5CX_LPN, (.lpn = GPIO_DT_SPEC_INST_GET_OR(inst, lpn_gpios, {0}),)) \
	};											\
	SENSOR_DEVICE_DT_INST_DEFINE(inst,								\
				    vl53l5x_init,								\
				    NULL,									\
				    &vl53l5x_data_##inst,							\
				    &vl53l5x_config_##inst,							\
				    POST_KERNEL,								\
				    CONFIG_SENSOR_INIT_PRIORITY,						\
				    &vl53l5x_api);

DT_INST_FOREACH_STATUS_OKAY(VL53L5X_DEFINE)

