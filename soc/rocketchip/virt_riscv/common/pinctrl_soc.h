/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal pinctrl_soc.h for the Chipyard/Rocket "virt_riscv" SoC. This SoC's peripheral pins are
 * fixed by the FPGA harness (no runtime pin-mux), so pinctrl is a no-op and CONFIG_PINCTRL stays
 * off. This header exists only to satisfy <zephyr/drivers/pinctrl.h>'s unconditional include so
 * that drivers which pull it in (e.g. the SiFive SPI/PWM drivers, which use pinctrl solely under
 * #ifdef CONFIG_PINCTRL) can compile on this SoC.
 */
#ifndef ZEPHYR_SOC_ROCKETCHIP_VIRT_RISCV_PINCTRL_SOC_H_
#define ZEPHYR_SOC_ROCKETCHIP_VIRT_RISCV_PINCTRL_SOC_H_

#include <zephyr/types.h>
#include <zephyr/devicetree.h>

typedef uint32_t pinctrl_soc_pin_t;

#define Z_PINCTRL_STATE_PIN_INIT(node_id, prop, idx) \
	(uint32_t)DT_PROP_BY_IDX(node_id, prop, idx),

#define Z_PINCTRL_STATE_PINS_INIT(node_id, prop) \
	{ DT_FOREACH_PROP_ELEM(node_id, prop, Z_PINCTRL_STATE_PIN_INIT) }

#endif /* ZEPHYR_SOC_ROCKETCHIP_VIRT_RISCV_PINCTRL_SOC_H_ */
