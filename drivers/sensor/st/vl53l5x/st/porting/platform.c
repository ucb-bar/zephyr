/**
  ******************************************************************************
  * @file    platform.h
  * @author  IMG SW Application Team
  * @brief   This file contains all the platform functions prototypes
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

#include "platform.h"

#include <zephyr/kernel.h>

uint8_t VL53L5CX_RdByte(
		VL53L5CX_Platform *p_platform,
		uint16_t RegisterAdress,
		uint8_t *p_value)
{
  return p_platform->Read(p_platform, RegisterAdress, p_value, 1U);
}

uint8_t VL53L5CX_WrByte(
		VL53L5CX_Platform *p_platform,
		uint16_t RegisterAdress,
		uint8_t value)
{
  return p_platform->Write(p_platform, RegisterAdress, &value, 1U);
}

uint8_t VL53L5CX_WrMulti(
		VL53L5CX_Platform *p_platform,
		uint16_t RegisterAdress,
		uint8_t *p_values,
		uint32_t size)
{
  return p_platform->Write(p_platform, RegisterAdress, p_values, (uint16_t)size);
}

uint8_t VL53L5CX_RdMulti(
		VL53L5CX_Platform *p_platform,
		uint16_t RegisterAdress,
		uint8_t *p_values,
		uint32_t size)
{
  return p_platform->Read(p_platform, RegisterAdress, p_values, (uint16_t)size);
}

void VL53L5CX_SwapBuffer(
    uint8_t     *buffer,
    uint16_t     size)
{
  uint32_t i, tmp;

  /* Example of possible implementation using <string.h> */
  for(i = 0; i < size; i = i + 4)
  {
    tmp = (
      buffer[i]<<24)
    |(buffer[i+1]<<16)
    |(buffer[i+2]<<8)
    |(buffer[i+3]);

    memcpy(&(buffer[i]), &tmp, 4);
  }
}

uint8_t VL53L5CX_WaitMs(
		VL53L5CX_Platform *p_platform,
		uint32_t TimeMs)
{
  (void)p_platform;
  k_msleep(TimeMs);
  return 0U;
}

