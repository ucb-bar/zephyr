/*
 * Copyright (c) 2018-2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ZEPHYR_INCLUDE_POSIX_SCHED_H_
#define ZEPHYR_INCLUDE_POSIX_SCHED_H_

#include <zephyr/kernel.h>
#include <zephyr/posix/posix_types.h>

#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Other mandatory scheduling policy. Must be numerically distinct. May
 * execute identically to SCHED_RR or SCHED_FIFO. For Zephyr this is a
 * pseudonym for SCHED_RR.
 */
#define SCHED_OTHER 0

/* Cooperative scheduling policy */
#define SCHED_FIFO 1

/* Priority based preemptive scheduling policy */
#define SCHED_RR 2

#if defined(CONFIG_MINIMAL_LIBC) || defined(CONFIG_PICOLIBC) || defined(CONFIG_ARMCLANG_STD_LIBC) \
	|| defined(CONFIG_ARCMWDT_LIBC)
struct sched_param {
	int sched_priority;
};
#endif

/**
 * @brief Yield the processor
 *
 * See IEEE 1003.1
 */
int sched_yield(void);

int sched_get_priority_min(int policy);
int sched_get_priority_max(int policy);

int sched_getparam(pid_t pid, struct sched_param *param);
int sched_getscheduler(pid_t pid);

int sched_setparam(pid_t pid, const struct sched_param *param);
int sched_setscheduler(pid_t pid, int policy, const struct sched_param *param);
int sched_rr_get_interval(pid_t pid, struct timespec *interval);

/*
 * cpu_set_t + CPU_* macros for pthread_attr_{set,get}affinity_np().
 * Vendored as part of CONFIG_POSIX_THREADS_AFFINITY (see lib/posix/options/Kconfig.pthread).
 * 64-bit bitmask covers up to CONFIG_MP_MAX_NUM_CPUS=64; embedded targets are well below that.
 */
#ifdef CONFIG_POSIX_THREADS_AFFINITY
#define CPU_SETSIZE (sizeof(uint64_t) * 8)
typedef struct {
	uint64_t bits;
} cpu_set_t;
#define CPU_ZERO(s)     ((s)->bits = 0)
#define CPU_SET(c, s)   ((s)->bits |= (1ULL << (c)))
#define CPU_CLR(c, s)   ((s)->bits &= ~(1ULL << (c)))
#define CPU_ISSET(c, s) (((s)->bits >> (c)) & 1U)
#define CPU_COUNT(s)    __builtin_popcountll((s)->bits)
#endif /* CONFIG_POSIX_THREADS_AFFINITY */

#ifdef __cplusplus
}
#endif

#endif /* ZEPHYR_INCLUDE_POSIX_SCHED_H_ */
