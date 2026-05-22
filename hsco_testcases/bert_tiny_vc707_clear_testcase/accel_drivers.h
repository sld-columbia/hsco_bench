/*
 * Accelerator Driver Registry
 * Single entry point for all accelerator includes, init, and cleanup.
 *
 * systest.c includes ONLY this file. When adding a new accelerator:
 *   1. Add #include for the new driver header
 *   2. Add init call in accel_init_all()
 *   3. Add cleanup call in accel_cleanup_all()
 *
 * systest.c never needs to change.
 */

#ifndef ACCEL_DRIVERS_H
#define ACCEL_DRIVERS_H

#ifdef USE_ACCELERATOR

/* ============================================================================ */
/* Shared accelerator infrastructure                                            */
/* ============================================================================ */
#include "accel_common.h"

/* ============================================================================ */
/* Per-accelerator driver includes                                              */
/* Add new accelerator drivers here:                                            */
/* ============================================================================ */
/* --- include accelerator drivers here --- */

/* ============================================================================ */
/* Initialize all accelerators                                                  */
/* Returns 0 on success, -1 if any critical accelerator fails                   */
/* ============================================================================ */
static int accel_init_all(void) {
    /* BSS is not zeroed on Ariane baremetal - must explicitly init */
    accel_common_init();

    printf("Initializing accelerators...\n");

    /* --- Add new accelerator init calls here --- */

    return 0;
}

/* ============================================================================ */
/* Cleanup all accelerators (per-driver cleanup first, then shared memory)       */
/* ============================================================================ */
static void accel_cleanup_all(void) {
    /* --- Add new accelerator cleanup calls here --- */

    /* Shared memory last */
    accel_common_free();
}

#endif /* USE_ACCELERATOR */

#endif /* ACCEL_DRIVERS_H */
