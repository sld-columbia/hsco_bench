/*
 * DUMMY Accelerator Configuration (REFERENCE ONLY)
 *
 * Accelerator: dummy (structural reference — identity kernel only)
 * Device ID:   0x200
 *
 * Register offsets are assigned to XML <param>s in REVERSE order:
 * the LAST <param> takes 0x40, the second-to-last 0x44, the first 0x48.
 * For this accelerator the XML is [rows, cols, loaded_cols], so:
 *   rows        (1st in XML)  -> 0x48
 *   cols        (2nd in XML)  -> 0x44
 *   loaded_cols (3rd in XML)  -> 0x40
 */

#ifndef DUMMY_CFG_H
#define DUMMY_CFG_H

// Device identification
#define DUMMY_DEV_ID    0x200
#define DUMMY_DEV_NAME  "sld,dummy_sysc_catapult"

// Register offsets (reverse order from XML)
#define DUMMY_REG_ROWS  0x48
#define DUMMY_REG_COLS  0x44
#define DUMMY_REG_LOADED_COLS  0x40

// Fixed-point configuration
#define DUMMY_FX_WL  32
#define DUMMY_FX_IL  16

// DMA alignment: dma64 tile carries 2 x 32-bit tokens per beat.
// Offsets between A/B/O regions must be rounded up to this boundary so
// they agree with round_up(length, DMA_WORD_PER_BEAT) inside the HLS kernel.
#define DUMMY_DMA_WORD_PER_BEAT 2

#endif // DUMMY_CFG_H
