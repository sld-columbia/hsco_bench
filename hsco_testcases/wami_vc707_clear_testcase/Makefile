# Copyright (c) 2011-2023 Columbia University, System Level Design Group
# SPDX-License-Identifier: Apache-2.0
##########################
### ESP Local Makefile ###
##########################

all: help

### Global design variables ###
DESIGN_PATH = $(PWD)
ESP_ROOT ?= $(error ESP_ROOT is not set. Please export it before running make.)
TECHLIB  = virtex7
BOARD    = xilinx-vc707-xc7vx485t
DESIGN   = esp-$(BOARD)
SMP = 0
BASE_FREQ_MHZ ?= 50
LEON3_STACK ?= 0x5bfffff0
USE_OPENSBI ?= 1
ESP_DEFCONFIG = $(DESIGN_PATH)/esp_xilinx-vc707-xc7vx485t_defconfig

### Design top level and testbench ###
TOP    = top
SIMTOP = testbench


### Modelsim Simulation Options ###

# Compile flags
VCOMOPT +=
VLOGOPT +=

XCOMOPT +=
XLOGOPT +=

VSIMOPT +=


### Additional design files ###
TOP_VHDL_RTL_PKGS +=
TOP_VHDL_RTL_SRCS +=
TOP_VLOG_RTL_SRCS +=
TOP_VHDL_SIM_PKGS +=
TOP_VHDL_SIM_SRCS +=
TOP_VLOG_SIM_SRCS +=


### Include global Makefile ###
include $(ESP_ROOT)/utils/Makefile

### Baremetal ESP drivers configuration ###
DRV_BARE = $(ESP_ROOT)/soft/common/drivers/baremetal
DRV_COMMON = $(ESP_ROOT)/soft/common/drivers/common

# Include paths for ESP baremetal
BARE_CFLAGS  = -I$(DRV_COMMON)/include
BARE_CFLAGS += -I$(DRV_BARE)/include
BARE_CFLAGS += -I$(DRV_COMMON)/monitors/include
BARE_CFLAGS += -I$(SOFT)/riscv-pk/machine
BARE_CFLAGS += -DCACHELINE_WIDTH=$(CONFIG_CACHE_LINE_SIZE)

# Source files for ESP drivers
BARE_SRCS  = $(DRV_BARE)/probe/probe.c
BARE_SRCS += $(DRV_BARE)/probe/fdt.c
BARE_SRCS += $(DRV_COMMON)/monitors/libmonitors.c

### Override systest.exe to include ESP drivers and wami.c ###
# NOTE: -fno-fast-math overrides the framework's -ffast-math from RISCV_CFLAGS.
# -ffast-math breaks IEEE 754 compliance (reciprocal-math, no-NaN assumptions)
# causing Lucas-Kanade iteration to diverge on RISC-V.
$(SOFT_BUILD)/systest.exe: systest.c wami.c $(SOFT_BUILD)/uart.o
	@mkdir -p $(SOFT_BUILD)
	$(QUIET_CC) $(CROSS_COMPILE_ELF)gcc $(RISCV_CFLAGS) -fno-fast-math $(BARE_CFLAGS) \
	$(SOFT)/common/syscalls.c \
	$(BARE_SRCS) \
	$(RISCV_TESTS)/benchmarks/common/crt.S \
	-T $(RISCV_TESTS)/benchmarks/common/test.ld -o $@ \
	-I$(DESIGN_PATH)/$(ESP_CFG_BUILD) \
	$(SOFT_BUILD)/uart.o $< wami.c
