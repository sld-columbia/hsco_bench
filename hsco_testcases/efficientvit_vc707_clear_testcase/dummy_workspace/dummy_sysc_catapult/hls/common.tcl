# Copyright (c) 2011-2022 Columbia University, System Level Design Group
# SPDX-License-Identifier: Apache-2.0

set TECH "virtex7"

#
# Technology Libraries
#
if {$TECH eq "virtexup"} {
    set FPGA_PART_NUM "xcvu9p-flga2104-2-e"
    set FPGA_FAMILY "VIRTEX-uplus"
    set FPGA_SPEED_GRADE "-2"
}

if {$TECH eq "virtex7"} {
    set FPGA_PART_NUM "xc7vx485tffg1761-2"
    set FPGA_FAMILY "VIRTEX-7"
    set FPGA_SPEED_GRADE "-2"
}
