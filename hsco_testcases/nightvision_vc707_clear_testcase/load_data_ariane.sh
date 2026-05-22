#!/bin/bash
# Load nightvision test data into FPGA memory for Ariane processor
# Memory layout starting at 0xB0000000
# No model binary needed — nightvision is a signal processing benchmark

make esplink

BASE_ADDR=0xB0000000

echo "Loading nightvision test data..."

# Load test input: [nRows:u32][nCols:u32][imgA: int32 x 1200][imgB: int32 x 1200]
# Size: 8 + 9600 = 9608 bytes
./socgen/esp/esplink --load -i data_files/test_input.bin -a $(printf "0x%X" $BASE_ADDR)

# Load gold output right after input
# test_input.bin is 9608 bytes
INPUT_SIZE=9608
GOLD_ADDR=$((BASE_ADDR + INPUT_SIZE))

echo "Input loaded at 0xB0000000"
echo "Gold output will be loaded at $(printf "0x%X" $GOLD_ADDR)"

# Load gold output: [nPxls:u32][ens1: int32 x 1200][ens2: int32 x 1200][ens3: int32 x 1200]
# Size: 4 + 14400 = 14404 bytes
./socgen/esp/esplink --load -i data_files/gold_output.bin -a $(printf "0x%X" $GOLD_ADDR)

echo "Done! Data loaded to FPGA memory"
echo "  Test input:       0xB0000000 (9,608 bytes)"
echo "  Gold output:      $(printf "0x%X" $GOLD_ADDR) (14,404 bytes)"
echo "  Total:            24,012 bytes"
