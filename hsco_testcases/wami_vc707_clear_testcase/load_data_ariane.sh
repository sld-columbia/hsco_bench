#!/bin/bash
# Load WAMI test data into FPGA memory for Ariane processor
# Memory layout starting at 0xB0000000
# No model binary needed — WAMI is a signal processing benchmark
# Sizes are computed automatically from the data files.

make esplink

BASE_ADDR=0xB0000000

INPUT_FILE=data_files/test_input.bin
GOLD_FILE=data_files/gold_output.bin

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: $INPUT_FILE not found. Run prepare_data.py first."
    exit 1
fi

INPUT_SIZE=$(stat -c%s "$INPUT_FILE")
GOLD_SIZE=0
if [ -f "$GOLD_FILE" ]; then
    GOLD_SIZE=$(stat -c%s "$GOLD_FILE")
fi

GOLD_ADDR=$((BASE_ADDR + INPUT_SIZE))
TOTAL=$((INPUT_SIZE + GOLD_SIZE))

echo "Loading WAMI test data..."
echo "  test_input.bin:  $INPUT_SIZE bytes"
echo "  gold_output.bin: $GOLD_SIZE bytes"
echo "  Total:           $TOTAL bytes"

./socgen/esp/esplink --load -i "$INPUT_FILE" -a $(printf "0x%X" $BASE_ADDR)
echo "Input loaded at $(printf "0x%X" $BASE_ADDR)"

if [ -f "$GOLD_FILE" ]; then
    echo "Gold output loading at $(printf "0x%X" $GOLD_ADDR)"
    ./socgen/esp/esplink --load -i "$GOLD_FILE" -a $(printf "0x%X" $GOLD_ADDR)
fi

echo "Done! Data loaded to FPGA memory"
echo "  Test input:  $(printf "0x%X" $BASE_ADDR) ($INPUT_SIZE bytes)"
echo "  Gold output: $(printf "0x%X" $GOLD_ADDR) ($GOLD_SIZE bytes)"
