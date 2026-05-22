#!/bin/bash
# Load EfficientViT-M0 model and test data into FPGA memory for Ariane processor
# Memory layout starting at 0xB0000000
# This version loads the entire model binary file instead of separate weight files

make esplink

BASE_ADDR=0xB0000000

echo "Loading EfficientViT-M0 model and test data..."

# Load entire model binary (includes all weights with header)
# efficientvit_m0_aligned.bin is ~9.4MB
./socgen/esp/esplink --load -i model_files/efficientvit_m0_aligned.bin -a $(printf "0x%X" $BASE_ADDR)

# Calculate address after model binary
# Model binary size is approximately 9,383,201 bytes, round up to 0xA00000 (10,485,760 bytes) for safe alignment
MODEL_SIZE=0xA00000
TEST_DATA_ADDR=$((BASE_ADDR + MODEL_SIZE))

echo "Model loaded at 0xB0000000"
echo "Test data will be loaded at $(printf "0x%X" $TEST_DATA_ADDR)"

# Load test input data (3 x 224 x 224 floats = 602112 bytes, raw float32 array)
./socgen/esp/esplink --load -i model_files/test_input.bin -a $(printf "0x%X" $TEST_DATA_ADDR)

# Load expected test output (1000 floats = 4000 bytes, raw float32 array)
# Place after test input: 602112 bytes
OUTPUT_ADDR=$((TEST_DATA_ADDR + 602112))
./socgen/esp/esplink --load -i model_files/test_output.bin -a $(printf "0x%X" $OUTPUT_ADDR)

echo "Done! Data loaded to FPGA memory"
echo "  Model binary:     0xB0000000"
echo "  Test input:       $(printf "0x%X" $TEST_DATA_ADDR)"
echo "  Expected output:  $(printf "0x%X" $OUTPUT_ADDR)"
