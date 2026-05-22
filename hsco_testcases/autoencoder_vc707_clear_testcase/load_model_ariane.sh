#!/bin/bash
# Load autoencoder model and test data into FPGA memory for Ariane processor
# Memory layout starting at 0xB0000000
# This version loads the entire model binary file instead of separate weight files

make esplink

BASE_ADDR=0xB0000000

echo "Loading autoencoder model and test data..."

# Load entire model binary (includes all weights with header)
# autoencoder.bin is ~1.1MB
./socgen/esp/esplink --load -i model_files/autoencoder_aligned.bin -a $(printf "0x%X" $BASE_ADDR)

# Calculate address after model binary
# Model binary size is approximately 1081665 bytes, round up to 0x110000 (1114112 bytes) for safe alignment
MODEL_SIZE=0x110000
TEST_DATA_ADDR=$((BASE_ADDR + MODEL_SIZE))

echo "Model loaded at 0xB0000000"
echo "Test data will be loaded at $(printf "0x%X" $TEST_DATA_ADDR)"

# Load test input data (5 samples, 640 floats each = 12800 bytes)
./socgen/esp/esplink --load -i model_files/test_input.bin -a $(printf "0x%X" $TEST_DATA_ADDR)

# Load test labels (5 ints = 20 bytes)
# Place after test input: test_input.bin has 8 byte header + 12800 bytes data = 12808 bytes
LABELS_ADDR=$((TEST_DATA_ADDR + 12808))
./socgen/esp/esplink --load -i model_files/test_input_labels.bin -a $(printf "0x%X" $LABELS_ADDR)

echo "Done! Data loaded to FPGA memory"
echo "  Model binary: 0xB0000000"
echo "  Test inputs:  $(printf "0x%X" $TEST_DATA_ADDR)"
echo "  Test labels:  $(printf "0x%X" $LABELS_ADDR)"
