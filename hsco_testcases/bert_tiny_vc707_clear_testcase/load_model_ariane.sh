#!/bin/bash
# Load BERT-tiny model and test data into FPGA memory for Ariane processor
# Memory layout starting at 0xB0000000
# This version loads the entire model binary file instead of separate weight files

make esplink

BASE_ADDR=0xB0000000

echo "Loading BERT-tiny model and test data..."

# Load entire model binary (includes all weights with header)
# bert_tiny_model_aligned.bin is ~17.5MB
./socgen/esp/esplink --load -i model_files/bert_tiny_model_aligned.bin -a $(printf "0x%X" $BASE_ADDR)

# Calculate address after model binary
# Model binary size is approximately 17,545,733 bytes, round up to 0x1100000 (17,825,792 bytes) for safe alignment
MODEL_SIZE=0x1100000
TEST_DATA_ADDR=$((BASE_ADDR + MODEL_SIZE))

echo "Model loaded at 0xB0000000"
echo "Test data will be loaded at $(printf "0x%X" $TEST_DATA_ADDR)"

# Load test input data (128 int32 input_ids + 128 int32 attention_mask = 1024 bytes, raw)
./socgen/esp/esplink --load -i model_files/test_input.bin -a $(printf "0x%X" $TEST_DATA_ADDR)

# Load expected test output (128 float32 = 512 bytes, raw)
# Place after test input: 1024 bytes
OUTPUT_ADDR=$((TEST_DATA_ADDR + 1024))
./socgen/esp/esplink --load -i model_files/test_output.bin -a $(printf "0x%X" $OUTPUT_ADDR)

echo "Done! Data loaded to FPGA memory"
echo "  Model binary:     0xB0000000"
echo "  Test input:       $(printf "0x%X" $TEST_DATA_ADDR)"
echo "  Expected output:  $(printf "0x%X" $OUTPUT_ADDR)"
