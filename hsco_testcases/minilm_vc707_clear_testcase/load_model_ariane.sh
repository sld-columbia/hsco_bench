#!/bin/bash
# Load MiniLM (lite) model and test data into FPGA memory for Ariane processor
# Memory layout starting at 0xB0000000
# Lite mode: encoder_pooler weights + pre-computed embeddings

make esplink

BASE_ADDR=0xB0000000

echo "Loading MiniLM (lite) model and test data..."

# Load encoder+pooler model binary (aligned format, ~21.9 MB)
./socgen/esp/esplink --load -i model_files/encoder_pooler_aligned.bin -a $(printf "0x%X" $BASE_ADDR)

# Calculate address after model binary
# encoder_pooler_aligned.bin is ~21.9MB, round up to 0x1500000 (22,020,096 bytes)
MODEL_SIZE=0x1500000
TEST_DATA_ADDR=$((BASE_ADDR + MODEL_SIZE))

echo "Model loaded at 0xB0000000"
echo "Test data will be loaded at $(printf "0x%X" $TEST_DATA_ADDR)"

# Load pre-computed embeddings + attention mask (raw format)
# [128*384 float32 embeddings = 196,608 bytes] + [128 int32 mask = 512 bytes] = 197,120 bytes
./socgen/esp/esplink --load -i model_files/test_input.bin -a $(printf "0x%X" $TEST_DATA_ADDR)

# Load expected test output (384 float32 = 1,536 bytes, raw)
# Place after test input: 197,120 bytes
INPUT_SIZE=197120
OUTPUT_ADDR=$((TEST_DATA_ADDR + INPUT_SIZE))
./socgen/esp/esplink --load -i model_files/test_output.bin -a $(printf "0x%X" $OUTPUT_ADDR)

echo "Done! Data loaded to FPGA memory"
echo "  Model binary:     0xB0000000"
echo "  Test input:       $(printf "0x%X" $TEST_DATA_ADDR)"
echo "  Expected output:  $(printf "0x%X" $OUTPUT_ADDR)"
