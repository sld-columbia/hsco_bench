#!/bin/bash
# Load TinyCLIP Vision Single-Layer test data into FPGA memory for Ariane processor
# Memory layout starting at 0xB0000000
#
# Layout:
#   0xB0000000                         Model binary (aligned, ~33 MB — for layer 0 weights)
#   0xB0000000 + 0x2200000             layer0_input.bin  (197*256*4 = 201,728 bytes)
#   0xB0000000 + 0x2200000 + 201728    layer0_output.bin (197*256*4 = 201,728 bytes)

make esplink

BASE_ADDR=0xB0000000

echo "Loading TinyCLIP Vision Single-Layer test data..."

# Load model binary (aligned format, ~33 MB)
echo "  Loading vision model binary (~33 MB)..."
./socgen/esp/esplink --load -i model_files/tinyclip_vision_aligned.bin -a $(printf "0x%X" $BASE_ADDR)

# Calculate address after model binary
# tinyclip_vision_aligned.bin is ~33 MB, round up to 0x2200000 (35,651,584 bytes)
MODEL_OFFSET=0x2200000
LAYER_IO_SIZE=201728  # 197 * 256 * 4 bytes

INPUT_ADDR=$((BASE_ADDR + MODEL_OFFSET))
OUTPUT_ADDR=$((INPUT_ADDR + LAYER_IO_SIZE))

echo "  Model loaded at 0xB0000000"

# Load layer 0 input (golden reference from standalone dump)
echo "  Loading layer0_input.bin at $(printf "0x%X" $INPUT_ADDR)..."
./socgen/esp/esplink --load -i model_files/layer0_input.bin -a $(printf "0x%X" $INPUT_ADDR)

# Load layer 0 expected output (golden reference from standalone dump)
echo "  Loading layer0_output.bin at $(printf "0x%X" $OUTPUT_ADDR)..."
./socgen/esp/esplink --load -i model_files/layer0_output.bin -a $(printf "0x%X" $OUTPUT_ADDR)

echo ""
echo "Done! Data loaded to FPGA memory"
echo "  Model binary:     0xB0000000"
echo "  Layer 0 input:    $(printf "0x%X" $INPUT_ADDR) ($LAYER_IO_SIZE bytes)"
echo "  Layer 0 output:   $(printf "0x%X" $OUTPUT_ADDR) ($LAYER_IO_SIZE bytes)"
