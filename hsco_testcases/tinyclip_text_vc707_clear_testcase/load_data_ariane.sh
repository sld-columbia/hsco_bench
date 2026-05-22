#!/bin/bash
# Load TinyCLIP Text-Only model and test data into FPGA memory for Ariane processor
# Memory layout starting at 0xB0000000
#
# Layout:
#   0xB0000000                                    Model binary (aligned, ~10 MB)
#   0xB0000000 + 0xA00000                         text_embeds.bin (with header, 78,864 bytes)
#   0xB0000000 + 0xA00000 + 78864                 text_features_golden.bin (with header, 2,056 bytes)

make esplink

BASE_ADDR=0xB0000000

echo "Loading TinyCLIP Text-Only model and test data..."

# Load model binary (aligned format, ~10 MB)
echo "  Loading text model binary (~10 MB)..."
./socgen/esp/esplink --load -i model_files/tinyclip_text_aligned.bin -a $(printf "0x%X" $BASE_ADDR)

# Calculate address after model binary
# tinyclip_text_aligned.bin is ~10 MB, round up to 0xA00000 (10,485,760 bytes)
MODEL_SIZE=0xA00000
EMBEDS_ADDR=$((BASE_ADDR + MODEL_SIZE))

echo "  Model loaded at 0xB0000000"
echo "  Text embeds will be loaded at $(printf "0x%X" $EMBEDS_ADDR)"

# Load pre-computed text embeddings (with header: nq + cl + ed + eot_positions + embeds)
# 12 + 4 + 1*77*256*4 = 78,864 bytes
echo "  Loading pre-computed text embeddings..."
./socgen/esp/esplink --load -i model_files/text_embeds.bin -a $(printf "0x%X" $EMBEDS_ADDR)

# Load golden text features after text embeddings
TEXT_EMBEDS_SIZE=78864
GOLDEN_ADDR=$((EMBEDS_ADDR + TEXT_EMBEDS_SIZE))
echo "  Loading golden text features..."
./socgen/esp/esplink --load -i model_files/text_features_golden.bin -a $(printf "0x%X" $GOLDEN_ADDR)

echo ""
echo "Done! Data loaded to FPGA memory"
echo "  Model binary:       0xB0000000"
echo "  Text embeddings:    $(printf "0x%X" $EMBEDS_ADDR) (78,864 bytes)"
echo "  Golden features:    $(printf "0x%X" $GOLDEN_ADDR) (2,056 bytes)"
