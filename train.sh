#!/bin/bash
# ============================================================
# train.sh — Launch LoRA training with Kohya sd-scripts
# ============================================================
#
# Usage:
#   ./train.sh luna     # Train Luna LoRA
#   ./train.sh scene    # Train Scene LoRA
# ============================================================

set -e

# Activate venv
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null

SUBJECT=${1:-luna}

case $SUBJECT in
    luna)
        CONFIG="configs/lora1_luna.toml"
        DATA_DIR="data/processed/luna"
        ;;
    scene)
        CONFIG="configs/lora2_scene.toml"
        DATA_DIR="data/processed/scene"
        ;;
    *)
        echo "❌ Unknown subject: $SUBJECT"
        echo "Usage: ./train.sh [luna|scene]"
        exit 1
        ;;
esac

# Pre-flight checks
echo "🔍 Pre-flight checks..."

if [ ! -d "sd-scripts" ]; then
    echo "❌ sd-scripts not found. Run ./setup_gpu.sh first."
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "❌ Config not found: $CONFIG"
    exit 1
fi

# Count images and captions
IMG_COUNT=$(find "$DATA_DIR" -name "*.jpg" -o -name "*.png" | wc -l | tr -d ' ')
TXT_COUNT=$(find "$DATA_DIR" -name "*.txt" | wc -l | tr -d ' ')

echo "  📁 Data: $DATA_DIR"
echo "  🖼️  Images: $IMG_COUNT"
echo "  📝 Captions: $TXT_COUNT"

if [ "$IMG_COUNT" -eq 0 ]; then
    echo "❌ No images found in $DATA_DIR"
    exit 1
fi

if [ "$TXT_COUNT" -eq 0 ]; then
    echo "❌ No caption files found! Run captioning first:"
    echo "   python scripts/05_caption.py --subject $SUBJECT --trigger <your_trigger>"
    exit 1
fi

if [ "$IMG_COUNT" -ne "$TXT_COUNT" ]; then
    echo "⚠️  Warning: Image count ($IMG_COUNT) ≠ Caption count ($TXT_COUNT)"
    echo "   Some images may be missing captions."
    read -p "   Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output dir
mkdir -p data/output

echo ""
echo "🚀 Starting LoRA training: $SUBJECT"
echo "   Config: $CONFIG"
echo ""

# Run training
cd sd-scripts
python sdxl_train_network.py --config_file "../$CONFIG"

echo ""
echo "✅ Training complete!"
echo "   Output: data/output/"
echo ""
echo "Test with:"
echo "   python generate.py --lora data/output/${SUBJECT}_lora.safetensors --prompt '<trigger>, a woman...'"
