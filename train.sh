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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Activate venv
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null

SUBJECT=${1:-luna}

case $SUBJECT in
    luna)
        CONFIG="configs/lora1_luna.toml"
        DATA_DIR="data/processed/luna"
        DEFAULT_DATASET_CONFIG="configs/lora1_luna_dataset.toml"
        REG_DATASET_CONFIG="configs/lora1_luna_reg_dataset.toml"
        REG_DIR="data/processed/reg_woman"
        SAMPLE_PROMPTS="configs/lora1_luna_samples.txt"
        SAMPLE_EVERY_N_EPOCHS=2
        ;;
    scene)
        CONFIG="configs/lora2_scene.toml"
        DATA_DIR="data/processed/scene"
        DEFAULT_DATASET_CONFIG="configs/lora2_scene_dataset.toml"
        REG_DATASET_CONFIG=""
        REG_DIR=""
        SAMPLE_PROMPTS="configs/lora2_scene_samples.txt"
        SAMPLE_EVERY_N_EPOCHS=5
        ;;
    *)
        echo "❌ Unknown subject: $SUBJECT"
        echo "Usage: ./train.sh [luna|scene]"
        exit 1
        ;;
esac

LOGGING_DIR="data/logs"
LOG_WITH="tensorboard"
LOG_PREFIX="${SUBJECT}_"
SAMPLE_SAMPLER="euler_a"

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

DATASET_CONFIG="$DEFAULT_DATASET_CONFIG"
if [ -n "$REG_DIR" ] && [ -d "$REG_DIR" ]; then
    REG_IMG_COUNT=$(find "$REG_DIR" -maxdepth 1 \( -name "*.jpg" -o -name "*.png" \) | wc -l | tr -d ' ')
    if [ "$REG_IMG_COUNT" -gt 0 ]; then
        DATASET_CONFIG="$REG_DATASET_CONFIG"
    fi
fi

if [ ! -f "$DATASET_CONFIG" ]; then
    echo "❌ Dataset config not found: $DATASET_CONFIG"
    exit 1
fi

if [ ! -f "$SAMPLE_PROMPTS" ]; then
    echo "❌ Sample prompt file not found: $SAMPLE_PROMPTS"
    exit 1
fi

# Count images and captions
IMG_COUNT=$(find "$DATA_DIR" -name "*.jpg" -o -name "*.png" | wc -l | tr -d ' ')
TXT_COUNT=$(find "$DATA_DIR" -name "*.txt" | wc -l | tr -d ' ')

echo "  📁 Data: $DATA_DIR"
echo "  🖼️  Images: $IMG_COUNT"
echo "  📝 Captions: $TXT_COUNT"
echo "  🧩 Dataset config: $DATASET_CONFIG"
echo "  📊 TensorBoard logs: $LOGGING_DIR"
echo "  🧪 Sample prompts: $SAMPLE_PROMPTS"
echo "  🔁 Sample cadence: every $SAMPLE_EVERY_N_EPOCHS epoch(s) + before training"

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

if [ "$DATASET_CONFIG" = "$REG_DATASET_CONFIG" ]; then
    REG_TXT_COUNT=$(find "$REG_DIR" -maxdepth 1 -name "*.txt" | wc -l | tr -d ' ')
    echo "  🛡️  Regularization: $REG_DIR"
    echo "  🖼️  Reg images: $REG_IMG_COUNT"
    echo "  📝 Reg captions: $REG_TXT_COUNT"
    if [ "$REG_IMG_COUNT" -eq 0 ]; then
        echo "❌ No regularization images found in $REG_DIR"
        exit 1
    fi
fi

# Create output dir
mkdir -p data/output
mkdir -p "$LOGGING_DIR"

echo ""
echo "🚀 Starting LoRA training: $SUBJECT"
echo "   Config: $CONFIG"
printf '   \033[1;32m📊 VIEW PROGRESS: tensorboard --logdir %s\033[0m\n' "$LOGGING_DIR"
echo ""

# Run training
python sd-scripts/sdxl_train_network.py \
    --config_file "$CONFIG" \
    --dataset_config "$DATASET_CONFIG" \
    --logging_dir "$LOGGING_DIR" \
    --log_with "$LOG_WITH" \
    --log_prefix "$LOG_PREFIX" \
    --sample_prompts "$SAMPLE_PROMPTS" \
    --sample_every_n_epochs "$SAMPLE_EVERY_N_EPOCHS" \
    --sample_sampler "$SAMPLE_SAMPLER" \
    --sample_at_first

echo ""
echo "✅ Training complete!"
echo "   Output: data/output/"
echo "   Samples: data/output/sample/"
echo "   Logs: $LOGGING_DIR/"
echo ""
echo "Test with:"
echo "   python generate.py --lora data/output/${SUBJECT}_lora.safetensors --prompt '<trigger>, a woman...'"
