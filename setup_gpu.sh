#!/bin/bash
# ============================================================
# setup_gpu.sh — Set up the RTX 4090 machine for LoRA training
# ============================================================
# Supports both Linux and Windows (WSL/Git Bash)
#
# Usage:
#   chmod +x setup_gpu.sh
#   ./setup_gpu.sh
# ============================================================

set -e

echo "🚀 Setting up LoRA training environment..."
echo ""

# ---- 1. Python virtual environment ----
echo "📦 Creating Python virtual environment..."
python3 -m venv .venv || python -m venv .venv
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements_gpu.txt

# ---- 2. Clone Kohya_ss (sd-scripts) ----
if [ ! -d "sd-scripts" ]; then
    echo "📥 Cloning Kohya sd-scripts..."
    git clone https://github.com/kohya-ss/sd-scripts.git
    cd sd-scripts
    pip install -r requirements.txt
    pip install -e .
    cd ..
else
    echo "✅ sd-scripts already exists"
fi

# ---- 3. Download base model ----
mkdir -p models
MODEL_PATH="models/realvisxl-v5.safetensors"

if [ ! -f "$MODEL_PATH" ]; then
    echo ""
    echo "📥 You need to download the RealVisXL V5.0 base model."
    echo "   Download from: https://civitai.com/models/139562/realvisxl"
    echo "   Save as: $MODEL_PATH"
    echo ""
    echo "   Alternative (automatic via huggingface-cli):"
    echo "   pip install huggingface-hub"
    echo "   huggingface-cli download SG161222/RealVisXL_V5.0 --local-dir models/realvisxl-v5 --include '*.safetensors'"
    echo ""
else
    echo "✅ Base model found: $MODEL_PATH"
fi

# ---- 4. Verify GPU ----
echo ""
echo "🔍 Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'  ✅ GPU: {gpu}')
    print(f'  ✅ VRAM: {vram:.1f} GB')
else:
    print('  ❌ No CUDA GPU found!')
    print('  Make sure NVIDIA drivers and CUDA toolkit are installed.')
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download RealVisXL V5.0 → models/realvisxl-v5.safetensors"
echo "  2. Place training images in data/processed/luna/ and data/processed/scene/"
echo "  3. Run captioning: python scripts/05_caption.py --subject luna --trigger lttluna"
echo "  4. Run training:   ./train.sh luna"
