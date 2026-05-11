# SDXL LoRA Training Pipeline

Train character-specific LoRAs on SDXL (RealVisXL) using a two-machine workflow.

## Architecture

| Step | Machine | Command |
|---|---|---|
| Data prep (extract, dedup, filter, resize) | Mac | `python scripts/01_extract.py` → `04_resize.py` |
| Captioning | RTX 4090 | `python scripts/05_caption.py` |
| Training | RTX 4090 | `./train.sh luna` |
| Inference | RTX 4090 | `python generate.py --lora ... --prompt ...` |

## LoRAs

| Name | Trigger | Source | Description |
|---|---|---|---|
| Luna | `lttluna` | trainingset1 | Single subject — 1,160 photos |
| Scene | `ankdlisla` | trainingset2 | Two subjects — 112 photos |

---

## Mac Setup (Data Preparation)

```bash
cd lora-training

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements_mac.txt

# Run the pipeline
python scripts/01_extract.py                          # Extract ZIPs
python scripts/02_deduplicate.py                      # Remove duplicates
python scripts/03_quality_filter.py                   # Remove bad images
python scripts/04_resize.py                           # Resize for training
```

After this, `data/processed/luna/` and `data/processed/scene/` will contain
the curated, resized images ready for captioning.

## RTX 4090 Setup (Training)

```bash
cd lora-training

# One-time setup
chmod +x setup_gpu.sh train.sh
./setup_gpu.sh

# Download base model
# → Download RealVisXL V5.0 from https://civitai.com/models/139562/realvisxl
# → Save as models/realvisxl-v5.safetensors

# Transfer processed images from Mac to data/processed/

# Caption images (uses JoyCaption)
python scripts/05_caption.py --subject luna --trigger lttluna
python scripts/05_caption.py --subject scene --trigger ankdlisla

# Review captions
python scripts/06_review_captions.py --input data/processed/luna

# Train!
./train.sh luna     # ~1-2 hours
./train.sh scene    # ~30-60 min

# Test the results
python generate.py \
  --lora data/output/luna_lora.safetensors \
  --prompt "lttluna, a woman with blonde hair, wearing a white dress, studio portrait, soft lighting" \
  --count 4
```

## File Structure

```
lora-training/
├── scripts/
│   ├── 01_extract.py          # Extract ZIPs → data/raw/
│   ├── 02_deduplicate.py      # Remove near-duplicates
│   ├── 03_quality_filter.py   # Remove blurry/small images
│   ├── 04_resize.py           # Resize → data/processed/
│   ├── 05_caption.py          # Auto-caption (JoyCaption/Florence-2)
│   └── 06_review_captions.py  # HTML gallery for review
├── configs/
│   ├── lora1_luna.toml        # Training config — Luna
│   └── lora2_scene.toml       # Training config — Scene
├── setup_gpu.sh               # RTX 4090 setup
├── train.sh                   # Training launcher
├── generate.py                # Inference / test
├── requirements_mac.txt
├── requirements_gpu.txt
└── data/                      # (gitignored — transfer separately)
    ├── raw/
    ├── processed/
    │   ├── luna/              # Images + .txt captions
    │   └── scene/
    └── output/                # Trained LoRA weights
```

## Training Config Notes

- **Base model**: RealVisXL V5.0 (SDXL-based, photorealistic)
- **Network rank**: 32 (good balance of quality vs VRAM)
- **Optimizer**: AdamW8bit (saves ~4GB VRAM)
- **Resolution**: 1024 with aspect ratio bucketing
- **Mixed precision**: fp16 + gradient checkpointing
- **VRAM usage**: ~12-14GB (safe for 16GB)

## Tips

1. **Caption quality matters most.** Review captions before training.
2. **Start with fewer epochs** (10-15) and increase if results lack detail.
3. **LoRA scale** at inference: 0.6-0.9 usually works best. Too high = overfit look.
4. **If you get OOM**, reduce `train_batch_size` to 1 and `network_dim` to 16.
