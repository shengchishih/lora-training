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

## RTX 4090 Laptop — Step-by-Step Guide

Follow these steps after cloning this repo on your RTX 4090 laptop.

### Prerequisites

- **OS**: Windows (with WSL2) or Linux (Ubuntu recommended)
- **GPU**: NVIDIA RTX 4090 with 16GB VRAM
- **NVIDIA Driver**: 535+ (check: `nvidia-smi`)
- **CUDA Toolkit**: 12.1+ (check: `nvcc --version`)
- **Python**: 3.10 or 3.11 (check: `python3 --version`)
- **Git**: installed (check: `git --version`)
- **Disk space**: ~30GB free (model + data + cached latents + output)

> **Windows users**: Run all commands inside WSL2 (Ubuntu). Open "Ubuntu" from the Start menu.

---

### Step 1: Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/lora-training.git
cd lora-training
```

### Step 2: Run the setup script

This creates a virtual environment, installs all Python dependencies, and clones Kohya sd-scripts:

```bash
chmod +x setup_gpu.sh train.sh
./setup_gpu.sh
```

This takes ~5-10 minutes. It will:
- Create `.venv/` virtual environment
- Install PyTorch, transformers, diffusers, etc.
- Clone and install [Kohya sd-scripts](https://github.com/kohya-ss/sd-scripts)

### Step 3: Download the base model

Download **RealVisXL V5.0** (SDXL-based, photorealistic):

**Option A — From Civitai (browser download):**
1. Go to https://civitai.com/models/139562/realvisxl
2. Download the V5.0 safetensors file
3. Save it inside `models/realvisxl-v5/` (for example `models/realvisxl-v5/RealVisXL_V5.0_fp16.safetensors`)

**Option B — From HuggingFace (command line):**
```bash
source .venv/bin/activate
pip install huggingface-hub
mkdir -p models
hf download SG161222/RealVisXL_V5.0 \
  --local-dir models/realvisxl-v5 \
  --include "*.safetensors"
```

### Step 4: Transfer the processed images

The data prep was done on the Mac. You need to copy the processed images to this machine.

**Option A — USB drive:**
Copy the entire `data/processed/` folder from the Mac to `data/processed/` on this machine.

**Option B — Re-download from Google Drive and re-run prep:**
```bash
# Download ZIPs to ~/Downloads/ then:
source .venv/bin/activate
pip install Pillow imagehash opencv-python-headless numpy tqdm
python scripts/01_extract.py --zip1 ~/Downloads/trainingset1.zip --zip2 ~/Downloads/trainingset2.zip
python scripts/02_deduplicate.py
python scripts/03_quality_filter.py
python scripts/04_resize.py
```

**Verify the data is in place:**
```bash
ls data/processed/luna/ | head -5    # Should show luna_0001.jpg, luna_0002.jpg, ...
ls data/processed/scene/ | head -5   # Should show scene_0001.jpg, scene_0002.jpg, ...
echo "Luna: $(ls data/processed/luna/*.jpg | wc -l) images"
echo "Scene: $(ls data/processed/scene/*.jpg | wc -l) images"
```

Expected: ~946 luna images, ~112 scene images.

### Step 5: Caption all images (JoyCaption, Qwen, or Florence)

`scripts/05_caption.py` supports three captioning backends:

| Model | Flag | Best for | Notes |
|---|---|---|---|
| [JoyCaption Alpha Two](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava) | `--model joycaption` | Best caption quality for LoRA training | Highest VRAM/RAM usage, slowest on 16GB GPUs, first run downloads ~8GB |
| [Qwen2.5-VL 3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | `--model qwen` | Best speed/quality balance on a 16GB RTX 4090 | First run downloads ~8GB |
| [Florence-2 Large](https://huggingface.co/microsoft/Florence-2-large) | `--model florence` | Lowest resource usage / fallback option | First run downloads ~1.5GB, captions are usually more generic |

```bash
source .venv/bin/activate

# JoyCaption: best caption quality, slowest
python scripts/05_caption.py --subject luna --trigger lttluna --model joycaption

# Qwen: recommended default on a 16GB RTX 4090
python scripts/05_caption.py --subject luna --trigger lttluna --model qwen

# Florence: lightest fallback if VRAM/RAM is tight
python scripts/05_caption.py --subject luna --trigger lttluna --model florence

# Scene dataset example
python scripts/05_caption.py --subject scene --trigger ankdlisla --model qwen
```

To regenerate existing `.txt` captions with a different model, add `--overwrite`.

Each image gets a `.txt` file with a description like:
```
lttluna, a woman with blonde hair wearing a white top, sitting on a bed, soft natural lighting, indoor setting
```

### Step 6: Review captions

Generate an HTML gallery to review the auto-generated captions:

```bash
python scripts/06_review_captions.py --input data/processed/luna
python scripts/06_review_captions.py --input data/processed/scene
```

This opens a browser page showing each image + its caption. Review and edit any captions that are inaccurate by opening the `.txt` files in a text editor.

**What to look for:**
- ❌ Incorrect descriptions (wrong hair color, wrong clothing, etc.)
- ❌ Missing key details
- ❌ Overly generic captions like "a photo of a woman"
- ✅ Good captions describe: appearance, clothing, pose, expression, setting, lighting

### Step 7: Generate regularization images (recommended)

Regularization images prevent the model from "forgetting" general concepts. Without them, prompting "a woman" might always generate Luna even without the trigger word.

```bash
# Generate 200 generic "a woman" images from the base model
python generate.py \
  --model models/realvisxl-v5 \
  --prompt "a woman, photo, portrait, professional lighting, high quality" \
  --negative "ugly, blurry, deformed, watermark, text, low quality" \
  --count 200 \
  --output data/processed/reg_woman \
  --seed 42
# ⏱️ Takes ~15-20 min
```

Then caption them:
```bash
# Simple generic captions for reg images (no trigger word!)
for f in data/processed/reg_woman/*.png; do
  echo "a woman, photo, portrait" > "${f%.png}.txt"
done
```

If `data/processed/reg_woman/` exists, `./train.sh luna` will automatically include it as a regularization subset during training.

> **Note:** If you skip this step, the model will still train but may have "language drift" — generating the trained subject even without the trigger word.

### Step 8: Train the LoRAs

```bash
# Train Luna LoRA (conservative starting point)
./train.sh luna

# Train Scene LoRA (~30-60 min)
./train.sh scene
```

If the first Luna run learns the subject quickly but starts binding too strongly to repetitive caption concepts, run the shorter Luna v2 pass manually:

```bash
source .venv/bin/activate

python sd-scripts/sdxl_train_network.py \
  --config_file configs/lora1_luna_v2.toml \
  --dataset_config configs/lora1_luna_reg_dataset.toml \
  --logging_dir data/logs \
  --log_with tensorboard \
  --log_prefix luna_v2_ \
  --sample_prompts configs/lora1_luna_samples.txt \
  --sample_every_n_epochs 1 \
  --sample_sampler euler_a \
  --sample_at_first
```

`configs/lora1_luna_v2.toml` is a shorter second-pass Luna config:
- Trains for 4 epochs instead of 8
- Saves a checkpoint every epoch
- Lowers the U-Net learning rate to `7.5e-5`
- Trains U-Net LoRA only
- Writes `data/output/luna_lora_v2.safetensors`

`train.sh` now enables local training monitoring by default:
- TensorBoard logs go to `data/logs/`
- Sample images are generated once before training and then on a fixed cadence into `data/output/sample/`
- Prompt sets live in `configs/lora1_luna_samples.txt` and `configs/lora2_scene_samples.txt`

To watch training live in another terminal:

```bash
source .venv/bin/activate
tensorboard --logdir data/logs
```

Then open `http://localhost:6006/` in your browser.

**What happens during training:**
1. Caches all image latents to disk (first epoch only)
2. Trains the LoRA weights over multiple epochs
3. Saves checkpoints on the cadence defined in the config (Luna every 2 epochs, Scene every 5 epochs) to `data/output/`
4. Final model saved as `data/output/luna_lora.safetensors` or `scene_lora.safetensors`

**Monitor training:**
- Watch the loss value — it should gradually decrease
- Check `data/output/sample/` after each sampling interval to catch drift, overfitting, or prompt collapse early
- If loss stops decreasing or starts increasing, the model is overfitting → stop early
- If you get OOM errors, reduce `network_dim` from 32 to 16 in the config

For the Luna v2 run, compare the samples and checkpoints from epochs 1-4 and prefer the earliest checkpoint that preserves identity without collapsing into the same pose, outfit, or room style.

### Step 9: Test the trained LoRA

```bash
# Test Luna LoRA
python generate.py \
  --lora data/output/luna_lora.safetensors \
  --prompt "lttluna, a woman with blonde hair, wearing a red dress, standing in a garden, golden hour lighting" \
  --count 4 \
  --lora-scale 0.8

# Test Scene LoRA
python generate.py \
  --lora data/output/scene_lora.safetensors \
  --prompt "ankdlisla, two women in a professional studio, dramatic lighting" \
  --count 4 \
  --lora-scale 0.8

# Test WITHOUT trigger word (should NOT look like the trained subjects)
python generate.py \
  --lora data/output/luna_lora.safetensors \
  --prompt "a woman with black hair, wearing a business suit, office setting" \
  --count 4 \
  --lora-scale 0.8
```

Generated images are saved to `generated/`.

### Step 10: Tune if needed

| Problem | Fix |
|---|---|
| Images don't look like the subject | Increase epochs, or increase `network_dim` to 64 |
| Images are exact copies of training data | Decrease epochs, or add regularization images |
| Prompting without trigger still generates subject | Add regularization images (Step 7) |
| Outputs keep collapsing toward the same repeated concepts | Clean up captions to reduce repetitive pose / clothing / setting terms, then rerun Luna v2 |
| OOM / out of memory | Reduce `network_dim` to 16, set `train_batch_size` to 1 |
| Colors/contrast look wrong | Add `min_snr_gamma = 5.0` to the config |
| LoRA effect too strong/weak at inference | Adjust `--lora-scale` (try 0.5 to 1.0) |

### After Luna v2

After the second Luna run finishes:
- Compare epoch 1-4 checkpoints in TensorBoard and from `data/output/sample/`
- Prefer the earliest checkpoint that gets the subject right while keeping prompt range
- Test both trigger and non-trigger prompts before choosing a final checkpoint
- If the model still over-binds to narrow recurring caption concepts, edit captions first instead of adding more epochs
- When editing captions, keep the trigger plus stable identity details, then describe only the useful visual facts: hair, face, outfit, pose, setting, and lighting
- Remove or soften repetitive terms that dominate too many captions, especially recurring pose, clothing, room, or anatomy-heavy phrasing that you do not want bound tightly to the trigger
- Keep wording consistent across the dataset for the same traits; for example, use the same hair-color and hairstyle terms instead of many near-synonyms
- Prefer short, literal captions over long descriptive prose; if a detail is not important for prompting, remove it
- Do not put the trigger word in regularization captions; keep those generic, such as `a woman, photo, portrait`
- Keep regularization images enabled for later reruns unless they are clearly hurting likeness

---

## File Structure

```
lora-training/
├── scripts/
│   ├── 01_extract.py          # Extract ZIPs → data/raw/
│   ├── 02_deduplicate.py      # Remove near-duplicates
│   ├── 03_quality_filter.py   # Remove blurry/small images
│   ├── 04_resize.py           # Resize → data/processed/
│   ├── 05_caption.py          # Auto-caption (JoyCaption/Qwen/Florence-2)
│   └── 06_review_captions.py  # HTML gallery for review
├── configs/
│   ├── lora1_luna.toml        # Training config — Luna
│   ├── lora1_luna_v2.toml     # Optional second-pass Luna config
│   ├── lora1_luna_samples.txt # Luna monitoring prompts
│   ├── lora2_scene.toml       # Training config — Scene
│   └── lora2_scene_samples.txt# Scene monitoring prompts
├── setup_gpu.sh               # RTX 4090 setup
├── train.sh                   # Training launcher
├── generate.py                # Inference / test
├── requirements_mac.txt
├── requirements_gpu.txt
└── data/                      # (gitignored — transfer separately)
    ├── raw/                   # Extracted original images
    ├── processed/
    │   ├── luna/              # 946 curated images + .txt captions
    │   ├── scene/             # 112 curated images + .txt captions
    │   └── reg_woman/         # ~200 regularization images (generated)
    ├── logs/                  # TensorBoard event files
    └── output/                # Trained LoRA .safetensors files
```

## Training Config Notes

- **Base model**: RealVisXL V5.0 (SDXL-based, photorealistic)
- **Network rank**: 32 (good balance of quality vs VRAM)
- **Network alpha**: 16 (alpha/dim = 0.5, acts as implicit regularization)
- **Loss**: MSE with Min-SNR weighting (gamma=5.0) for stable training
- **Optimizer**: AdamW8bit (saves ~4GB VRAM)
- **Resolution**: 1024 with aspect ratio bucketing
- **Mixed precision**: fp16 + gradient checkpointing
- **VRAM usage**: ~12-14GB (safe for 16GB)
- **Epochs**: Luna=8 to start, Scene=30
- **Luna v2 refinement**: 4 epochs, U-Net-only LoRA, lower LR (`7.5e-5`), save every epoch

## Tips

1. **Caption quality matters most.** Review captions before training. Bad captions = bad model.
2. **Start with fewer epochs**. Luna starts at 8 epochs; the v2 refinement pass intentionally cuts back to 4 epochs with per-epoch saves.
3. **LoRA scale** at inference: 0.6-0.9 usually works best. Too high = overfit look.
4. **If you get OOM**, reduce `train_batch_size` to 1 and `network_dim` to 16.
5. **Regularization images** prevent language drift. Always use them for character LoRAs.
6. **Save checkpoints** frequently so you can pick the best one before overfitting. Luna now saves every 2 epochs.
7. **After Luna v2**, compare checkpoints 1-4 and choose the earliest one that keeps identity without collapsing into recurring concepts.
8. **If Luna v2 still drifts**, clean up captions before increasing epochs or `network_dim`.
