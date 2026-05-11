#!/usr/bin/env python3
"""
05_caption.py — Auto-caption training images using JoyCaption or Florence-2.

Runs on the RTX 4090 machine. Generates a .txt file alongside each image
with a natural language description + trigger word.

Usage:
    python scripts/05_caption.py --subject luna --trigger lttluna [--model joycaption]
    python scripts/05_caption.py --subject scene --trigger ankdlisla [--model florence]
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def load_joycaption():
    """Load JoyCaption Alpha Two model."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_id = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
    print(f"📥 Loading JoyCaption from {model_id}...")
    print("   (First run downloads ~8GB — subsequent runs use cache)")

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return processor, model, "joycaption"


def load_florence():
    """Load Florence-2 model (smaller, works on CPU/MPS too)."""
    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = "microsoft/Florence-2-large"
    print(f"📥 Loading Florence-2 from {model_id}...")
    print("   (First run downloads ~1.5GB)")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return processor, model, "florence"


def caption_joycaption(image_path: Path, processor, model, trigger: str) -> str:
    """Generate caption using JoyCaption."""
    image = Image.open(image_path).convert("RGB")

    prompt = "Write a detailed description of this image for AI image generation training. " \
             "Describe the person's appearance, clothing, pose, expression, setting, and lighting. " \
             "Be specific and descriptive. Do not include any ethical commentary."

    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    # Add the image
    inputs["pixel_values"] = processor.image_processor(image, return_tensors="pt")["pixel_values"].to(
        model.device, dtype=model.dtype
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    # Decode only new tokens
    caption = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Prepend trigger word
    return f"{trigger}, {caption}"


def caption_florence(image_path: Path, processor, model, trigger: str) -> str:
    """Generate caption using Florence-2."""
    image = Image.open(image_path).convert("RGB")
    device = next(model.parameters()).device

    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    # Match model dtype
    if inputs.get("pixel_values") is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    # Florence returns the prompt prefix — remove it
    if caption.startswith(prompt.replace("<", "").replace(">", "")):
        caption = caption[len(prompt) - 2:].strip()

    return f"{trigger}, {caption}"


def main():
    parser = argparse.ArgumentParser(description="Auto-caption training images")
    parser.add_argument("--subject", type=str, required=True, choices=["luna", "scene"],
                        help="Which subject folder to caption")
    parser.add_argument("--trigger", type=str, required=True,
                        help="Trigger word to prepend (e.g., 'lttluna')")
    parser.add_argument("--model", type=str, default="joycaption", choices=["joycaption", "florence"],
                        help="Captioning model (default: joycaption)")
    parser.add_argument("--input", type=str, default="data/processed",
                        help="Base directory with processed images")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing captions")
    args = parser.parse_args()

    input_dir = Path(args.input) / args.subject
    if not input_dir.exists():
        print(f"❌ Directory not found: {input_dir}")
        return

    images = sorted([f for f in input_dir.glob("*.jpg")] + [f for f in input_dir.glob("*.png")])
    if not images:
        print(f"❌ No images found in {input_dir}")
        return

    print(f"\n🏷️  Captioning {len(images)} images in {input_dir}/")
    print(f"   Trigger word: {args.trigger}")
    print(f"   Model: {args.model}")

    # Load model
    if args.model == "joycaption":
        processor, model, model_type = load_joycaption()
        caption_fn = caption_joycaption
    else:
        processor, model, model_type = load_florence()
        caption_fn = caption_florence

    # Caption all images
    captioned = 0
    skipped = 0
    for img_path in tqdm(images, desc="  Captioning"):
        txt_path = img_path.with_suffix(".txt")

        if txt_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            caption = caption_fn(img_path, processor, model, args.trigger)
            txt_path.write_text(caption, encoding="utf-8")
            captioned += 1
        except Exception as e:
            print(f"\n  ⚠️  Error captioning {img_path.name}: {e}")

    print(f"\n✅ Captioned: {captioned} | Skipped (existing): {skipped}")
    print(f"📁 Captions saved alongside images in {input_dir}/")


if __name__ == "__main__":
    main()
