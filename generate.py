#!/usr/bin/env python3
"""
generate.py — Generate test images using a trained LoRA.

Usage:
    python generate.py --lora data/output/luna_lora.safetensors \
                       --prompt "lttluna, a woman with blonde hair, studio portrait" \
                       --count 4
"""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description="Generate images with trained LoRA")
    parser.add_argument("--lora", type=str, required=True,
                        help="Path to trained LoRA .safetensors file")
    parser.add_argument("--model", type=str, default="models/realvisxl-v5.safetensors",
                        help="Base model path")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Generation prompt (include trigger word!)")
    parser.add_argument("--negative", type=str,
                        default="ugly, blurry, low quality, deformed, bad anatomy, watermark, text",
                        help="Negative prompt")
    parser.add_argument("--count", type=int, default=4,
                        help="Number of images to generate")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=7.0,
                        help="CFG scale")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--lora-scale", type=float, default=0.8,
                        help="LoRA strength (0.0-1.0, default: 0.8)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="generated",
                        help="Output directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  No GPU found. Generation will be very slow.")

    # Load base model
    print(f"📥 Loading base model: {args.model}")
    if args.model.endswith(".safetensors"):
        pipe = StableDiffusionXLPipeline.from_single_file(
            args.model,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
        ).to(device)

    # Load LoRA
    print(f"📥 Loading LoRA: {args.lora}")
    pipe.load_lora_weights(args.lora)

    # Generate
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"\n🎨 Generating {args.count} images...")
    print(f"   Prompt: {args.prompt}")
    print(f"   LoRA scale: {args.lora_scale}")
    print(f"   Steps: {args.steps} | CFG: {args.cfg}")
    print(f"   Size: {args.width}×{args.height}")

    for i in range(args.count):
        seed = args.seed + i if args.seed else None
        if seed:
            generator = torch.Generator(device=device).manual_seed(seed)

        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            width=args.width,
            height=args.height,
            generator=generator,
            cross_attention_kwargs={"scale": args.lora_scale},
        ).images[0]

        filename = f"gen_{i + 1:03d}.png"
        image.save(output_dir / filename)
        print(f"  ✅ Saved: {output_dir / filename}")

    print(f"\n🎉 Done! {args.count} images saved to {output_dir}/")


if __name__ == "__main__":
    main()
