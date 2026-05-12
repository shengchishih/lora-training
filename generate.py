#!/usr/bin/env python3
"""
generate.py — Generate test images using a trained LoRA.

Usage:
    python generate.py --lora data/output/luna_lora.safetensors \
                       --prompt "lttluna, a woman with blonde hair, studio portrait" \
                       --count 4
"""

import argparse
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description="Generate images with trained LoRA")
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to trained LoRA .safetensors file (optional — omit to use base model only)")
    parser.add_argument("--model", type=str, default="models/realvisxl-v5",
                        help="Base model path (directory or .safetensors file)")
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

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if device == "mps":
        print("🍎 Using Apple Silicon GPU (MPS). Slower than CUDA but much faster than CPU.")
    elif device == "cpu":
        print("⚠️  No GPU found. Generation will be very slow.")

    # MPS has limited float16 support — use float32 on Mac
    dtype = torch.float32 if device == "mps" else torch.float16

    # Load base model
    print(f"📥 Loading base model: {args.model}")
    if args.model.endswith(".safetensors"):
        import os
        from safetensors import safe_open
        
        model_path = os.path.abspath(args.model)
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=True
        )
        
        # Workaround for diffusers bug: extract text_encoder directly from the safetensors file
        state_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("conditioner.embedders.0.transformer."):
                    new_k = k.replace("conditioner.embedders.0.transformer.", "")
                    state_dict[new_k] = f.get_tensor(k)
        
        # Allocate empty memory for the meta tensor
        pipe.text_encoder.to_empty(device="cpu")
        
        # Clear garbage memory from buffers/parameters
        for param in pipe.text_encoder.parameters():
            param.data.zero_()
        for buf in pipe.text_encoder.buffers():
            buf.data.zero_()
            
        # Re-initialize the position_ids buffer specifically
        if hasattr(pipe.text_encoder, "text_model"):
            pipe.text_encoder.text_model.embeddings.position_ids.data = torch.arange(77).expand((1, -1))
        elif hasattr(pipe.text_encoder, "embeddings"):
            pipe.text_encoder.embeddings.position_ids.data = torch.arange(77).expand((1, -1))
            
        # Load the extracted state dict and move the pipeline to GPU
        pipe.text_encoder.load_state_dict(state_dict, strict=False)
        pipe.to(device)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model,
            torch_dtype=dtype,
        ).to(device)

    # Load LoRA (optional)
    if args.lora:
        print(f"📥 Loading LoRA: {args.lora}")
        pipe.load_lora_weights(args.lora)
    else:
        print("ℹ️  No LoRA specified — using base model only")

    # Generate
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MPS generator must use CPU device
    gen_device = "cpu" if device == "mps" else device
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=gen_device).manual_seed(args.seed)

    print(f"\n🎨 Generating {args.count} images...")
    print(f"   Prompt: {args.prompt}")
    print(f"   LoRA scale: {args.lora_scale}")
    print(f"   Steps: {args.steps} | CFG: {args.cfg}")
    print(f"   Size: {args.width}×{args.height}")

    total_start = time.time()

    for i in range(args.count):
        seed = args.seed + i if args.seed else None
        if seed:
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Only pass LoRA scale when a LoRA is loaded
        pipe_kwargs = dict(
            prompt=args.prompt,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            width=args.width,
            height=args.height,
            generator=generator,
        )
        if args.lora:
            pipe_kwargs["cross_attention_kwargs"] = {"scale": args.lora_scale}

        img_start = time.time()
        image = pipe(**pipe_kwargs).images[0]
        img_time = time.time() - img_start

        filename = f"gen_{i + 1:03d}.png"
        image.save(output_dir / filename)
        print(f"  ✅ [{i + 1}/{args.count}] Saved: {output_dir / filename}  ⏱️ {img_time:.1f}s")

    total_time = time.time() - total_start
    avg_time = total_time / args.count
    print(f"\n🎉 Done! {args.count} images saved to {output_dir}/")
    print(f"⏱️  Total: {total_time:.1f}s | Average: {avg_time:.1f}s per image")


if __name__ == "__main__":
    main()
