#!/usr/bin/env python3
"""
04_resize.py — Resize images for SDXL training.

Resizes so the shortest side = target resolution. Preserves aspect ratio.
Kohya_ss handles aspect ratio bucketing during training, so we don't force square crops.

Usage:
    python scripts/04_resize.py [--input data/raw] [--output data/processed] [--resolution 1024]
"""

import argparse
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


def resize_image(image_path: Path, output_path: Path, target_resolution: int) -> bool:
    """Resize an image so shortest side = target_resolution. Preserve aspect ratio."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed (handle RGBA, palette, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Fix orientation from EXIF
            img = ImageOps.exif_transpose(img)

            w, h = img.size
            shortest = min(w, h)

            if shortest <= target_resolution:
                # Image is already at or below target — just save as-is
                img.save(output_path, "JPEG", quality=95)
                return True

            # Resize so shortest side = target_resolution
            scale = target_resolution / shortest
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Round to nearest multiple of 8 (required by SD models)
            new_w = (new_w // 8) * 8
            new_h = (new_h // 8) * 8

            img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(output_path, "JPEG", quality=95)
            return True

    except Exception as e:
        print(f"  ⚠️  Error processing {image_path.name}: {e}")
        return False


def process_folder(input_dir: Path, output_dir: Path, resolution: int) -> int:
    """Process all images in a folder."""
    label = input_dir.name
    print(f"\n📐 Resizing: {label}/ → resolution {resolution}")

    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in input_dir.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    if not images:
        print(f"  ⚠️  No images found")
        return 0

    count = 0
    for i, img_path in enumerate(tqdm(images, desc="  Resizing"), 1):
        # Clean filename: luna_001.jpg, scene_001.jpg
        new_name = f"{label}_{i:04d}.jpg"
        output_path = output_dir / new_name

        if resize_image(img_path, output_path, resolution):
            count += 1

    print(f"  ✅ Resized {count} images → {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Resize images for SDXL training")
    parser.add_argument("--input", type=str, default="data/raw",
                        help="Input directory with image folders")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Output directory for resized images")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Target shortest side (default: 1024)")
    args = parser.parse_args()

    input_base = Path(args.input)
    output_base = Path(args.output)
    total = 0

    for subfolder in sorted(input_base.iterdir()):
        if subfolder.is_dir() and not subfolder.name.startswith("_"):
            output_dir = output_base / subfolder.name
            total += process_folder(subfolder, output_dir, args.resolution)

    print(f"\n🎉 Total: {total} images resized and saved to {output_base}/")


if __name__ == "__main__":
    main()
