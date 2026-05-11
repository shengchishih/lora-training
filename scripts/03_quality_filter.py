#!/usr/bin/env python3
"""
03_quality_filter.py — Remove low-quality images (blurry, too small, corrupted).

Usage:
    python scripts/03_quality_filter.py [--input data/raw] [--min-size 512] [--blur-threshold 50]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def check_blur(image_path: Path, threshold: float = 50.0) -> tuple:
    """Check if an image is blurry using Laplacian variance."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True, 0.0  # Can't read = treat as bad
        variance = cv2.Laplacian(img, cv2.CV_64F).var()
        return variance < threshold, variance
    except Exception:
        return True, 0.0


def check_resolution(image_path: Path, min_size: int = 512) -> tuple:
    """Check if image meets minimum resolution."""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            shortest = min(w, h)
            return shortest < min_size, w, h
    except Exception:
        return True, 0, 0


def filter_folder(folder: Path, min_size: int, blur_threshold: float, dry_run: bool) -> dict:
    """Filter images in a folder by quality."""
    label = folder.name
    print(f"\n🔎 Quality filtering: {label}/")

    images = sorted([f for f in folder.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    if not images:
        print(f"  ⚠️  No images found")
        return {"kept": 0, "removed_blur": 0, "removed_small": 0, "removed_corrupt": 0}

    filter_dir = folder.parent / f"_filtered_{label}"
    filter_dir.mkdir(exist_ok=True)

    stats = {"kept": 0, "removed_blur": 0, "removed_small": 0, "removed_corrupt": 0}

    for img_path in tqdm(images, desc="  Checking"):
        # Check if corrupted
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            stats["removed_corrupt"] += 1
            if not dry_run:
                img_path.rename(filter_dir / img_path.name)
            continue

        # Check resolution
        too_small, w, h = check_resolution(img_path, min_size)
        if too_small:
            stats["removed_small"] += 1
            if not dry_run:
                img_path.rename(filter_dir / img_path.name)
            continue

        # Check blur
        is_blurry, variance = check_blur(img_path, blur_threshold)
        if is_blurry:
            stats["removed_blur"] += 1
            if not dry_run:
                img_path.rename(filter_dir / img_path.name)
            continue

        stats["kept"] += 1

    total_removed = stats["removed_blur"] + stats["removed_small"] + stats["removed_corrupt"]
    print(f"  ✅ Kept: {stats['kept']} | Removed: {total_removed}")
    print(f"     Blurry: {stats['removed_blur']} | Too small: {stats['removed_small']} | Corrupt: {stats['removed_corrupt']}")
    if not dry_run and total_removed > 0:
        print(f"  📁 Filtered images moved to: _filtered_{label}/")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Filter out low-quality images")
    parser.add_argument("--input", type=str, default="data/raw",
                        help="Input directory containing image folders")
    parser.add_argument("--min-size", type=int, default=512,
                        help="Minimum shortest side in pixels (default: 512)")
    parser.add_argument("--blur-threshold", type=float, default=50.0,
                        help="Laplacian variance threshold (lower = more strict, default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be removed without removing")
    args = parser.parse_args()

    base = Path(args.input)
    for subfolder in sorted(base.iterdir()):
        if subfolder.is_dir() and not subfolder.name.startswith("_"):
            filter_folder(subfolder, args.min_size, args.blur_threshold, args.dry_run)


if __name__ == "__main__":
    main()
