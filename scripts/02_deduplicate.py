#!/usr/bin/env python3
"""
02_deduplicate.py — Remove near-duplicate images using perceptual hashing.

OnlyFans sets often have 5-10 nearly identical shots. This removes them,
keeping the highest resolution version from each duplicate group.

Usage:
    python scripts/02_deduplicate.py [--input data/raw] [--threshold 8]
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

from PIL import Image
import imagehash
from tqdm import tqdm


def compute_hash(image_path: Path) -> tuple:
    """Compute perceptual hash (average + difference) for an image."""
    try:
        img = Image.open(image_path)
        ahash = imagehash.average_hash(img, hash_size=12)
        dhash = imagehash.dhash(img, hash_size=12)
        return ahash, dhash, img.size[0] * img.size[1]  # hash + pixel count
    except Exception as e:
        print(f"  ⚠️  Error processing {image_path.name}: {e}")
        return None, None, 0


def find_duplicates(image_dir: Path, threshold: int = 8) -> list:
    """Find groups of near-duplicate images."""
    images = sorted(image_dir.glob("*"))
    images = [f for f in images if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

    if not images:
        return []

    print(f"  Computing hashes for {len(images)} images...")
    hashes = []
    for img_path in tqdm(images, desc="  Hashing"):
        ahash, dhash, pixels = compute_hash(img_path)
        if ahash is not None:
            hashes.append((img_path, ahash, dhash, pixels))

    # Group duplicates
    duplicate_groups = []
    used = set()

    for i, (path_i, ah_i, dh_i, px_i) in enumerate(hashes):
        if i in used:
            continue
        group = [(path_i, px_i)]
        for j in range(i + 1, len(hashes)):
            if j in used:
                continue
            path_j, ah_j, dh_j, px_j = hashes[j]
            # Both hashes must be similar
            if (ah_i - ah_j) < threshold and (dh_i - dh_j) < threshold:
                group.append((path_j, px_j))
                used.add(j)
        if len(group) > 1:
            duplicate_groups.append(group)
            used.add(i)

    return duplicate_groups


def deduplicate_folder(folder: Path, threshold: int, dry_run: bool = False) -> int:
    """Remove duplicates from a folder, keeping highest resolution."""
    label = folder.name
    print(f"\n🔍 Deduplicating: {label}/")

    before_count = len([f for f in folder.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    groups = find_duplicates(folder, threshold)

    if not groups:
        print(f"  ✅ No duplicates found ({before_count} unique images)")
        return 0

    # Create duplicates folder
    dup_dir = folder.parent / f"_duplicates_{label}"
    dup_dir.mkdir(exist_ok=True)

    removed = 0
    for group in groups:
        # Sort by pixel count (largest first)
        group.sort(key=lambda x: x[1], reverse=True)
        keeper = group[0]
        for img_path, px in group[1:]:
            if dry_run:
                print(f"  Would remove: {img_path.name} (dup of {keeper[0].name})")
            else:
                img_path.rename(dup_dir / img_path.name)
            removed += 1

    after_count = before_count - removed
    print(f"  ✅ Removed {removed} duplicates ({before_count} → {after_count} images)")
    if not dry_run:
        print(f"  📁 Duplicates moved to: _duplicates_{label}/")

    return removed


def main():
    parser = argparse.ArgumentParser(description="Remove near-duplicate images")
    parser.add_argument("--input", type=str, default="data/raw",
                        help="Input directory containing image folders")
    parser.add_argument("--threshold", type=int, default=8,
                        help="Hash distance threshold (lower = stricter, default: 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be removed without actually removing")
    args = parser.parse_args()

    base = Path(args.input)
    total_removed = 0

    for subfolder in sorted(base.iterdir()):
        if subfolder.is_dir() and not subfolder.name.startswith("_"):
            total_removed += deduplicate_folder(subfolder, args.threshold, args.dry_run)

    print(f"\n🎉 Total duplicates removed: {total_removed}")


if __name__ == "__main__":
    main()
