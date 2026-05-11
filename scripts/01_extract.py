#!/usr/bin/env python3
"""
01_extract.py — Extract training set ZIPs into organized folders.

Usage:
    python scripts/01_extract.py [--zip1 PATH] [--zip2 PATH]

Defaults to looking for trainingset1.zip and trainingset2.zip in ~/Downloads/
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path


def extract_zip(zip_path: Path, output_dir: Path, label: str) -> int:
    """Extract a ZIP file, flatten structure, keep only images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    count = 0

    print(f"\n📦 Extracting {label}: {zip_path.name}")
    print(f"   → {output_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # Skip directories and hidden files
            if member.endswith("/") or "/__MACOSX" in member or member.startswith("__MACOSX"):
                continue

            ext = Path(member).suffix.lower()
            if ext not in image_extensions:
                continue

            # Extract to flat structure with clean name
            filename = Path(member).name
            target = output_dir / filename

            # Handle duplicates
            if target.exists():
                stem = target.stem
                i = 1
                while target.exists():
                    target = output_dir / f"{stem}_{i}{ext}"
                    i += 1

            # Extract
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1

    print(f"   ✅ Extracted {count} images")
    return count


def main():
    parser = argparse.ArgumentParser(description="Extract training set ZIPs")
    parser.add_argument("--zip1", type=str, default=os.path.expanduser("~/Downloads/trainingset1.zip"),
                        help="Path to trainingset1.zip (Luna)")
    parser.add_argument("--zip2", type=str, default=os.path.expanduser("~/Downloads/trainingset2.zip"),
                        help="Path to trainingset2.zip (Scene)")
    parser.add_argument("--output", type=str, default="data/raw",
                        help="Output base directory")
    args = parser.parse_args()

    base = Path(args.output)

    total = 0

    # Extract trainingset1 → luna
    zip1 = Path(args.zip1)
    if zip1.exists():
        total += extract_zip(zip1, base / "luna", "Luna (single subject)")
    else:
        print(f"⚠️  ZIP not found: {zip1}")

    # Extract trainingset2 → scene
    zip2 = Path(args.zip2)
    if zip2.exists():
        total += extract_zip(zip2, base / "scene", "Scene (multi-subject)")
    else:
        print(f"⚠️  ZIP not found: {zip2}")

    print(f"\n🎉 Total: {total} images extracted")


if __name__ == "__main__":
    main()
