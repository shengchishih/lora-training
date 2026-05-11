#!/usr/bin/env python3
"""
06_review_captions.py — Generate an HTML gallery to review image captions.

Opens in your browser. Shows each image with its caption for manual review.

Usage:
    python scripts/06_review_captions.py [--input data/processed/luna]
"""

import argparse
import base64
import webbrowser
from pathlib import Path


def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 for embedding in HTML."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = image_path.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{data}"


def generate_html(input_dir: Path, output_path: Path):
    """Generate an HTML gallery for caption review."""
    images = sorted([f for f in input_dir.glob("*.jpg")] + [f for f in input_dir.glob("*.png")])

    if not images:
        print(f"❌ No images found in {input_dir}")
        return

    print(f"📸 Generating gallery for {len(images)} images...")

    html_parts = ["""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Caption Review — {folder}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
h1 {{ color: #e94560; text-align: center; }}
.stats {{ text-align: center; color: #888; margin-bottom: 30px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }}
.card {{ background: #16213e; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }}
.card img {{ width: 100%; height: 300px; object-fit: cover; }}
.caption {{ padding: 15px; font-size: 14px; line-height: 1.5; color: #ccc; }}
.caption .trigger {{ color: #e94560; font-weight: bold; }}
.filename {{ padding: 0 15px 10px; font-size: 12px; color: #555; }}
.no-caption {{ color: #ff6b6b; font-style: italic; }}
</style>
</head>
<body>
<h1>Caption Review — {folder}</h1>
<div class="stats">{count} images</div>
<div class="grid">
""".format(folder=input_dir.name, count=len(images))]

    captioned = 0
    uncaptioned = 0

    for img_path in images:
        txt_path = img_path.with_suffix(".txt")
        img_b64 = image_to_base64(img_path)

        if txt_path.exists():
            caption = txt_path.read_text(encoding="utf-8").strip()
            # Highlight trigger word
            parts = caption.split(", ", 1)
            if len(parts) == 2:
                caption_html = f'<span class="trigger">{parts[0]}</span>, {parts[1]}'
            else:
                caption_html = caption
            captioned += 1
        else:
            caption_html = '<span class="no-caption">⚠️ No caption file found</span>'
            uncaptioned += 1

        html_parts.append(f"""
<div class="card">
    <img src="{img_b64}" alt="{img_path.name}">
    <div class="caption">{caption_html}</div>
    <div class="filename">{img_path.name}</div>
</div>
""")

    html_parts.append("</div></body></html>")

    output_path.write_text("".join(html_parts), encoding="utf-8")
    print(f"✅ Gallery saved to: {output_path}")
    print(f"   Captioned: {captioned} | Uncaptioned: {uncaptioned}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate HTML caption review gallery")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory with images and captions (e.g., data/processed/luna)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML file path (default: review_<folder>.html)")
    parser.add_argument("--open", action="store_true", default=True,
                        help="Open in browser after generating")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"❌ Directory not found: {input_dir}")
        return

    output = Path(args.output) if args.output else Path(f"review_{input_dir.name}.html")
    result = generate_html(input_dir, output)

    if result and args.open:
        webbrowser.open(f"file://{result.resolve()}")


if __name__ == "__main__":
    main()
