#!/usr/bin/env python3
"""
upscale.py — CLI to upscale images 4× using DRCT Super-Resolution.

Usage:
    python upscale.py photo.jpg
    python upscale.py photo.jpg -o result.jpg
    python upscale.py photo.jpg --precision fp16 --tile 256
"""

import argparse
import sys
from pathlib import Path

MODELS_CACHE = Path.home() / ".cache" / "drct-models"
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def die(msg: str):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def run_upscale(
    input_path: Path,
    output_path: Path,
    scale: int,
    precision: str,
    tile,
    tile_overlap: int,
):
    import torch
    from inference import load_model, upscale_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Precision: {precision}")

    model_path = MODELS_CACHE / "DRCT_L_x4.pth"
    MODELS_CACHE.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model(model_path, scale, device)

    print(f"Upscaling '{Path(input_path).name}' ({scale}x, {precision})...")
    upscale_image(
        str(input_path),
        str(output_path),
        model,
        scale,
        precision,
        tile,
        tile_overlap,
        device,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Upscale images 4× using DRCT Super-Resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python upscale.py photo.jpg
  python upscale.py photo.jpg -o hd_photo.jpg
  python upscale.py photo.jpg --precision fp16
  python upscale.py photo.jpg --tile 256 --tile_overlap 32
        """,
    )
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path (default: <name>_upscaled.<ext> next to the input)",
    )
    parser.add_argument(
        "--scale", type=int, default=4, choices=[4],
        help="Upscale factor — only 4× supported (default: 4)",
    )
    parser.add_argument(
        "--precision", default="fp32", choices=["fp32", "fp16"],
        help="Inference precision; fp16 is faster on GPU (default: fp32)",
    )
    parser.add_argument(
        "--tile", type=int, default=None,
        help="Tile size for large images (e.g. 256). Reduces VRAM / RAM usage.",
    )
    parser.add_argument(
        "--tile_overlap", type=int, default=32,
        help="Tile overlap in pixels (default: 32, ignored without --tile)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        die(f"Input file not found: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_FORMATS:
        die(
            f"Unsupported format '{input_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    output_path = (
        Path(args.output).resolve()
        if args.output
        else input_path.parent / f"{input_path.stem}_upscaled{input_path.suffix}"
    )

    run_upscale(input_path, output_path, args.scale, args.precision, args.tile, args.tile_overlap)
    print(f"\nDone! Saved to: {output_path}")


if __name__ == "__main__":
    main()
