# just-another-image-upscaler

A **4× image upscaler** powered by [DRCT-L](https://github.com/ming053l/DRCT) — a state-of-the-art transformer-based super-resolution model.
Available as a **CLI tool** and a **web interface**.

---

## Features

- **4× upscaling** via DRCT-L super-resolution transformer
- **Web UI** — upload an image in the browser, compare before/after, and download the result
- **CLI** — scriptable command-line interface for batch or automated use
- **GPU acceleration** — automatically uses CUDA when available; falls back to CPU
- **Tiled inference** — process large images without running out of VRAM or RAM
- **fp16 / fp32 precision** — faster inference on GPU with `--precision fp16`
- **Automatic model download** — weights fetched from Hugging Face on first run and cached locally
- **Broad format support** — JPEG, PNG, BMP, TIFF, WebP

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0 |
| NVIDIA GPU *(optional)* | CUDA-compatible |

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/AlexandreGazagnes/just-another-image-upscaler.git
cd just-another-image-upscaler

# Set up the environment (one-time)
bash setup/setup.sh

# Launch the web UI
./run.sh --server
# → open http://localhost:8000

# Or upscale directly from the CLI
./run.sh photo.jpg
```

---

## Web UI

Start the server:

```bash
./run.sh --server
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

- Drag and drop (or click to select) any supported image
- Optionally set precision (`fp32` / `fp16`) and tile size
- Hit **Upscale Image** — a spinner shows while the model runs
- The result is shown side-by-side with the original, with pixel dimensions
- Click **Download Upscaled Image** to save the result

To use a custom port:

```bash
PORT=8080 ./run.sh --server
```

---

## CLI Usage

```bash
./run.sh <image> [options]
```

| Argument | Default | Description |
|---|---|---|
| `image` | *(required)* | Path to the input image |
| `-o`, `--output` | `<name>_upscaled.<ext>` | Output file path |
| `--scale` | `4` | Upscale factor (only 4× is supported) |
| `--precision` | `fp32` | Inference precision: `fp32` or `fp16` |
| `--tile` | *(disabled)* | Tile size in pixels — reduces memory usage for large images |
| `--tile_overlap` | `32` | Overlap between tiles in pixels (requires `--tile`) |

### Examples

```bash
# Basic upscale
./run.sh photo.jpg

# Custom output path
./run.sh photo.jpg -o results/hd_photo.jpg

# Faster inference on GPU
./run.sh photo.jpg --precision fp16

# Tiled inference for large images
./run.sh photo.jpg --tile 256 --tile_overlap 32

# Combine options
./run.sh photo.jpg -o out.jpg --precision fp16 --tile 256
```

---

## How It Works

```
Web UI (browser)              CLI (./run.sh photo.jpg)
       │                               │
       ▼                               │
  server.py  ─────────────────────────┤
       │                               │
       └──────────────────► inference.py
                                  │
                  ┌───────────────┼────────────────┐
                  │               │                │
          load model         _infer_full    _infer_tiled
        (HuggingFace,      (single pass)   (tiled, low
         cached at                          VRAM mode)
     ~/.cache/drct-models)
                  │
                  └─► output image (4× resolution)
```

Model weights are cached at `~/.cache/drct-models` and reused on subsequent runs.

---

## Model

**DRCT-L × 4** — Dense-Residual-Connected Transformer for image super-resolution.

- Architecture: Swin Transformer backbone with dense residual connections
- Input → Output: arbitrary resolution → 4× resolution
- Weights: [`aaronespasa/drct-super-resolution`](https://huggingface.co/aaronespasa/drct-super-resolution) on Hugging Face
- Original repository: [`ming053l/DRCT`](https://github.com/ming053l/DRCT)

---

## Supported Formats

`.jpg` · `.jpeg` · `.png` · `.bmp` · `.tiff` · `.tif` · `.webp`

---

## Troubleshooting

**Out of memory on large images**
Use `--tile 256` (or a smaller value) to process the image in patches. Increase `--tile_overlap` if seams are visible.

**`fp16` has no effect**
Half-precision is silently disabled on CPU. Use a GPU or switch to `--precision fp32`.

**First run is slow**
Model weights (~300 MB) are downloaded once and cached. Subsequent runs skip the download.

---

## Running Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/
```

Tests mock all torch and inference calls — no GPU or model weights required.

---

## License

This project is released under the [MIT License](LICENSE).
DRCT model weights are subject to the license of the original [DRCT repository](https://github.com/ming053l/DRCT).
