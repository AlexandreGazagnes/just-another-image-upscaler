#!/bin/bash
set -e

VENV_DIR=".venv"
PORT="${PORT:-8000}"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Error: virtual environment not found at '$VENV_DIR'."
    echo "Run setup/setup.sh first to create it."
    exit 1
fi

if [[ "$1" == "--server" ]]; then
    echo "Starting web UI on http://localhost:${PORT}"
    exec "$VENV_DIR/bin/python" -m uvicorn server:app --host 0.0.0.0 --port "$PORT"
fi

if [[ $# -eq 0 ]]; then
    echo "Usage:"
    echo "  ./run.sh <image> [options]   — upscale an image via CLI"
    echo "  ./run.sh --server            — launch the web UI"
    echo "  PORT=8080 ./run.sh --server  — web UI on a custom port"
    echo ""
    echo "CLI options:"
    echo "  -o, --output <path>      output file path (default: <name>_upscaled.<ext>)"
    echo "  --precision fp32|fp16    inference precision (default: fp32)"
    echo "  --tile <size>            tile size in pixels, e.g. 256 (reduces VRAM usage)"
    echo "  --tile_overlap <px>      tile overlap in pixels (default: 32)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh photo.jpg"
    echo "  ./run.sh photo.jpg -o out.jpg --precision fp16"
    echo "  ./run.sh photo.jpg --tile 256"
    exit 1
fi

exec "$VENV_DIR/bin/python" upscale.py "$@"
