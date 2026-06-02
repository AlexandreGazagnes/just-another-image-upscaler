#!/bin/bash
# just-another-image-upscaler — Environment setup
# Exact versions tested and validated
# Usage: bash setup/setup.sh

set -e

# ── Helpers ──────────────────────────────────────────────────────────────────

BOLD="\033[1m"; RESET="\033[0m"
RED="\033[31m"; GREEN="\033[32m"; YELLOW="\033[33m"; CYAN="\033[36m"

header() { echo -e "\n${BOLD}${CYAN}== $* ==${RESET}"; }
info()   { echo -e "  ${CYAN}→ $*${RESET}"; }
ok()     { echo -e "  ${GREEN}✓ $*${RESET}"; }
warn()   { echo -e "  ${YELLOW}⚠ $*${RESET}"; }
err()    { echo -e "  ${RED}✗ $*${RESET}"; exit 1; }

PYTHON=$(command -v python3 || command -v python || err "Python 3 not found")
info "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

# ── .venv — DRCT Super-Resolution ────────────────────────────────────────────
header ".venv — DRCT Super-Resolution + PyTorch 2.4.1 CUDA 12.1"

if [ -d ".venv" ]; then
    warn ".venv already exists — delete it to reinstall"
else
    $PYTHON -m venv .venv
    source .venv/bin/activate

    pip install --quiet --upgrade pip

    info "PyTorch 2.4.1 + CUDA 12.1..."
    pip install --quiet \
        torch==2.4.1 \
        torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121

    info "Core dependencies..."
    pip install --quiet \
        "numpy==1.26.4" \
        "einops==0.6.0" \
        "opencv-python-headless==4.10.0.84" \
        "huggingface_hub==0.24.6" \
        "basicsr==1.4.2"

    info "DRCT architecture (from GitHub)..."
    pip install --quiet "git+https://github.com/ming053l/DRCT.git"

    # Verify
    python3 -c "
import numpy, torch, cv2, einops, huggingface_hub, basicsr

assert numpy.__version__ < '2', f'NumPy must be <2, got {numpy.__version__}'
assert torch.__version__.startswith('2.4.1'), f'Wrong torch: {torch.__version__}'

print('  numpy          :', numpy.__version__)
print('  torch          :', torch.__version__)
print('  CUDA available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU            :', torch.cuda.get_device_name(0))
print('  opencv         :', cv2.__version__)
print('  einops         :', einops.__version__)
print('  huggingface_hub:', huggingface_hub.__version__)
print('  basicsr        :', basicsr.__version__)

try:
    from drct.archs.DRCT_arch import DRCT
    print('  drct           : OK (drct.archs)')
except ImportError:
    from basicsr.archs.drct_arch import DRCT
    print('  drct           : OK (basicsr.archs fallback)')
" || err ".venv verification failed"

    deactivate
    ok ".venv ready"
fi

# ── Tests ─────────────────────────────────────────────────────────────────────
header "Running test suite"

source .venv/bin/activate

info "Installing pytest..."
pip install --quiet pytest

info "Running tests/..."
python3 -m pytest tests/ -v || err "Tests failed"

deactivate
ok "All tests passed"

echo ""
ok "Setup complete. Run with: ./run.sh photo.jpg"
