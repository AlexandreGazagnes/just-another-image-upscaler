#!/usr/bin/env python3
"""
Core inference module for DRCT Super-Resolution.
Handles model loading, weight download, and image upscaling.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast


MODELS_CACHE = Path.home() / ".cache" / "drct-models"
DEFAULT_MODEL_PATH = MODELS_CACHE / "DRCT_L_x4.pth"
WINDOW_SIZE = 16


def _get_drct_class():
    try:
        from drct.archs.DRCT_arch import DRCT
        return DRCT
    except ImportError:
        pass
    try:
        from basicsr.archs.drct_arch import DRCT
        return DRCT
    except ImportError:
        pass
    sys.exit(
        "Error: Cannot import DRCT architecture.\n"
        "Install it with: pip install git+https://github.com/ming053l/DRCT.git"
    )


def _download_weights(model_path: Path):
    if model_path.exists():
        return
    print("Model weights not found — downloading from Hugging Face (one-time)...")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="aaronespasa/drct-super-resolution",
            filename=model_path.name,
            local_dir=str(model_path.parent),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        sys.exit(f"Error downloading model weights: {exc}")
    if not model_path.exists():
        sys.exit(f"Error: download succeeded but file missing at {model_path}")
    print("Model downloaded successfully.")


def load_model(model_path: Path, scale: int, device: torch.device) -> torch.nn.Module:
    _download_weights(model_path)
    DRCT = _get_drct_class()
    model = DRCT(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=[6] * 12,
        embed_dim=180,
        num_heads=[6] * 12,
        gc=32,
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    checkpoint = torch.load(str(model_path), map_location=device)
    params = checkpoint.get("params", checkpoint)
    model.load_state_dict(params, strict=True)
    model.eval()
    return model.to(device)


def _to_tensor(img_bgr: np.ndarray, fp16: bool, device: torch.device) -> torch.Tensor:
    img = img_bgr.astype(np.float32) / 255.0
    t = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0)
    if fp16:
        t = t.half()
    return t.to(device)


def _to_image(tensor: torch.Tensor) -> np.ndarray:
    out = tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    out = np.transpose(out[[2, 1, 0], :, :], (1, 2, 0))
    return (out * 255.0).round().astype(np.uint8)


def _infer_full(model, img: torch.Tensor, scale: int) -> torch.Tensor:
    _, _, h, w = img.size()
    h_pad = (h // WINDOW_SIZE + 1) * WINDOW_SIZE - h
    w_pad = (w // WINDOW_SIZE + 1) * WINDOW_SIZE - w
    img_p = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h + h_pad, :]
    img_p = torch.cat([img_p, torch.flip(img_p, [3])], 3)[:, :, :, : w + w_pad]
    out = model(img_p)
    return out[..., : h * scale, : w * scale]


def _infer_tiled(
    model,
    img: torch.Tensor,
    scale: int,
    tile: int,
    tile_overlap: int,
    device: torch.device,
) -> torch.Tensor:
    b, c, h, w = img.size()
    tile = min(tile, h, w)
    tile = (tile // WINDOW_SIZE) * WINDOW_SIZE
    if tile <= tile_overlap:
        raise ValueError(f"Tile size ({tile}) must be larger than tile_overlap ({tile_overlap})")
    stride = tile - tile_overlap

    h_pad = (stride - (h - tile) % stride) % stride
    w_pad = (stride - (w - tile) % stride) % stride
    img_p = torch.nn.functional.pad(img, (0, w_pad, 0, h_pad), mode="reflect")
    hp, wp = img_p.shape[2:]

    E = torch.zeros(b, c, hp * scale, wp * scale, dtype=img.dtype, device=device)
    W = torch.zeros_like(E)

    for hi in range(0, hp - tile + 1, stride):
        for wi in range(0, wp - tile + 1, stride):
            patch = img_p[..., hi : hi + tile, wi : wi + tile]
            out = model(patch)
            mask = torch.ones_like(out)
            E[..., hi * scale : (hi + tile) * scale, wi * scale : (wi + tile) * scale] += out
            W[..., hi * scale : (hi + tile) * scale, wi * scale : (wi + tile) * scale] += mask

    return (E / W)[..., : h * scale, : w * scale]


def upscale_image(
    input_path: str,
    output_path: str,
    model: torch.nn.Module,
    scale: int,
    precision: str,
    tile: int | None,
    tile_overlap: int,
    device: torch.device,
):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Error: cannot read image '{input_path}'")

    fp16 = precision == "fp16" and device.type != "cpu"
    if fp16:
        model = model.half()

    img_tensor = _to_tensor(img, fp16, device)

    with torch.no_grad():
        with autocast("cuda", enabled=fp16):
            if tile is None:
                out_tensor = _infer_full(model, img_tensor, scale)
            else:
                out_tensor = _infer_tiled(model, img_tensor, scale, tile, tile_overlap, device)

    result = _to_image(out_tensor)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, result)
