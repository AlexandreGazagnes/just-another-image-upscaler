#!/usr/bin/env python3
import shutil
import sys
import tempfile
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from inference import DEFAULT_MODEL_PATH, MODELS_CACHE, load_model, upscale_image
from upscale import SUPPORTED_FORMATS

SCALE = 4
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

_state: dict = {}
_inference_lock = threading.Lock()

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".webp": "image/webp",
}

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DRCT Image Upscaler</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0f0f13;
    --surface: #1a1a22;
    --border: #2e2e3e;
    --accent: #7c6fef;
    --accent-hover: #9087f5;
    --text: #e8e8f0;
    --muted: #8888a8;
    --success: #4ade80;
    --error: #f87171;
    --radius: 12px;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 20px 60px;
  }

  header {
    text-align: center;
    margin-bottom: 36px;
  }
  header h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
  header p { color: var(--muted); margin-top: 6px; font-size: 0.95rem; }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px;
    width: 100%;
    max-width: 700px;
  }

  /* Drop zone */
  #drop-zone {
    border: 2px dashed var(--border);
    border-radius: 10px;
    padding: 40px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
  }
  #drop-zone:hover, #drop-zone.drag-over {
    border-color: var(--accent);
    background: rgba(124, 111, 239, 0.06);
  }
  #drop-zone .icon { font-size: 2.5rem; margin-bottom: 10px; }
  #drop-zone .hint { color: var(--muted); font-size: 0.88rem; margin-top: 6px; }
  #drop-zone .filename {
    margin-top: 10px;
    font-size: 0.9rem;
    color: var(--accent);
    font-weight: 500;
    word-break: break-all;
  }
  #file-input { display: none; }

  /* Options */
  .options {
    display: flex;
    gap: 16px;
    margin-top: 20px;
    flex-wrap: wrap;
  }
  .field { display: flex; flex-direction: column; gap: 5px; flex: 1; min-width: 130px; }
  .field label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .field select, .field input {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    padding: 8px 12px;
    font-size: 0.9rem;
    outline: none;
    transition: border-color 0.2s;
  }
  .field select:focus, .field input:focus { border-color: var(--accent); }
  .field input::placeholder { color: var(--muted); }

  /* Submit button */
  #submit-btn {
    margin-top: 22px;
    width: 100%;
    padding: 13px;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s, opacity 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
  }
  #submit-btn:hover:not(:disabled) { background: var(--accent-hover); }
  #submit-btn:disabled { opacity: 0.55; cursor: not-allowed; }

  /* Spinner */
  .spinner {
    width: 18px; height: 18px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Error banner */
  #error-banner {
    display: none;
    margin-top: 16px;
    padding: 12px 16px;
    background: rgba(248, 113, 113, 0.12);
    border: 1px solid rgba(248, 113, 113, 0.35);
    border-radius: 8px;
    color: var(--error);
    font-size: 0.9rem;
  }

  /* Results */
  #results {
    display: none;
    margin-top: 28px;
    width: 100%;
    max-width: 700px;
  }
  #results h2 {
    font-size: 1rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 14px;
  }
  .comparison {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }
  @media (max-width: 560px) { .comparison { grid-template-columns: 1fr; } }

  .img-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .img-panel .label {
    padding: 8px 14px;
    font-size: 0.8rem;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .img-panel .label span.tag {
    font-weight: 600;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 20px;
  }
  .img-panel .label span.tag.before { background: rgba(136,136,168,0.18); color: var(--muted); }
  .img-panel .label span.tag.after  { background: rgba(74,222,128,0.15); color: var(--success); }
  .img-panel img {
    width: 100%;
    display: block;
    object-fit: contain;
    max-height: 380px;
  }
  .img-panel .dims {
    padding: 6px 14px;
    font-size: 0.78rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
  }

  #download-btn {
    display: block;
    margin-top: 16px;
    width: 100%;
    padding: 13px;
    background: transparent;
    color: var(--accent);
    border: 1.5px solid var(--accent);
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    text-align: center;
    text-decoration: none;
    transition: background 0.2s, color 0.2s;
  }
  #download-btn:hover { background: var(--accent); color: #fff; }
</style>
</head>
<body>

<header>
  <h1>DRCT Image Upscaler</h1>
  <p>4&times; super-resolution powered by DRCT-L</p>
</header>

<div class="card">
  <div id="drop-zone">
    <div class="icon">&#128444;</div>
    <div>Drag &amp; drop an image here</div>
    <div class="hint">or click to select &mdash; JPG, PNG, BMP, TIFF, WebP</div>
    <div class="filename" id="filename-label"></div>
    <input type="file" id="file-input" accept=".jpg,.jpeg,.png,.bmp,.tiff,.tif,.webp">
  </div>

  <div class="options">
    <div class="field">
      <label for="precision-select">Precision</label>
      <select id="precision-select">
        <option value="fp32">fp32 (default)</option>
        <option value="fp16">fp16 (GPU only)</option>
      </select>
    </div>
    <div class="field">
      <label for="tile-input">Tile size <span style="font-weight:400">(optional)</span></label>
      <input type="number" id="tile-input" placeholder="e.g. 256" min="64" step="16">
    </div>
  </div>

  <button id="submit-btn" disabled>
    <div class="spinner" id="spinner"></div>
    <span id="btn-label">Upscale Image</span>
  </button>

  <div id="error-banner"></div>
</div>

<div id="results">
  <h2>Result</h2>
  <div class="comparison">
    <div class="img-panel">
      <div class="label">
        <span class="tag before">Original</span>
        <span id="orig-dims" style="font-size:0.78rem"></span>
      </div>
      <img id="orig-img" alt="Original">
      <div class="dims" id="orig-dims-bar"></div>
    </div>
    <div class="img-panel">
      <div class="label">
        <span class="tag after">Upscaled</span>
        <span id="up-dims" style="font-size:0.78rem"></span>
      </div>
      <img id="up-img" alt="Upscaled">
      <div class="dims" id="up-dims-bar"></div>
    </div>
  </div>
  <a id="download-btn" href="#" download>&#8595; Download Upscaled Image</a>
</div>

<script>
(function () {
  const dropZone    = document.getElementById("drop-zone");
  const fileInput   = document.getElementById("file-input");
  const filenameLabel = document.getElementById("filename-label");
  const submitBtn   = document.getElementById("submit-btn");
  const btnLabel    = document.getElementById("btn-label");
  const spinner     = document.getElementById("spinner");
  const errorBanner = document.getElementById("error-banner");
  const results     = document.getElementById("results");
  const origImg     = document.getElementById("orig-img");
  const upImg       = document.getElementById("up-img");
  const origDims    = document.getElementById("orig-dims");
  const upDims      = document.getElementById("up-dims");
  const downloadBtn = document.getElementById("download-btn");

  let selectedFile = null;
  let prevObjectUrl = null;

  // --- Drop zone interactions ---
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    selectedFile = file;
    filenameLabel.textContent = file.name;
    submitBtn.disabled = false;
    hideError();
    // Preview original immediately
    const reader = new FileReader();
    reader.onload = e => {
      origImg.src = e.target.result;
      origImg.onload = () => {
        origDims.textContent = origImg.naturalWidth + " × " + origImg.naturalHeight;
      };
    };
    reader.readAsDataURL(file);
  }

  // --- Submit ---
  submitBtn.addEventListener("click", async () => {
    if (!selectedFile || submitBtn.disabled) return;
    setLoading(true);
    hideError();
    results.style.display = "none";

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("precision", document.getElementById("precision-select").value);
    const tile = document.getElementById("tile-input").value;
    if (tile) formData.append("tile", tile);

    try {
      const resp = await fetch("/upscale", { method: "POST", body: formData });
      if (!resp.ok) {
        let detail = resp.statusText;
        try { detail = (await resp.json()).detail; } catch (_) {}
        throw new Error(detail);
      }

      const blob = await resp.blob();

      // Free previous object URL
      if (prevObjectUrl) URL.revokeObjectURL(prevObjectUrl);
      prevObjectUrl = URL.createObjectURL(blob);

      upImg.src = prevObjectUrl;
      upImg.onload = () => {
        upDims.textContent = upImg.naturalWidth + " × " + upImg.naturalHeight;
      };

      const stem = selectedFile.name.replace(/\.[^.]+$/, "");
      const ext  = selectedFile.name.match(/\.[^.]+$/)?.[0] ?? "";
      downloadBtn.href = prevObjectUrl;
      downloadBtn.download = stem + "_upscaled" + ext;

      results.style.display = "block";
    } catch (err) {
      showError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  });

  function setLoading(on) {
    submitBtn.disabled = on;
    spinner.style.display = on ? "block" : "none";
    btnLabel.textContent = on ? "Processing…" : "Upscale Image";
  }

  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.style.display = "block";
    setTimeout(() => { errorBanner.style.display = "none"; }, 8000);
  }

  function hideError() {
    errorBanner.style.display = "none";
  }
})();
</script>
</body>
</html>"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"[server] Loading model on {device}...")
    _state["model"] = load_model(DEFAULT_MODEL_PATH, SCALE, device)
    _state["device"] = device
    print("[server] Model ready. Open http://localhost:8000")
    yield
    _state.clear()


app = FastAPI(title="DRCT Image Upscaler", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=FRONTEND_HTML)


@app.post("/upscale")
async def upscale_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    precision: str = Form(default="fp32"),
    tile: int | None = Form(default=None),
    tile_overlap: int = Form(default=32),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")
    if precision not in ("fp32", "fp16"):
        raise HTTPException(status_code=400, detail="precision must be fp32 or fp16")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 100 MB)")

    tmp_dir = tempfile.mkdtemp(prefix="upscaler_")
    tmp_path = Path(tmp_dir)

    try:
        input_path = tmp_path / f"input{suffix}"
        output_path = tmp_path / f"output{suffix}"
        input_path.write_bytes(contents)

        # Validate image is actually readable
        check = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if check is None:
            raise HTTPException(status_code=422, detail="File is not a valid image or is corrupted")
        del check

        with _inference_lock:
            upscale_image(
                str(input_path),
                str(output_path),
                _state["model"],
                SCALE,
                precision,
                tile,
                tile_overlap,
                _state["device"],
            )

        if not output_path.exists():
            raise HTTPException(status_code=500, detail="Upscaling produced no output")

        background_tasks.add_task(shutil.rmtree, tmp_dir, True)

        stem = Path(file.filename).stem
        return FileResponse(
            path=str(output_path),
            media_type=_MEDIA_TYPES.get(suffix, "application/octet-stream"),
            filename=f"{stem}_upscaled{suffix}",
        )

    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except SystemExit as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
