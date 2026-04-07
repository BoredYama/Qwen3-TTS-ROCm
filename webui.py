#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-TTS Web UI — FastAPI backend with model management and TTS endpoints.

Usage:
    python webui.py [--host 0.0.0.0] [--port 7860]
"""

import argparse
import base64
import gc
import io
import os
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# ROCm / MIOpen performance environment variables
# MUST be set before importing torch / MIOpen
# ---------------------------------------------------------------------------
# [CRITICAL] Disable GEMM-based conv solvers.  On RDNA3 (gfx1100/1102) the
# GEMM path reports workspace_size=0 through legacy PyTorch APIs, which forces
# MIOpen to fall back to ConvDirectNaive — orders of magnitude slower.
# Disabling GEMM lets MIOpen pick Composable-Kernel / Winograd solvers instead.
os.environ.setdefault("MIOPEN_DEBUG_CONV_GEMM", "0")
# Incremental tuning: tune each new shape once, cache results in User DB
# (~/.config/miopen/).  After the first run, subsequent runs reuse the cache.
os.environ.setdefault("MIOPEN_FIND_ENFORCE", "3")
# Expandable memory segments: prevents HIP memory fragmentation so MIOpen
# solvers can actually allocate the workspace they need.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
# hipBLAS tunable operations: benchmark GEMM kernels and cache the fastest.
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "1")
os.environ.setdefault("PYTORCH_TUNABLEOP_TUNING", "1")
# Faster kernel launches on ROCm.
os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")
# Suppress noisy MIOpen workspace warnings (level 5 = errors only).
os.environ.setdefault("MIOPEN_LOG_LEVEL", "5")

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from qwen_tts import Qwen3TTSModel
from qwen_tts.device_utils import get_device

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "Qwen3-TTS-12Hz-1.7B-CustomVoice": {
        "hf_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "type": "custom_voice",
        "size": "1.7B",
    },
    "Qwen3-TTS-12Hz-0.6B-CustomVoice": {
        "hf_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "type": "custom_voice",
        "size": "0.6B",
    },
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
        "hf_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "type": "voice_design",
        "size": "1.7B",
    },
    "Qwen3-TTS-12Hz-1.7B-Base": {
        "hf_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "type": "base",
        "size": "1.7B",
    },
    "Qwen3-TTS-12Hz-0.6B-Base": {
        "hf_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "type": "base",
        "size": "0.6B",
    },
}

# ---------------------------------------------------------------------------
# ROCm detection and attention implementation selection
# ---------------------------------------------------------------------------
def _is_rocm() -> bool:
    """Check if running on AMD ROCm (HIP runtime)."""
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def _default_attn_impl() -> str:
    """Pick the best attention implementation for the current hardware.

    - NVIDIA: flash_attention_2 (fastest, well-tested)
    - AMD ROCm: sdpa (Scaled Dot Product Attention — safe on RDNA3 consumer GPUs)
    - CPU/MPS: eager
    """
    if not torch.cuda.is_available():
        return "eager"
    if _is_rocm():
        # Flash Attention 2 can segfault on RDNA3 consumer GPUs (gfx1100/1102)
        # during inference. SDPA is safe and still hardware-accelerated.
        return "sdpa"
    return "flash_attention_2"


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app = FastAPI(title="Qwen3-TTS Web UI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE = get_device()
ATTN_IMPL = _default_attn_impl()
DO_WARMUP = True  # Run a warmup pass after model load to pre-tune MIOpen kernels
current_model: Optional[Qwen3TTSModel] = None
current_model_name: Optional[str] = None
current_model_type: Optional[str] = None

# Enable MIOpen benchmark mode (finds fastest kernels, caches results)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def _unload_model():
    global current_model, current_model_name, current_model_type
    if current_model is not None:
        del current_model
        current_model = None
        current_model_name = None
        current_model_type = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_model(model_key: str):
    global current_model, current_model_name, current_model_type

    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}")

    if current_model_name == model_key:
        return  # Already loaded

    _unload_model()

    info = MODEL_REGISTRY[model_key]
    print(f"[WebUI] Loading model: {model_key} ({info['hf_id']}) on {DEVICE}")
    print(f"[WebUI] Attention implementation: {ATTN_IMPL}")
    t0 = time.time()

    attn_kwargs = {"attn_implementation": ATTN_IMPL} if ATTN_IMPL != "eager" else {}

    try:
        tts = Qwen3TTSModel.from_pretrained(
            info["hf_id"],
            device_map=DEVICE,
            dtype=torch.bfloat16,
            **attn_kwargs,
        )
    except Exception as e:
        print(f"[WebUI] Attention '{ATTN_IMPL}' failed ({e}), falling back to eager...")
        tts = Qwen3TTSModel.from_pretrained(
            info["hf_id"],
            device_map=DEVICE,
            dtype=torch.bfloat16,
        )

    current_model = tts
    current_model_name = model_key
    current_model_type = info["type"]
    elapsed = time.time() - t0
    print(f"[WebUI] Model loaded in {elapsed:.1f}s")

    # Warmup: run a short dummy generation to trigger MIOpen kernel tuning.
    # First inference is always slow as MIOpen searches for the fastest kernels.
    # Subsequent runs reuse the cached results and are much faster.
    if DO_WARMUP and _is_rocm():
        _warmup_model(model_key, info["type"])


def _wav_to_base64(wav: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _get_model_info() -> Dict[str, Any]:
    if current_model is None:
        return {"loaded": False}

    speakers = None
    languages = None
    if current_model_type == "custom_voice":
        spks = current_model.get_supported_speakers()
        speakers = sorted(spks) if spks else None
        langs = current_model.get_supported_languages()
        languages = sorted(langs) if langs else None
    elif current_model_type == "voice_design":
        langs = current_model.get_supported_languages()
        languages = sorted(langs) if langs else None
    elif current_model_type == "base":
        langs = current_model.get_supported_languages()
        languages = sorted(langs) if langs else None

    return {
        "loaded": True,
        "model_name": current_model_name,
        "model_type": current_model_type,
        "speakers": speakers,
        "languages": languages,
        "device": str(DEVICE),
        "attn_impl": ATTN_IMPL,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# Serve static files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>Missing static/index.html</h1>", status_code=500)
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/models")
async def list_models():
    models = []
    for key, info in MODEL_REGISTRY.items():
        models.append({
            "key": key,
            "hf_id": info["hf_id"],
            "type": info["type"],
            "size": info["size"],
            "loaded": key == current_model_name,
        })
    return JSONResponse({"models": models})


@app.post("/api/load-model")
async def load_model(model_key: str = Form(...)):
    try:
        _load_model(model_key)
        return JSONResponse({"status": "ok", "info": _get_model_info()})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/model-info")
async def model_info():
    return JSONResponse(_get_model_info())


@app.post("/api/custom-voice")
async def generate_custom_voice(
    text: str = Form(...),
    language: str = Form("Auto"),
    speaker: str = Form("Vivian"),
    instruct: str = Form(""),
    max_new_tokens: int = Form(2048),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.05),
):
    if current_model is None:
        return JSONResponse({"status": "error", "message": "No model loaded."}, status_code=400)
    if current_model_type != "custom_voice":
        return JSONResponse({"status": "error", "message": f"Current model is '{current_model_type}', not 'custom_voice'."}, status_code=400)

    try:
        t0 = time.time()
        wavs, sr = current_model.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker,
            instruct=instruct.strip() or None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        elapsed = time.time() - t0
        audio_b64 = _wav_to_base64(wavs[0], sr)
        return JSONResponse({
            "status": "ok",
            "audio": audio_b64,
            "sample_rate": sr,
            "duration": round(len(wavs[0]) / sr, 2),
            "elapsed": round(elapsed, 2),
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/voice-design")
async def generate_voice_design(
    text: str = Form(...),
    language: str = Form("Auto"),
    instruct: str = Form(...),
    max_new_tokens: int = Form(2048),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.05),
):
    if current_model is None:
        return JSONResponse({"status": "error", "message": "No model loaded."}, status_code=400)
    if current_model_type != "voice_design":
        return JSONResponse({"status": "error", "message": f"Current model is '{current_model_type}', not 'voice_design'."}, status_code=400)

    try:
        t0 = time.time()
        wavs, sr = current_model.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=instruct.strip(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        elapsed = time.time() - t0
        audio_b64 = _wav_to_base64(wavs[0], sr)
        return JSONResponse({
            "status": "ok",
            "audio": audio_b64,
            "sample_rate": sr,
            "duration": round(len(wavs[0]) / sr, 2),
            "elapsed": round(elapsed, 2),
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/voice-clone")
async def generate_voice_clone(
    text: str = Form(...),
    language: str = Form("Auto"),
    ref_text: str = Form(""),
    x_vector_only: bool = Form(False),
    ref_audio: UploadFile = File(...),
    max_new_tokens: int = Form(2048),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.05),
):
    if current_model is None:
        return JSONResponse({"status": "error", "message": "No model loaded."}, status_code=400)
    if current_model_type != "base":
        return JSONResponse({"status": "error", "message": f"Current model is '{current_model_type}', not 'base' (voice clone)."}, status_code=400)

    try:
        # Save uploaded audio to temp file
        audio_bytes = await ref_audio.read()
        suffix = os.path.splitext(ref_audio.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            t0 = time.time()
            wavs, sr = current_model.generate_voice_clone(
                text=text.strip(),
                language=language,
                ref_audio=tmp_path,
                ref_text=ref_text.strip() if ref_text.strip() else None,
                x_vector_only_mode=x_vector_only,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            elapsed = time.time() - t0
        finally:
            os.unlink(tmp_path)

        audio_b64 = _wav_to_base64(wavs[0], sr)
        return JSONResponse({
            "status": "ok",
            "audio": audio_b64,
            "sample_rate": sr,
            "duration": round(len(wavs[0]) / sr, 2),
            "elapsed": round(elapsed, 2),
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Warmup: pre-tune MIOpen kernels with a short dummy generation
# ---------------------------------------------------------------------------
def _warmup_model(model_key: str, model_type: str):
    """Run a tiny generation to force MIOpen kernel tuning.

    MIOpen's first-run kernel search is O(minutes) for complex models.
    After tuning, results are cached in ~/.config/miopen/ and reused.
    """
    print(f"[WebUI] Warming up MIOpen kernels (this speeds up all future generations)...")
    t0 = time.time()

    try:
        if model_type == "custom_voice":
            current_model.generate_custom_voice(
                text="Hello.",
                language="English",
                speaker="Vivian",
                max_new_tokens=50,
            )
        elif model_type == "voice_design":
            current_model.generate_voice_design(
                text="Hello.",
                language="English",
                instruct="A warm voice.",
                max_new_tokens=50,
            )
        elif model_type == "base":
            # Base model (voice clone) needs ref audio — skip warmup for it
            # as we'd need a dummy audio file
            print("[WebUI] Skipping warmup for base model (needs reference audio)")
            return

        elapsed = time.time() - t0
        print(f"[WebUI] Warmup complete in {elapsed:.1f}s — future generations will be faster")
    except Exception as e:
        print(f"[WebUI] Warmup failed (non-critical): {e}")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def main():
    global ATTN_IMPL, DO_WARMUP

    parser = argparse.ArgumentParser(description="Qwen3-TTS Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--model", default=None, help="Pre-load a model on startup, e.g. Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument(
        "--attn-impl",
        default=None,
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation (default: auto-detect — sdpa on ROCm, flash_attention_2 on NVIDIA)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip MIOpen warmup pass after model load",
    )
    args = parser.parse_args()

    if args.attn_impl:
        ATTN_IMPL = args.attn_impl
    if args.no_warmup:
        DO_WARMUP = False

    is_rocm = _is_rocm()
    print(f"[WebUI] Device: {DEVICE}")
    print(f"[WebUI] ROCm detected: {is_rocm}{'  (HIP ' + str(torch.version.hip) + ')' if is_rocm else ''}")
    print(f"[WebUI] Attention implementation: {ATTN_IMPL}")
    if is_rocm:
        print(f"[WebUI] HIP memory allocator: expandable_segments={os.environ.get('PYTORCH_HIP_ALLOC_CONF', 'default')}")
        print(f"[WebUI] Tunable ops: enabled={os.environ.get('PYTORCH_TUNABLEOP_ENABLED', '0')}")
        print(f"[WebUI] MIOpen warmup: {'enabled' if DO_WARMUP else 'disabled'}")
    print(f"[WebUI] Starting server at http://{args.host}:{args.port}")

    if args.model:
        _load_model(args.model)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
