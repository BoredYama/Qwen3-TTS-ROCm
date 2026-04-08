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
# ROCm / MIOpen performance & stability environment variables
# MUST be set before importing torch / MIOpen
# ---------------------------------------------------------------------------
# Force RDNA3 detection for RX 7600 XT if not natively supported
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

# [CRITICAL] Disable GEMM-based conv solvers. On RDNA3 (gfx1100/1102) the
# GEMM path reports workspace_size=0 through legacy PyTorch APIs.
os.environ.setdefault("MIOPEN_DEBUG_CONV_GEMM", "0")

# MIOpen tuning and find mode
# Set to 'NORMAL' for better stability during kernel selection.
os.environ.setdefault("MIOPEN_FIND_MODE", "NORMAL")
# Incremental tuning: tune each new shape once, cache results in User DB.
# Set to 1 (Standard) for higher stability, or 3 for incremental caching.
os.environ.setdefault("MIOPEN_FIND_ENFORCE", "1")

# Expandable memory segments: prevents HIP memory fragmentation.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

# [STABILITY] Disable TunableOp as it can cause non-deterministic crashes on RDNA3.
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "0")
os.environ.setdefault("PYTORCH_TUNABLEOP_TUNING", "0")

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
# Monkey-patching for ROCm Stability
# ---------------------------------------------------------------------------
_original_multinomial = torch.multinomial

def _safe_multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
    """Safety wrapper for torch.multinomial on ROCm to prevent 0x1016 crashes."""
    if torch.isnan(input).any() or torch.isinf(input).any():
        print("[ROCm-Safety] WARNING: NaN or Inf detected in multinomial input! Cleaning...")
        # Replace NaNs with 0 and Infs with large values to prevent crash
        input = torch.nan_to_num(input, nan=0.0, posinf=1e38, neginf=0.0)
        # Ensure it's non-negative (required by multinomial)
        input = torch.clamp(input, min=1e-10)
    
    try:
        return _original_multinomial(input, num_samples, replacement, generator=generator, out=out)
    except Exception as e:
        if "unspecified launch failure" in str(e) or "hardware exception" in str(e).lower():
            print("[ROCm-Safety] CRITICAL: Multinomial kernel crashed. Falling back to CPU for this ops...")
            # Fallback to CPU to avoid black screen
            return _original_multinomial(input.cpu(), num_samples, replacement, generator=generator).to(input.device)
        raise e

# Apply the patch
torch.multinomial = _safe_multinomial

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
    """Pick the best attention implementation for the current hardware."""
    if not torch.cuda.is_available():
        return "eager"
    if _is_rocm():
        # User reported 'sdpa' worked for them before, so we'll stick to it.
        return "sdpa"
    return "flash_attention_2"


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app = FastAPI(title="Qwen3-TTS Web UI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# These will be initialized in main() based on args
DEVICE = torch.device("cpu")
ATTN_IMPL = "eager"
DO_WARMUP = False 

current_model: Optional[Qwen3TTSModel] = None
current_model_name: Optional[str] = None
current_model_type: Optional[str] = None

# Performance tweaks
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


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
    global current_model, current_model_name, current_model_type, DEVICE, ATTN_IMPL

    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}")

    if current_model_name == model_key:
        return 

    _unload_model()

    info = MODEL_REGISTRY[model_key]
    print(f"[WebUI] Loading model: {model_key} ({info['hf_id']}) on {DEVICE}")
    print(f"[WebUI] Attention implementation: {ATTN_IMPL}")
    t0 = time.time()

    attn_kwargs = {"attn_implementation": ATTN_IMPL} if ATTN_IMPL != "eager" else {}

    try:
        # Move model to selected device
        tts = Qwen3TTSModel.from_pretrained(
            info["hf_id"],
            device_map=str(DEVICE),
            dtype=torch.bfloat16 if str(DEVICE) != "cpu" else torch.float32,
            **attn_kwargs,
        )
    except Exception as e:
        print(f"[WebUI] Attention '{ATTN_IMPL}' failed ({e}), falling back to eager...")
        tts = Qwen3TTSModel.from_pretrained(
            info["hf_id"],
            device_map=str(DEVICE),
            dtype=torch.bfloat16 if str(DEVICE) != "cpu" else torch.float32,
        )

    current_model = tts
    current_model_name = model_key
    current_model_type = info["type"]
    elapsed = time.time() - t0
    print(f"[WebUI] Model loaded in {elapsed:.1f}s")


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
        # Check if it's a ROCm hardware exception that needs a restart
        msg = str(e)
        if "unspecified launch failure" in msg or "0x1016" in msg:
            msg = "ROCm Hardware Exception detected. Please reboot or restart the server with AMD_SERIALIZE_KERNEL=3."
        return JSONResponse({"status": "error", "message": msg}, status_code=500)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def main():
    global ATTN_IMPL, DO_WARMUP, DEVICE

    parser = argparse.ArgumentParser(description="Qwen3-TTS Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--model", default=None, help="Pre-load a model on startup")
    parser.add_argument(
        "--device", 
        default="auto", 
        help="Device to use (auto, cpu, cuda:0). Default: auto (detects ROCm/CUDA)"
    )
    parser.add_argument(
        "--attn-impl",
        default=None,
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation",
    )
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        DEVICE = get_device()
    else:
        DEVICE = torch.device(args.device)

    # Attention selection
    if args.attn_impl:
        ATTN_IMPL = args.attn_impl
    else:
        # Default based on device
        if str(DEVICE) == "cpu":
            ATTN_IMPL = "eager"
        else:
            ATTN_IMPL = _default_attn_impl()

    is_rocm = _is_rocm()
    print(f"[WebUI] Selection - Device: {DEVICE}")
    print(f"[WebUI] Selection - Attention: {ATTN_IMPL}")
    
    if is_rocm and str(DEVICE) != "cpu":
        print("[WebUI] ROCm-Safety: torch.multinomial monkey-patch applied.")
        print(f"[WebUI] HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'default')}")
        print(f"[WebUI] TunableOps: disabled (for stability)")

    print(f"[WebUI] Starting server at http://{args.host}:{args.port}")

    if args.model:
        _load_model(args.model)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
