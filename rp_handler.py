#!/usr/bin/env python3
"""
RunPod Serverless handler for ACE-Step 1.5 music generation.

Lifecycle
---------
1. Container cold-start: ``_load_models()`` runs ONCE at import time. It
   initializes the DiT pipeline (VAE + text encoder + diffusion transformer)
   and, optionally, the 5Hz language model used for Chain-of-Thought reasoning.
   Weights are pulled from HuggingFace on first run via the upstream
   ``model_downloader`` (which honours ``HF_TOKEN`` for gated repos).
2. Per-request: ``handler(job)`` reads ``job["input"]``, runs one generation,
   and returns base64-encoded WAV audio. Models are NEVER reloaded per request.

Job input schema (``job["input"]``)
------------------------------------
    prompt   (str)            caption / text description of the music. Required.
    lyrics   (str)            lyrics text. Optional. "[Instrumental]" for no vocals.
    duration (float)          target length in seconds (10-600). Optional (-1 = auto).
    steps    (int)            diffusion inference steps (turbo: ~8). Optional.
    seed     (int)            RNG seed. Optional (-1 = random).
    audio_format (str)        "wav" (default) | "mp3" | "flac".
    instrumental (bool)       force instrumental output.
    thinking (bool)           enable 5Hz-LM CoT reasoning (requires LM loaded).

Response
--------
    {
        "audio_base64": "<base64 WAV>",
        "sample_rate": 48000,
        "format": "wav",
        "seed": 1234,
        "duration_seconds": 30.0,
        "generation_time_seconds": 12.3,
        "status_message": "..."
    }

On failure the handler returns ``{"error": "..."}`` so the RunPod client gets a
structured error instead of an opaque worker crash.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import threading
import time
import traceback
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Environment hardening (must run before importing torch / acestep)
# ---------------------------------------------------------------------------
# Strip proxies that interfere with HuggingFace downloads inside the worker.
for _proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_proxy_var, None)

# torchaudio ffmpeg backend is the portable choice inside the container.
os.environ.setdefault("TORCHAUDIO_USE_BACKEND", "ffmpeg")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Non-interactive worker: suppress tqdm bars that would spam serverless logs.
os.environ.setdefault("ACESTEP_DISABLE_TQDM", "1")

# Make the repo importable regardless of the working directory RunPod uses.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# acestep's get_project_root() reads this to locate the checkpoints directory.
os.environ.setdefault("ACESTEP_PROJECT_ROOT", _PROJECT_ROOT)
# Serverless ALWAYS prefers the weights baked into the image at /app/checkpoints
# (see Dockerfile.runpod). setdefault honours an explicit override — e.g. a
# network volume — when one is supplied, but otherwise pins the in-image dir so
# get_checkpoints_dir() resolves there and never re-downloads on cold-start.
os.environ.setdefault("ACESTEP_CHECKPOINTS_DIR", "/app/checkpoints")

from loguru import logger  # noqa: E402  (after env setup, before heavy imports)

import runpod  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------
# HuggingFace repo that holds the core ACE-Step 1.5 weights. Wired into the
# upstream downloader so a fork can be pointed at a different repo.
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "ACE-Step/Ace-Step1.5").strip()

# Which DiT (diffusion) model variant to load. "acestep-v15-turbo" ships inside
# the main repo and is the fast default — ideal for serverless cold-starts.
DIT_CONFIG_PATH = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo").strip()

# 5Hz language model for Chain-of-Thought reasoning. "acestep-5Hz-lm-1.7B" is
# bundled in the main repo (no extra download) and fits comfortably on a 4090.
LM_MODEL_PATH = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B").strip()

# "pt" (PyTorch-native) is the most robust backend for a single-GPU worker.
# "vllm" is faster but adds startup complexity; opt in via env if desired.
LM_BACKEND = os.environ.get("ACESTEP_LM_BACKEND", "pt").strip()

DEVICE = os.environ.get("ACESTEP_DEVICE", "auto").strip()

# Whether to initialize the 5Hz LM at all. Disable for pure-DiT (faster, less
# VRAM, but no Chain-of-Thought / thinking mode).
INIT_LLM = os.environ.get("ACESTEP_INIT_LLM", "true").strip().lower() in ("1", "true", "yes", "auto")

# On a 24 GB RTX 4090, turbo DiT + 1.7B LM fit without offload. Override to
# "true" on smaller cards to trade speed for headroom.
OFFLOAD_TO_CPU = os.environ.get("ACESTEP_OFFLOAD_TO_CPU", "false").strip().lower() in ("1", "true", "yes")

# Preferred download source: huggingface | modelscope | auto.
DOWNLOAD_SOURCE = os.environ.get("ACESTEP_DOWNLOAD_SOURCE", "huggingface").strip().lower()

# Generation defaults / safety bounds.
DEFAULT_STEPS = int(os.environ.get("ACESTEP_DEFAULT_STEPS", "8"))
DEFAULT_DURATION = float(os.environ.get("ACESTEP_DEFAULT_DURATION", "-1"))
MAX_DURATION = float(os.environ.get("ACESTEP_MAX_DURATION", "600"))
MAX_STEPS = int(os.environ.get("ACESTEP_MAX_STEPS", "200"))

# ---------------------------------------------------------------------------
# Module-level model state (populated once by _load_models)
# ---------------------------------------------------------------------------
_dit_handler = None          # AceStepHandler (DiT + VAE + text encoder)
_llm_handler = None          # LLMHandler (5Hz LM) — may stay None if disabled/failed
_lm_available = False        # True only when the LM initialized successfully
_models_ready = False        # True after a successful cold-start load
_load_error: Optional[str] = None
# Single-GPU worker: serialize generations so concurrent jobs never race the GPU.
_gen_lock = threading.Lock()


def _apply_checkpoint_repo_override() -> None:
    """Point the upstream downloader at CHECKPOINT_PATH when it differs.

    The framework maps model *names* (e.g. "acestep-v15-turbo") to fixed HF
    repos via ``model_downloader.MAIN_MODEL_REPO``. Honouring CHECKPOINT_PATH
    lets a fork redirect the core download without editing library code.
    """
    try:
        from acestep import model_downloader

        default_repo = getattr(model_downloader, "MAIN_MODEL_REPO", "")
        if CHECKPOINT_PATH and CHECKPOINT_PATH.lower() != default_repo.lower():
            logger.info(
                "[rp_handler] Overriding main model repo: {} -> {}",
                default_repo,
                CHECKPOINT_PATH,
            )
            model_downloader.MAIN_MODEL_REPO = CHECKPOINT_PATH
    except Exception as exc:  # pragma: no cover - best-effort override
        logger.warning("[rp_handler] Could not apply CHECKPOINT_PATH override: {}", exc)


def _log_checkpoints_presence(checkpoint_dir: str) -> bool:
    """Log whether weights already exist locally and return that fact.

    A non-empty checkpoints directory means the image-baked weights are present,
    so the upstream loader will find them and skip any HuggingFace download. When
    the directory is missing/empty we only warn — the loader is still allowed to
    download as a fallback so a misbuilt image degrades instead of hard-failing.
    """
    try:
        has_weights = os.path.isdir(checkpoint_dir) and any(os.scandir(checkpoint_dir))
    except OSError:
        has_weights = False

    if has_weights:
        logger.info(
            "[rp_handler] Local checkpoints present at {} — no download needed.",
            checkpoint_dir,
        )
    else:
        logger.warning(
            "[rp_handler] No local checkpoints at {} — upstream download will run.",
            checkpoint_dir,
        )
    return has_weights


def _load_models() -> None:
    """Cold-start: load the DiT pipeline and (optionally) the 5Hz LM exactly once."""
    global _dit_handler, _llm_handler, _lm_available, _models_ready, _load_error

    if _models_ready:
        return

    t_start = time.time()
    logger.info("=" * 60)
    logger.info("[rp_handler] ACE-Step 1.5 cold-start")
    logger.info("  CHECKPOINT_PATH : {}", CHECKPOINT_PATH)
    logger.info("  DiT model       : {}", DIT_CONFIG_PATH)
    logger.info("  LM model        : {} (init={})", LM_MODEL_PATH, INIT_LLM)
    logger.info("  LM backend      : {}", LM_BACKEND)
    logger.info("  device          : {}", DEVICE)
    logger.info("  offload_to_cpu  : {}", OFFLOAD_TO_CPU)
    logger.info("=" * 60)

    try:
        _apply_checkpoint_repo_override()

        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        prefer_source = None if DOWNLOAD_SOURCE in ("", "auto") else DOWNLOAD_SOURCE

        # Resolve the checkpoints directory the SAME way the DiT pipeline does:
        # ACESTEP_CHECKPOINTS_DIR (e.g. a network volume at /runpod-volume/checkpoints)
        # wins over the in-image <project_root>/checkpoints fallback. Without this,
        # the LM below would re-download to the ephemeral layer on every cold-start
        # even when the DiT weights are served from a persistent volume.
        from acestep.model_downloader import get_checkpoints_dir

        checkpoint_dir = str(get_checkpoints_dir())
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info("[rp_handler] Checkpoints directory: {}", checkpoint_dir)
        _log_checkpoints_presence(checkpoint_dir)

        # ---- DiT pipeline (downloads weights on first run) ----
        logger.info("[rp_handler] Initializing DiT pipeline...")
        dit_handler = AceStepHandler()
        status_msg, success = dit_handler.initialize_service(
            project_root=_PROJECT_ROOT,
            config_path=DIT_CONFIG_PATH,
            device=DEVICE,
            offload_to_cpu=OFFLOAD_TO_CPU,
            prefer_source=prefer_source,
        )
        if not success:
            raise RuntimeError(f"DiT initialization failed: {status_msg}")
        _dit_handler = dit_handler
        logger.info("[rp_handler] DiT ready ({:.1f}s) — {}", time.time() - t_start, status_msg)

        # ---- 5Hz language model (optional) ----
        if INIT_LLM:
            try:
                logger.info("[rp_handler] Initializing 5Hz LM ({})...", LM_MODEL_PATH)
                llm_handler = LLMHandler()
                lm_status, lm_success = llm_handler.initialize(
                    checkpoint_dir=checkpoint_dir,
                    lm_model_path=LM_MODEL_PATH,
                    backend=LM_BACKEND,
                    device=DEVICE,
                    offload_to_cpu=OFFLOAD_TO_CPU,
                    dtype=None,
                )
                if lm_success:
                    _llm_handler = llm_handler
                    _lm_available = True
                    logger.info("[rp_handler] 5Hz LM ready — {}", lm_status)
                else:
                    logger.warning(
                        "[rp_handler] LM init failed ({}); continuing in pure-DiT mode.",
                        lm_status,
                    )
            except Exception as lm_exc:
                logger.warning(
                    "[rp_handler] LM init raised ({}); continuing in pure-DiT mode.",
                    lm_exc,
                )
        else:
            logger.info("[rp_handler] LM disabled (ACESTEP_INIT_LLM=false) — pure-DiT mode.")

        _models_ready = True
        logger.info(
            "[rp_handler] Cold-start complete in {:.1f}s (LM available: {})",
            time.time() - t_start,
            _lm_available,
        )
    except Exception as exc:
        _load_error = f"{exc}\n{traceback.format_exc()}"
        logger.error("[rp_handler] Cold-start FAILED: {}", _load_error)
        # Do not swallow: re-raise so the worker is marked unhealthy rather than
        # silently serving requests it can never fulfil.
        raise


# ---------------------------------------------------------------------------
# Input parsing & validation
# ---------------------------------------------------------------------------
def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_job_input(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the raw job input into generation parameters."""
    if not isinstance(job_input, dict):
        raise ValueError("'input' must be a JSON object")

    prompt = job_input.get("prompt") or job_input.get("caption") or ""
    prompt = str(prompt).strip()
    lyrics = str(job_input.get("lyrics") or "").strip()
    instrumental = bool(job_input.get("instrumental", False))

    # A caption or lyrics is required — generating from nothing is meaningless.
    if not prompt and not lyrics:
        raise ValueError("at least one of 'prompt' (caption) or 'lyrics' is required")

    duration = _coerce_float(job_input.get("duration"), DEFAULT_DURATION)
    if duration > 0:
        duration = max(10.0, min(duration, MAX_DURATION))

    steps = _coerce_int(job_input.get("steps"), DEFAULT_STEPS)
    steps = max(1, min(steps, MAX_STEPS))

    seed = _coerce_int(job_input.get("seed"), -1)

    audio_format = str(job_input.get("audio_format", "wav")).strip().lower()
    if audio_format not in ("wav", "mp3", "flac"):
        audio_format = "wav"

    # Thinking mode only works when the LM is loaded; auto-disable otherwise.
    thinking = bool(job_input.get("thinking", True)) and _lm_available

    return {
        "prompt": prompt,
        "lyrics": lyrics if lyrics else ("[Instrumental]" if instrumental else ""),
        "instrumental": instrumental,
        "duration": duration,
        "steps": steps,
        "seed": seed,
        "audio_format": audio_format,
        "thinking": thinking,
        "guidance_scale": _coerce_float(job_input.get("guidance_scale"), 1.0),
        "bpm": job_input.get("bpm"),
        "keyscale": str(job_input.get("keyscale") or ""),
        "vocal_language": str(job_input.get("vocal_language") or "unknown"),
    }


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------
def _tensor_to_wav_bytes(audio_tensor, sample_rate: int) -> bytes:
    """Encode a CPU float32 tensor [channels, samples] to in-memory WAV (PCM_16)."""
    import numpy as np
    import soundfile as sf

    array = audio_tensor.detach().cpu().to(dtype=__import__("torch").float32).numpy()
    # soundfile expects [samples, channels]; pipeline yields [channels, samples].
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim == 2:
        array = array.T
    array = np.clip(array, -1.0, 1.0)

    buffer = io.BytesIO()
    sf.write(buffer, array, samplerate=int(sample_rate), format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _encode_audio_dict(audio_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Turn one pipeline audio dict into a base64 WAV payload."""
    tensor = audio_dict.get("tensor")
    sample_rate = int(audio_dict.get("sample_rate") or 48000)
    if tensor is None:
        raise RuntimeError("generation produced no audio tensor")

    wav_bytes = _tensor_to_wav_bytes(tensor, sample_rate)
    num_samples = tensor.shape[-1]
    return {
        "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
        "sample_rate": sample_rate,
        "format": "wav",
        "duration_seconds": round(num_samples / float(sample_rate), 2),
        "key": audio_dict.get("key", ""),
    }


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless entrypoint: one job -> one generated track (base64 WAV)."""
    job_id = job.get("id", "unknown")

    # Guard: if cold-start failed, surface a clear error instead of crashing.
    if not _models_ready:
        return {"error": f"models not initialized: {_load_error or 'unknown error'}"}

    try:
        params_in = _parse_job_input(job.get("input") or {})
    except ValueError as exc:
        logger.warning("[rp_handler] job {} bad input: {}", job_id, exc)
        return {"error": f"invalid input: {exc}"}

    logger.info(
        "[rp_handler] job {} | steps={} duration={} seed={} thinking={} fmt={}",
        job_id,
        params_in["steps"],
        params_in["duration"],
        params_in["seed"],
        params_in["thinking"],
        params_in["audio_format"],
    )

    try:
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        gen_params = GenerationParams(
            task_type="text2music",
            caption=params_in["prompt"],
            lyrics=params_in["lyrics"],
            instrumental=params_in["instrumental"],
            duration=params_in["duration"],
            inference_steps=params_in["steps"],
            guidance_scale=params_in["guidance_scale"],
            seed=params_in["seed"],
            thinking=params_in["thinking"],
            bpm=params_in["bpm"],
            keyscale=params_in["keyscale"],
            vocal_language=params_in["vocal_language"],
        )
        gen_config = GenerationConfig(
            batch_size=1,
            audio_format="wav",
            use_random_seed=(params_in["seed"] < 0),
            seeds=None if params_in["seed"] < 0 else [params_in["seed"]],
        )

        t0 = time.time()
        # Serialize: a single GPU cannot run two diffusion jobs concurrently.
        with _gen_lock:
            result = generate_music(
                _dit_handler,
                _llm_handler,
                params=gen_params,
                config=gen_config,
                save_dir=None,  # in-memory only; we encode the tensor ourselves
            )
        elapsed = time.time() - t0

        if not result.success or not result.audios:
            msg = result.error or result.status_message or "generation returned no audio"
            logger.error("[rp_handler] job {} failed: {}", job_id, msg)
            return {"error": f"generation failed: {msg}"}

        payload = _encode_audio_dict(result.audios[0])
        seed_used = result.audios[0].get("params", {}).get("seed", params_in["seed"])
        payload.update(
            {
                "seed": seed_used,
                "generation_time_seconds": round(elapsed, 2),
                "status_message": result.status_message,
            }
        )
        logger.info(
            "[rp_handler] job {} done in {:.1f}s ({}s audio)",
            job_id,
            elapsed,
            payload["duration_seconds"],
        )
        return payload

    except Exception as exc:
        err = f"{exc}"
        logger.error("[rp_handler] job {} crashed: {}\n{}", job_id, err, traceback.format_exc())
        return {"error": f"internal error: {err}"}


# ---------------------------------------------------------------------------
# Cold-start at import: load models ONCE before the worker starts polling.
# Set RP_HANDLER_SKIP_WARMUP=1 to import the module without loading models
# (used by CI / import-smoke-tests; harmless when unset in production).
# ---------------------------------------------------------------------------
if os.environ.get("RP_HANDLER_SKIP_WARMUP", "").strip().lower() not in ("1", "true", "yes"):
    _load_models()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
