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
The worker accepts the **full ACE-Step generation contract** — the same field
names and aliases as the FastAPI ``/release_task`` endpoint. Parsing/mapping is
delegated to the canonical helpers (``RequestParser`` ->
``build_generate_music_request`` -> ``build_generation_setup``) so the serverless
and HTTP surfaces never drift. Highlights (see ``release_task_models.py`` for the
complete list and ``release_task_param_parser.PARAM_ALIASES`` for every alias):

    prompt / caption (str)    music description. Required unless sample_mode.
    lyrics   (str)            lyrics; "[Instrumental]" for no vocals.
    instrumental (bool)       force instrumental output.
    audio_duration (float)    target seconds (10-600); aliases duration/target_duration. -1 = auto.
    audio_format (str)        mp3 (default) | wav | wav32 | flac | opus. (aac -> mp3 in-worker.)
    batch_size (int)          1..8 tracks; extra tracks returned under "audios".
    seed / use_random_seed    reproducibility controls.
    thinking (bool)           5Hz-LM CoT (auto-disabled if the LM did not load).
    sample_mode / sample_query / use_format   LM caption/lyrics authoring.
    bpm / key_scale / time_signature / vocal_language   musical metadata.
    inference_steps / guidance_scale / shift / infer_method / timesteps / use_adg / cfg_interval_*
    lm_temperature / lm_cfg_scale / lm_top_k / lm_top_p / lm_repetition_penalty / lm_negative_prompt
    use_cot_caption / use_cot_language / constrained_decoding / allow_lm_batch
    task_type / reference_audio_path / src_audio_path / instruction / repainting_* / audio_cover_strength

Cold-start-only fields (``model``, ``lm_model_path``, ``lm_backend``) cannot be
switched per request — this worker serves the DiT/LM baked at build time — so a
value that differs from the loaded model is ignored with a warning.

Response
--------
    {
        "audio_base64": "<base64 audio>",
        "sample_rate": 48000,
        "format": "mp3",
        "seed": 1234,
        "duration_seconds": 30.0,
        "generation_time_seconds": 12.3,
        "status_message": "...",
        # Song context — LM-generated in thinking mode, else the resolved/user
        # values that were actually used for generation.
        "prompt": "A soft romantic pop song with gentle piano arpeggios...",
        "lyrics": "[Verse 1]\n...\n[Chorus]\n...",
        "bpm": 90,
        "key_scale": "G Major",
        "time_signature": "4",
        "vocal_language": "en",
        "audio_duration": 180,
        # Only present when batch_size > 1: every generated track (the top-level
        # audio_base64/seed mirror tracks[0] for single-track back-compat).
        "audios": [{"audio_base64": "...", "seed": 1234, ...}, ...]
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

# Strict LM mode. When the LM is requested (INIT_LLM=true) but cannot be made
# available — not baked into the image AND not downloadable — the worker would
# otherwise silently fall back to pure-DiT, returning empty thinking-mode
# metadata (lyrics/bpm/key) with no obvious cause. With REQUIRE_LLM=true the
# cold-start instead FAILS LOUDLY so a model/endpoint misconfiguration surfaces
# immediately (worker marked unhealthy) instead of shipping degraded output.
# Default true: if you explicitly asked for the LM, a missing LM is an error.
REQUIRE_LLM = os.environ.get("ACESTEP_REQUIRE_LLM", "true").strip().lower() in ("1", "true", "yes")

# On a 24 GB RTX 4090, turbo DiT + 1.7B LM fit without offload. Override to
# "true" on smaller cards to trade speed for headroom.
OFFLOAD_TO_CPU = os.environ.get("ACESTEP_OFFLOAD_TO_CPU", "false").strip().lower() in ("1", "true", "yes")

# Preferred download source: huggingface | modelscope | auto.
DOWNLOAD_SOURCE = os.environ.get("ACESTEP_DOWNLOAD_SOURCE", "huggingface").strip().lower()

# Generation defaults / safety bounds.
# Default steps suit the BASE DiT (acestep-v15-base), which needs many more
# steps than the distilled turbo model (~8) for good audio quality. Override
# with ACESTEP_DEFAULT_STEPS=8 on the endpoint if you switch to a turbo DiT.
DEFAULT_STEPS = int(os.environ.get("ACESTEP_DEFAULT_STEPS", "60"))
MAX_DURATION = float(os.environ.get("ACESTEP_MAX_DURATION", "600"))
MAX_STEPS = int(os.environ.get("ACESTEP_MAX_STEPS", "200"))
MAX_BATCH_SIZE = int(os.environ.get("ACESTEP_MAX_BATCH_SIZE", "8"))

# 5Hz-LM sampling defaults, mirroring the FastAPI server (acestep/api_server.py)
# so the two surfaces share one contract. Used when a request omits the field.
LM_DEFAULT_TEMPERATURE = 0.85
LM_DEFAULT_CFG_SCALE = 2.5
LM_DEFAULT_TOP_P = 0.9

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


def _ensure_lm_present(checkpoint_dir: str) -> tuple[bool, str]:
    """Make sure the requested 5Hz LM exists locally, downloading it if missing.

    The LM loader (``llm_inference.initialize``) does NOT download — it only
    checks ``<checkpoint_dir>/<LM_MODEL_PATH>`` and fails if absent. The image is
    expected to bake the configured LM (see Dockerfile.runpod ``LM_MODEL``), so
    in steady state this is a no-op. But if the endpoint's
    ``ACESTEP_LM_MODEL_PATH`` was changed WITHOUT a matching image rebuild, the LM
    would be missing and thinking mode would silently die. To stay
    fail-operational — mirroring the DiT loader's auto-download — we download the
    LM once here when it is missing. Returns ``(present, detail)``.
    """
    lm_path = os.path.join(checkpoint_dir, LM_MODEL_PATH)
    if os.path.isdir(lm_path) and os.listdir(lm_path):
        return True, f"present at {lm_path}"

    logger.warning(
        "[rp_handler] LM '{}' not found at {} — it was not baked into the image. "
        "Attempting one-time download (rebuild the image with LM_MODEL={} to avoid this).",
        LM_MODEL_PATH, lm_path, LM_MODEL_PATH,
    )
    try:
        from acestep.model_downloader import SUBMODEL_REGISTRY, download_submodel

        if LM_MODEL_PATH not in SUBMODEL_REGISTRY:
            return False, (
                f"LM '{LM_MODEL_PATH}' is absent locally and not in SUBMODEL_REGISTRY, "
                f"so it cannot be auto-downloaded — bake it into the image."
            )
        prefer_source = None if DOWNLOAD_SOURCE in ("", "auto") else DOWNLOAD_SOURCE
        ok, msg = download_submodel(
            LM_MODEL_PATH,
            checkpoints_dir=checkpoint_dir,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
            prefer_source=prefer_source,
        )
        return ok, msg
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        return False, f"LM download raised: {exc}"


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

        # ---- 5Hz language model (thinking/CoT) ----
        # Fail-operational: ensure the LM is present (download once if a config
        # change wasn't matched by an image rebuild), then init. If it still
        # can't load, either hard-fail (REQUIRE_LLM=true) so the misconfig is
        # obvious, or degrade to pure-DiT with a loud, actionable warning.
        if INIT_LLM:
            lm_failure: Optional[str] = None
            try:
                present, detail = _ensure_lm_present(checkpoint_dir)
                if not present:
                    lm_failure = f"LM unavailable: {detail}"
                else:
                    logger.info("[rp_handler] LM '{}' {} — initializing...", LM_MODEL_PATH, detail)
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
                        lm_failure = f"LM init failed: {lm_status}"
            except Exception as lm_exc:
                lm_failure = f"LM init raised: {lm_exc}"

            if lm_failure and not _lm_available:
                if REQUIRE_LLM:
                    # Thinking mode is required but unavailable — refuse to start
                    # in degraded mode so the bad config surfaces immediately.
                    raise RuntimeError(
                        f"{lm_failure}. ACESTEP_INIT_LLM=true and ACESTEP_REQUIRE_LLM=true, "
                        f"so refusing to serve in degraded pure-DiT mode. Bake "
                        f"'{LM_MODEL_PATH}' into the image (Dockerfile.runpod LM_MODEL) or "
                        f"set ACESTEP_REQUIRE_LLM=false to allow a pure-DiT fallback."
                    )
                logger.warning(
                    "[rp_handler] {} — continuing in pure-DiT mode (thinking disabled, "
                    "lyrics/bpm/key will not be generated). Set ACESTEP_REQUIRE_LLM=true "
                    "to fail fast on this instead.",
                    lm_failure,
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
def _resolve_audio_format(req_format: Optional[str]) -> str:
    """Pick the in-worker encoder format for a requested ``audio_format``.

    libsndfile (used for in-memory encoding here) has no AAC encoder, so an
    ``aac`` request is honoured as MP3 — the closest lossy format the worker can
    actually produce. Anything unrecognised also degrades to MP3, which keeps the
    inline base64 small enough for RunPod's /job-done response-size limit.
    """
    fmt = str(req_format or "mp3").strip().lower()
    if fmt == "aac":
        logger.warning("[rp_handler] 'aac' has no in-worker encoder; serving 'mp3' instead.")
        return "mp3"
    if fmt not in _AUDIO_FORMAT_MAP:
        if fmt:
            logger.warning("[rp_handler] unknown audio_format '{}'; serving 'mp3'.", fmt)
        return "mp3"
    return fmt


def _warn_cold_start_only_overrides(req: Any) -> None:
    """Warn when a request asks for a model/LM the baked worker can't switch to.

    The DiT and 5Hz-LM are loaded once at cold-start (Dockerfile.runpod bakes
    them). ``model`` / ``lm_model_path`` / ``lm_backend`` therefore cannot be
    honoured per request — surface the mismatch loudly instead of silently
    generating with a different model than the caller asked for.
    """
    model = (getattr(req, "model", None) or "").strip()
    if model and model != DIT_CONFIG_PATH:
        logger.warning(
            "[rp_handler] request model='{}' ignored — this worker serves the baked DiT '{}'. "
            "Deploy a separate endpoint to use a different model.",
            model, DIT_CONFIG_PATH,
        )
    lm_path = (getattr(req, "lm_model_path", None) or "").strip()
    if lm_path and lm_path != LM_MODEL_PATH:
        logger.warning(
            "[rp_handler] request lm_model_path='{}' ignored — baked LM is '{}'.",
            lm_path, LM_MODEL_PATH,
        )
    lm_backend = (getattr(req, "lm_backend", None) or "").strip()
    if lm_backend and lm_backend != LM_BACKEND:
        logger.warning(
            "[rp_handler] request lm_backend='{}' ignored — worker LM backend is '{}'.",
            lm_backend, LM_BACKEND,
        )


def _build_request(job_input: Dict[str, Any]) -> Any:
    """Parse raw job input into a validated ``GenerateMusicRequest``.

    Reuses the canonical alias map + request builder shared with the FastAPI
    server so both surfaces accept exactly the same contract. Applies the
    worker's safety envelope (duration/steps/batch clamps) and serverless-aware
    defaults (env-tuned steps, LM-gated thinking) on top.
    """
    if not isinstance(job_input, dict):
        raise ValueError("'input' must be a JSON object")

    from acestep.api.http.release_task_models import GenerateMusicRequest
    from acestep.api.http.release_task_param_parser import RequestParser
    from acestep.api.http.release_task_request_builder import build_generate_music_request
    from acestep.constants import DEFAULT_DIT_INSTRUCTION

    parser = RequestParser(job_input)

    prompt = (parser.str("prompt") or "").strip()
    lyrics = (parser.str("lyrics") or "").strip()
    sample_mode = parser.bool("sample_mode", False)
    sample_query = (parser.str("sample_query") or "").strip()

    # Need *some* intent: a caption, lyrics, or an LM authoring request.
    if not (prompt or lyrics or sample_mode or sample_query):
        raise ValueError(
            "provide at least one of 'prompt' (caption), 'lyrics', or 'sample_query'/'sample_mode'"
        )

    # The documented `instrumental` flag is not part of the FastAPI model (which
    # infers it from lyrics) — honour it by forcing the instrumental sentinel.
    if parser.bool("instrumental", False) and not lyrics:
        lyrics = "[Instrumental]"

    overrides: Dict[str, Any] = {
        # thinking defaults ON (per schema) but only when the LM actually loaded.
        "thinking": parser.bool("thinking", True) and _lm_available,
        "lyrics": lyrics,
    }

    # When the caller omits vocal_language, default to "unknown" so the 5Hz LM
    # auto-detects it (the shared builder would otherwise assume "en").
    if parser.get("vocal_language") is None:
        overrides["vocal_language"] = "unknown"

    # Step count resolution, in priority order:
    #   1. canonical `inference_steps` (handled by the shared builder),
    #   2. the legacy `steps` key the old rp_handler accepted (back-compat),
    #   3. the worker's env-tuned DEFAULT_STEPS (base-DiT friendly) — the shared
    #      builder would otherwise default to the turbo value (8).
    if parser.get("inference_steps") is None:
        legacy_steps = job_input.get("steps")
        if legacy_steps is not None:
            try:
                overrides["inference_steps"] = int(float(legacy_steps))
            except (TypeError, ValueError):
                overrides["inference_steps"] = DEFAULT_STEPS
        else:
            overrides["inference_steps"] = DEFAULT_STEPS

    # Back-compat: an explicit non-negative seed implies reproducibility even if
    # use_random_seed was not sent (the old handler keyed off seed>=0 alone).
    if parser.get("use_random_seed") is None and parser.get("seed") is not None:
        try:
            if int(float(parser.get("seed"))) >= 0:
                overrides["use_random_seed"] = False
        except (TypeError, ValueError):
            pass

    req = build_generate_music_request(
        parser,
        GenerateMusicRequest,
        default_dit_instruction=DEFAULT_DIT_INSTRUCTION,
        lm_default_temperature=LM_DEFAULT_TEMPERATURE,
        lm_default_cfg_scale=LM_DEFAULT_CFG_SCALE,
        lm_default_top_p=LM_DEFAULT_TOP_P,
        **overrides,
    )

    # Clamp to the worker's safety envelope (all env-tunable).
    if req.audio_duration is not None and float(req.audio_duration) > 0:
        req.audio_duration = max(10.0, min(float(req.audio_duration), MAX_DURATION))
    req.inference_steps = max(1, min(int(req.inference_steps), MAX_STEPS))
    if req.batch_size is not None:
        req.batch_size = max(1, min(int(req.batch_size), MAX_BATCH_SIZE))

    _warn_cold_start_only_overrides(req)
    return req


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------
# Requested audio_format -> (soundfile format, subtype). MP3 needs libsndfile
# >= 1.1; FLAC and WAV are available on every libsndfile build, so they make
# safe fallbacks if the container's encoder can't honor the request.
_AUDIO_FORMAT_MAP = {
    "wav": ("WAV", "PCM_16"),
    "wav32": ("WAV", "FLOAT"),
    "flac": ("FLAC", None),
    "mp3": ("MP3", None),
    "opus": ("OGG", "OPUS"),
}


def _tensor_to_audio_bytes(audio_tensor, sample_rate: int, audio_format: str):
    """Encode a CPU float32 tensor [channels, samples] to in-memory audio bytes.

    Returns (bytes, actual_format). Tries the requested format first, then
    degrades to FLAC and finally WAV so a job never fails purely because the
    container's libsndfile lacks a particular encoder.
    """
    import numpy as np
    import soundfile as sf

    array = audio_tensor.detach().cpu().to(dtype=__import__("torch").float32).numpy()
    # soundfile expects [samples, channels]; pipeline yields [channels, samples].
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim == 2:
        array = array.T
    array = np.clip(array, -1.0, 1.0)

    # De-duplicate the fallback chain while preserving order.
    candidates = list(dict.fromkeys([audio_format, "flac", "wav"]))
    for fmt in candidates:
        sf_format, sf_subtype = _AUDIO_FORMAT_MAP.get(fmt, ("WAV", "PCM_16"))
        try:
            buffer = io.BytesIO()
            sf.write(
                buffer,
                array,
                samplerate=int(sample_rate),
                format=sf_format,
                subtype=sf_subtype,
            )
            return buffer.getvalue(), fmt
        except Exception as exc:
            logger.warning("[encode] {} encoding failed ({}); trying fallback.", fmt, exc)
    raise RuntimeError("no available audio encoder (wav/flac/mp3 all failed)")


def _encode_audio_dict(audio_dict: Dict[str, Any], audio_format: str = "mp3") -> Dict[str, Any]:
    """Turn one pipeline audio dict into a base64 audio payload.

    Defaults to a compressed format so the inline base64 stays under RunPod's
    /job-done response-size limit; raw WAV of a long clip exceeds it and the
    platform rejects the result with HTTP 400.
    """
    tensor = audio_dict.get("tensor")
    sample_rate = int(audio_dict.get("sample_rate") or 48000)
    if tensor is None:
        raise RuntimeError("generation produced no audio tensor")

    audio_bytes, actual_format = _tensor_to_audio_bytes(tensor, sample_rate, audio_format)
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    num_samples = tensor.shape[-1]

    # Log payload size so any future oversize 400 is trivial to diagnose.
    logger.info(
        "[encode] format={} bytes={} base64_len={} (~{:.2f} MB)",
        actual_format,
        len(audio_bytes),
        len(audio_b64),
        len(audio_b64) / 1_048_576,
    )
    return {
        "audio_base64": audio_b64,
        "sample_rate": sample_rate,
        "format": actual_format,
        "duration_seconds": round(num_samples / float(sample_rate), 2),
    }


# ---------------------------------------------------------------------------
# Generation metadata extraction
# ---------------------------------------------------------------------------
# Values the caller cares about (lyrics, bpm, key, etc.) can originate from
# three places, in decreasing priority:
#   1. The 5Hz LM (thinking mode) — it *generates* lyrics/bpm/key/time-sig from
#      the prompt. Surfaced on result.extra_outputs["lm_metadata"].
#   2. The per-audio params dict — the values actually fed to the DiT pipeline
#      (already merged with any LM/CoT output).
#   3. The user's raw request — what they explicitly passed in.
# "Empty" sentinels we skip so a real value further down the chain wins.
_EMPTY_META_VALUES = (None, "", "N/A", "n/a", "unknown", -1, "-1")


def _first_meta_value(sources: tuple, *keys: str) -> Any:
    """Return the first meaningful value found across sources for any of keys."""
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if isinstance(value, str):
                value = value.strip()
            if value not in _EMPTY_META_VALUES:
                return value
    return None


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_number(value: Any):
    """Coerce to int when whole (e.g. 180), else a 2-dp float; None on failure."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else round(number, 2)


def _build_generation_metadata(result, audio_dict: Dict[str, Any], params_in: Dict[str, Any]) -> Dict[str, Any]:
    """Collect the prompt/lyrics/musical metadata to echo back in the response.

    Prefers LM-generated values (thinking mode) over the resolved DiT params
    over the raw request, so the client sees what was *actually* used/created.
    """
    lm_meta = (getattr(result, "extra_outputs", None) or {}).get("lm_metadata") or {}
    audio_params = audio_dict.get("params") or {}
    sources = (lm_meta, audio_params, params_in)

    return {
        "prompt": _first_meta_value(sources, "caption", "prompt") or params_in.get("prompt", ""),
        "lyrics": _first_meta_value(sources, "lyrics") or params_in.get("lyrics", ""),
        "bpm": _as_int(_first_meta_value(sources, "bpm")),
        "key_scale": _first_meta_value(sources, "keyscale", "key_scale") or "",
        "time_signature": str(_first_meta_value(sources, "timesignature", "time_signature") or ""),
        "vocal_language": _first_meta_value(sources, "vocal_language") or "unknown",
        "audio_duration": _as_number(_first_meta_value(sources, "duration", "audio_duration")),
    }


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def _build_generation_setup(req: Any):
    """Resolve a request into ``(GenerationParams, GenerationConfig)``.

    Mirrors the FastAPI ``run_blocking_generate`` flow using the shared helpers:
    an LLM input pre-pass (sample_mode/use_format/CoT metadata) followed by the
    canonical ``build_generation_setup`` mapping — so serverless and HTTP produce
    identical params for the same request. The LM is already loaded at cold-start,
    so the readiness callback is a no-op and a lightweight app-state shim carries
    the LM-availability flag the helper expects.
    """
    from types import SimpleNamespace

    from acestep.api.job_generation_setup import build_generation_setup
    from acestep.api.llm_generation_inputs import prepare_llm_generation_inputs
    from acestep.api.server_utils import (
        is_instrumental,
        parse_description_hints,
        parse_timesteps,
    )
    from acestep.constants import DEFAULT_DIT_INSTRUCTION, TASK_INSTRUCTIONS
    from acestep.inference import create_sample, format_sample

    # The helper reads _llm_init_error to decide LM availability: when the LM
    # did not load, a non-None error makes it auto-disable CoT (use_cot_*) and
    # hard-fail requests that genuinely need the LM (thinking/sample_mode/format)
    # — exactly the FastAPI server's contract. None means "LM ready".
    app_state = SimpleNamespace(
        _llm_initialized=_lm_available,
        _llm_init_error=None if _lm_available else "5Hz LM not loaded on this worker",
    )

    prepared = prepare_llm_generation_inputs(
        app_state=app_state,
        llm_handler=_llm_handler,
        req=req,
        selected_handler_device=getattr(_dit_handler, "device", "cuda"),
        parse_description_hints=parse_description_hints,
        create_sample_fn=create_sample,
        format_sample_fn=format_sample,
        ensure_llm_ready_fn=lambda: None,  # LM is loaded once at cold-start.
        log_fn=lambda message: logger.info("{}", message),
    )

    setup = build_generation_setup(
        req=req,
        caption=prepared.caption,
        global_caption=prepared.global_caption,
        lyrics=prepared.lyrics,
        bpm=prepared.bpm,
        key_scale=prepared.key_scale,
        time_signature=prepared.time_signature,
        audio_duration=prepared.audio_duration,
        thinking=prepared.thinking,
        sample_mode=prepared.sample_mode,
        format_has_duration=prepared.format_has_duration,
        use_cot_caption=prepared.use_cot_caption,
        use_cot_language=prepared.use_cot_language,
        lm_top_k=prepared.lm_top_k,
        lm_top_p=prepared.lm_top_p,
        parse_timesteps=parse_timesteps,
        is_instrumental=is_instrumental,
        default_dit_instruction=DEFAULT_DIT_INSTRUCTION,
        task_instructions=TASK_INSTRUCTIONS,
    )
    return setup.params, setup.config, prepared


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless entrypoint: one job -> one or more generated tracks."""
    job_id = job.get("id", "unknown")

    # Guard: if cold-start failed, surface a clear error instead of crashing.
    if not _models_ready:
        return {"error": f"models not initialized: {_load_error or 'unknown error'}"}

    try:
        req = _build_request(job.get("input") or {})
    except ValueError as exc:
        logger.warning("[rp_handler] job {} bad input: {}", job_id, exc)
        return {"error": f"invalid input: {exc}"}

    audio_format = _resolve_audio_format(req.audio_format)
    logger.info(
        "[rp_handler] job {} | task={} steps={} duration={} batch={} thinking={} fmt={}",
        job_id,
        req.task_type,
        req.inference_steps,
        req.audio_duration,
        req.batch_size,
        req.thinking,
        audio_format,
    )

    try:
        from acestep.inference import generate_music

        params, config, prepared = _build_generation_setup(req)
        # We encode the returned tensor ourselves (save_dir=None); pin the
        # pipeline's own format to WAV so it never attempts a format the
        # container's encoder may lack.
        config.audio_format = "wav"

        t0 = time.time()
        # Serialize: a single GPU cannot run two diffusion jobs concurrently.
        with _gen_lock:
            result = generate_music(
                _dit_handler,
                _llm_handler,
                params=params,
                config=config,
                save_dir=None,  # in-memory only; we encode the tensor ourselves
            )
        elapsed = time.time() - t0

        if not result.success or not result.audios:
            msg = result.error or result.status_message or "generation returned no audio"
            logger.error("[rp_handler] job {} failed: {}", job_id, msg)
            return {"error": f"generation failed: {msg}"}

        # Encode every track (batch_size may be > 1). The top-level fields mirror
        # the first track so single-track clients keep working unchanged.
        tracks = []
        for audio_out in result.audios:
            track = _encode_audio_dict(audio_out, audio_format)
            track["seed"] = audio_out.get("params", {}).get("seed")
            tracks.append(track)

        # Metadata-echo fallback source: what the LM/DiT actually resolved.
        params_in = {
            "prompt": prepared.caption or req.prompt or "",
            "lyrics": prepared.lyrics or req.lyrics or "",
            "bpm": prepared.bpm,
            "key_scale": prepared.key_scale or "",
            "time_signature": prepared.time_signature or "",
            "vocal_language": req.vocal_language or "unknown",
            "audio_duration": prepared.audio_duration,
        }

        payload = dict(tracks[0])
        payload.update(
            {
                "generation_time_seconds": round(elapsed, 2),
                "status_message": result.status_message,
            }
        )
        # Echo back the prompt/lyrics/musical metadata (LM-generated in thinking
        # mode) so the client gets the full song context alongside the audio.
        payload.update(_build_generation_metadata(result, result.audios[0], params_in))
        if len(tracks) > 1:
            payload["audios"] = tracks
        logger.info(
            "[rp_handler] job {} done in {:.1f}s ({} track(s), {}s audio)",
            job_id,
            elapsed,
            len(tracks),
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
