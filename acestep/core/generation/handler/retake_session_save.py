"""Persist reusable session artifacts for session-backed repaint retake."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


def can_save_generation_session_artifacts(result: Any) -> bool:
    """Return whether a generation result has the artifacts retake needs.

    Args:
        result: ``GenerationResult``-like object with audios and extra outputs.

    Returns:
        ``True`` only when every output track has audio codes and matching final
        latents are present. This is intentionally capability-based rather than
        task-name-based so any ACE-generated result with saved codes can become
        a most-natural repaint source.
    """
    if not getattr(result, "success", False):
        return False
    audios = list(getattr(result, "audios", None) or [])
    if not audios:
        return False
    extra = getattr(result, "extra_outputs", None) or {}
    pred_latents = extra.get("pred_latents")
    if pred_latents is None:
        return False
    shape = getattr(pred_latents, "shape", None)
    if shape is not None and len(shape) > 0 and int(shape[0]) < len(audios):
        return False
    return all(_track_audio_codes(audio.get("params") or {}) for audio in audios)


def save_generation_session_artifacts(
    *,
    result: Any,
    session_dir: str,
    source: str = "acestep",
) -> None:
    """Persist generated outputs as reusable retake session artifacts.

    Args:
        result: ``GenerationResult`` returned by inference.
        session_dir: Destination session directory.
        source: Short source label stored in ``session.json``.

    Raises:
        ValueError: If required audio codes or final latents are missing.
    """
    root = Path(session_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    extra = result.extra_outputs or {}
    lm_metadata = extra.get("lm_metadata")
    if lm_metadata is not None:
        _write_json(root / "lm_metadata.json", lm_metadata)

    pred_latents = extra.get("pred_latents")
    if pred_latents is None:
        raise ValueError("save_session_artifacts requires pred_latents in generation extra_outputs")

    tracks = []
    for index, audio in enumerate(result.audios or [], start=1):
        params = dict(audio.get("params") or {})
        audio_codes = _track_audio_codes(params)
        if not audio_codes:
            raise ValueError("save_session_artifacts requires per-track audio_codes")
        params_path = root / f"{index:02d}_params.json"
        _copy_session_audio(root, params, audio, index)
        if index - 1 >= pred_latents.shape[0]:
            raise ValueError("save_session_artifacts requires one latent tensor per track")
        latent = pred_latents[index - 1].detach().cpu().float().numpy()
        np.save(root / f"{index:02d}_latents.npy", latent)
        params["session_latents_file"] = f"{index:02d}_latents.npy"
        _write_json(params_path, params)
        tracks.append({"index": index, "params_file": params_path.name})

    _write_json(root / "session.json", {"source": source, "tracks": tracks})
    logger.info("[retake_session] Saved reusable session artifacts to {}", root)


def _copy_session_audio(
    root: Path,
    params: dict[str, Any],
    audio: dict[str, Any],
    index: int,
) -> None:
    """Copy the generated audio file into the reusable session directory."""
    audio_path = audio.get("path")
    if not audio_path or not os.path.exists(audio_path):
        return
    suffix = Path(audio_path).suffix or ".wav"
    copied_name = f"{index:02d}{suffix}"
    shutil.copyfile(audio_path, root / copied_name)
    params["session_audio_file"] = copied_name


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write JSON using UTF-8 and stable indentation."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=str)


def _track_audio_codes(params: dict[str, Any]) -> str:
    """Return non-empty per-track audio codes from saved parameter values."""
    return str(params.get("audio_codes") or params.get("audio_code_string") or "").strip()
