"""Resolve hidden source-session metadata for Gradio result transfers."""

import json
import os


def source_session_for_result(
    batch_queue,
    current_batch_index,
    audio_file=None,
    result_index=1,
) -> tuple[str, int]:
    """Return source-session directory and one-based track index for a result.

    Args:
        batch_queue: Gradio batch history state.
        current_batch_index: Currently selected batch index.
        audio_file: Generated audio file path for JSON-sidecar fallback.
        result_index: One-based result-card index.

    Returns:
        ``("", 1)`` when no valid generated source session can be found.
    """
    track_index = _coerce_track_index(result_index, default=1)
    if isinstance(batch_queue, dict):
        try:
            batch_index = int(current_batch_index)
        except (TypeError, ValueError):
            batch_index = None
        if batch_index is not None:
            batch_data = batch_queue.get(batch_index) or {}
            extra_outputs = batch_data.get("extra_outputs") or {}
            session_dir = _existing_session_dir(extra_outputs.get("session_output_dir"))
            if session_dir:
                return session_dir, track_index

    session_dir, sidecar_track_index = _source_session_from_audio_sidecar(
        audio_file,
        default_track_index=track_index,
    )
    if session_dir:
        return session_dir, sidecar_track_index
    return "", 1


def _source_session_from_audio_sidecar(audio_file, default_track_index: int) -> tuple[str, int]:
    """Load source-session metadata from a generated audio sidecar JSON."""
    if not audio_file:
        return "", 1
    try:
        audio_path = os.fspath(audio_file)
    except TypeError:
        return "", 1
    json_path = os.path.splitext(audio_path)[0] + ".json"
    if not os.path.exists(os.path.expanduser(json_path)):
        return "", 1
    try:
        with open(json_path, encoding="utf-8") as file_obj:
            params = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return "", 1
    if not isinstance(params, dict):
        return "", 1
    session_dir = _existing_session_dir(params.get("session_output_dir"))
    if not session_dir:
        return "", 1
    track_index = _coerce_track_index(
        params.get("session_track_index"),
        default=default_track_index,
    )
    return session_dir, track_index


def _existing_session_dir(session_dir) -> str:
    """Return ``session_dir`` only when it resolves to an existing directory."""
    session_dir = str(session_dir or "").strip()
    if not session_dir:
        return ""
    expanded = os.path.expanduser(session_dir)
    return session_dir if os.path.isdir(expanded) else ""


def _coerce_track_index(value, default: int) -> int:
    """Coerce one-based source track indices from Gradio values."""
    try:
        index = int(value)
    except (TypeError, ValueError):
        return int(default)
    return index if index > 0 else int(default)
