#!/usr/bin/env python3
"""Cloudinary upload helper for the RunPod ACE-Step worker.

Generated audio is uploaded to Cloudinary and the worker returns a URL instead
of an inline base64 blob. This keeps the RunPod ``/job-done`` response small so
long songs never trip the platform's response-size limit (raw base64 of a
multi-minute clip exceeds it and the job is rejected with HTTP 400).

Audio is stored under Cloudinary's ``video`` resource type — Cloudinary models
audio as a subset of its video pipeline, which is also what enables on-upload
transcoding (e.g. wav -> aac, a format libsndfile cannot encode in-worker).

Configuration is read from the environment — set the three discrete
``CLOUDINARY_CLOUD_NAME`` / ``CLOUDINARY_API_KEY`` / ``CLOUDINARY_API_SECRET``
vars (or a single ``CLOUDINARY_URL``) as RunPod Secrets. The upload folder comes
entirely from the request's ``cloudinary_folder``. No credential is ever
hardcoded here.
"""
from __future__ import annotations

import io
import os
from typing import Any, Dict, Optional

from loguru import logger

# Audio is uploaded under Cloudinary's "video" resource type (audio is part of
# the video pipeline — this is what enables format transcoding like wav -> aac).
_RESOURCE_TYPE = "video"

# Discrete credential vars accepted as an alternative to a single CLOUDINARY_URL.
_DISCRETE_VARS = ("CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET")


class CloudinaryUploadError(RuntimeError):
    """Raised when audio cannot be uploaded (missing config or API failure)."""


def is_configured() -> bool:
    """True when Cloudinary credentials are present in the environment.

    Accepts either a single ``CLOUDINARY_URL`` or all three discrete
    cloud_name/api_key/api_secret vars.
    """
    if os.environ.get("CLOUDINARY_URL", "").strip():
        return True
    return all(os.environ.get(var, "").strip() for var in _DISCRETE_VARS)


def _ensure_config() -> None:
    """Load Cloudinary config from the environment (idempotent).

    ``cloudinary.config()`` with no args reads ``CLOUDINARY_URL``; when only the
    discrete vars are set they are passed explicitly so either style works.
    """
    import cloudinary

    if os.environ.get("CLOUDINARY_URL", "").strip():
        cloudinary.config(secure=True)
    else:
        cloudinary.config(
            cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", "").strip(),
            api_key=os.environ.get("CLOUDINARY_API_KEY", "").strip(),
            api_secret=os.environ.get("CLOUDINARY_API_SECRET", "").strip(),
            secure=True,
        )


def upload_audio(
    audio_bytes: bytes,
    folder: str = "",
    *,
    target_format: Optional[str] = None,
    public_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload in-memory audio bytes to Cloudinary and return its URL + metadata.

    Args:
        audio_bytes: the encoded audio payload to upload.
        folder: Cloudinary folder to upload into (the full path from the
            request's ``cloudinary_folder``). Empty uploads to the account root.
        target_format: when set, Cloudinary transcodes the stored asset to this
            format — used for ``aac`` (libsndfile cannot encode it locally, so
            the worker uploads lossless WAV and lets Cloudinary convert). When
            ``None`` Cloudinary keeps the uploaded bytes' own format.
        public_id: optional custom public id (without extension); when omitted
            Cloudinary assigns a unique random id (safe for batch uploads).

    Returns:
        ``{"url": str, "public_id": str|None, "format": str|None, "bytes": int|None}``

    Raises:
        CloudinaryUploadError: if Cloudinary is not configured or the API call
            fails / returns no URL.
    """
    if not is_configured():
        raise CloudinaryUploadError(
            "Cloudinary is not configured — set CLOUDINARY_URL (or "
            "CLOUDINARY_CLOUD_NAME/API_KEY/API_SECRET) on the endpoint."
        )

    try:
        import cloudinary.uploader

        _ensure_config()

        options: Dict[str, Any] = {"resource_type": _RESOURCE_TYPE}
        folder = str(folder or "").strip().strip("/")
        if folder:
            options["folder"] = folder
        if target_format:
            options["format"] = target_format
        if public_id:
            options["public_id"] = public_id

        result = cloudinary.uploader.upload(io.BytesIO(audio_bytes), **options)
    except CloudinaryUploadError:
        raise
    except Exception as exc:  # cloudinary.exceptions.Error and anything else
        raise CloudinaryUploadError(f"Cloudinary upload failed: {exc}") from exc

    url = result.get("secure_url") or result.get("url")
    if not url:
        raise CloudinaryUploadError(f"Cloudinary upload returned no URL: {result!r}")

    logger.info(
        "[cloudinary] uploaded public_id={} ({} bytes) -> {}",
        result.get("public_id"),
        result.get("bytes"),
        url,
    )
    return {
        "url": url,
        "public_id": result.get("public_id"),
        "format": result.get("format"),
        "bytes": result.get("bytes"),
    }
