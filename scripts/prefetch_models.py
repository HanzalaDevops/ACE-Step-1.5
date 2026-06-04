#!/usr/bin/env python3
"""
Pre-fetch ACE-Step 1.5 weights into a RunPod Network Volume.

Run this ONCE from a temporary RunPod **Pod** (not Serverless) that has the
network volume mounted at ``/runpod-volume``. After it completes, point your
Serverless endpoint at the same volume and set::

    ACESTEP_CHECKPOINTS_DIR=/runpod-volume/checkpoints

so every cold-start finds the weights instantly instead of re-downloading.

Usage (inside the Pod's web terminal / SSH):

    export HF_TOKEN=hf_xxx          # required for the gated repo
    python scripts/prefetch_models.py

Override the destination or repo if needed:

    ACESTEP_CHECKPOINTS_DIR=/runpod-volume/checkpoints \
    CHECKPOINT_PATH=ACE-Step/Ace-Step1.5 \
    python scripts/prefetch_models.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Destination on the mounted network volume. Must match ACESTEP_CHECKPOINTS_DIR
# on the Serverless endpoint.
CHECKPOINTS_DIR = os.environ.get(
    "ACESTEP_CHECKPOINTS_DIR", "/runpod-volume/checkpoints"
)
# Core weights repo (VAE + text encoder + default DiT).
MAIN_REPO = os.environ.get("CHECKPOINT_PATH", "ACE-Step/Ace-Step1.5")
# Token for the gated repo; huggingface_hub reads either name.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed. Run "
            "`pip install huggingface_hub` first.",
            file=sys.stderr,
        )
        return 1

    dest = Path(CHECKPOINTS_DIR)
    volume_root = dest.parent

    # Fail fast if the volume isn't actually mounted — otherwise we'd silently
    # write to the Pod's ephemeral disk and the weights would vanish on stop.
    if not volume_root.exists():
        print(
            f"ERROR: {volume_root} does not exist. Attach the network volume to "
            "this Pod (it should mount at /runpod-volume) before running.",
            file=sys.stderr,
        )
        return 1

    if not HF_TOKEN:
        print(
            "WARNING: no HF_TOKEN / HUGGING_FACE_HUB_TOKEN set. The download may "
            "fail or be rate-limited if the repo is gated.",
            file=sys.stderr,
        )

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MAIN_REPO}")
    print(f"  -> {dest}")

    snapshot_download(
        repo_id=MAIN_REPO,
        local_dir=str(dest),
        token=HF_TOKEN,
        # Resume-friendly: re-running skips files already present.
        resume_download=True,
    )

    print("Done. Contents of the checkpoints directory:")
    for child in sorted(dest.iterdir()):
        print(f"  {child.name}")
    print(
        "\nNext: on your Serverless endpoint set "
        f"ACESTEP_CHECKPOINTS_DIR={CHECKPOINTS_DIR} and attach this volume."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
