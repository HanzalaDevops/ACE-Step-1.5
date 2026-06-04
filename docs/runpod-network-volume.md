# Fixing RunPod cold-start timeouts with a Network Volume

## The problem

On a fresh worker the logs show:

```
Main model not found. Starting automatic download...
Downloading main model from ACE-Step/Ace-Step1.5 -> /app/checkpoints
```

The weights are **not** in the image, so every cold-start re-downloads the full
model (5–20 min). `/runsync` times out before the download finishes, so you get
no audio. The fix is to download the model **once** to a persistent Network
Volume and have every worker read from it.

## What the code already does

- `acestep.model_downloader.get_checkpoints_dir()` resolves the checkpoints
  directory from `ACESTEP_CHECKPOINTS_DIR` first, then falls back to
  `<project_root>/checkpoints`.
- Both the **DiT pipeline** (`initialize_service`) and the **5Hz LM**
  (`rp_handler.py`) now use that resolution, so a single env var redirects all
  weights to the volume. (The LM previously ignored it — fixed.)

## One-time setup

### 1. Create the Network Volume

RunPod Console → **Storage** → **Network Volume** → create one in the **same
data center** you'll run the endpoint in. Size it for the full model
(≈30–50 GB is a safe starting point; check the repo size and grow if needed).

### 2. Pre-fetch the weights from a temporary Pod

Serverless workers are ephemeral and time-limited, so populate the volume from
a normal **Pod** (not Serverless):

1. Deploy any small GPU/CPU Pod and **attach the network volume** — it mounts at
   `/runpod-volume`.
2. Open the Pod's web terminal / SSH and run:

   ```bash
   pip install huggingface_hub          # if not already present
   export HF_TOKEN=hf_your_token_here   # gated repo needs this
   git clone https://github.com/HanzalaDevops/ACE-Step-1.5.git
   cd ACE-Step-1.5
   python scripts/prefetch_models.py
   ```

   The script downloads `ACE-Step/Ace-Step1.5` into
   `/runpod-volume/checkpoints`, refuses to run if the volume isn't mounted, and
   is resume-safe (re-run if it's interrupted).

3. Confirm the files landed, then **terminate the Pod** (the volume and its
   contents persist).

### 3. Configure the Serverless endpoint

On your endpoint → **Settings**:

- **Attach** the same network volume (mounts at `/runpod-volume`).
- **Environment Variables** (use RunPod *Secrets* for the token):

  | Variable | Value |
  | --- | --- |
  | `ACESTEP_CHECKPOINTS_DIR` | `/runpod-volume/checkpoints` |
  | `HF_TOKEN` | `hf_…` |
  | `HUGGING_FACE_HUB_TOKEN` | `hf_…` |

  (Other ACE-Step vars are baked into `Dockerfile.runpod`; override only what
  you need — see `.env.runpod.example`.)

### 4. Verify

Send a job. The cold-start log should now read:

```
[rp_handler] Checkpoints directory: /runpod-volume/checkpoints
... Main model already exists at /runpod-volume/checkpoints
```

No download line → weights are being served from the volume. `/runsync` returns
audio in ~10–30 s once the worker is warm.

## Notes

- The volume is shared across all workers on the endpoint, so scaling out
  doesn't re-download anything.
- The `torch >= 2.11.0` cpp-extension warning is unrelated to this and is
  non-blocking — it only disables an optional speedup (code falls back to
  PyTorch SDPA).
- If you ever change `CHECKPOINT_PATH` to a different repo, re-run the prefetch
  step for that repo.
