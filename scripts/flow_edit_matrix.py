"""Flow-edit cross-model + cross-use-case smoke test (#1156, #1167).

Purpose
-------
Verify that flow-edit produces non-silent, on-prompt audio across the 6
DiT variants and the canonical use cases.  Designed to be runnable in
small slices so we don't burn GPU time before listening to earlier tier
results.

Phases (recommended order — each is independent):

  A. Turbo cross-model compat (PR #1167, NEW)
       --models turbo,xl_turbo --use-cases only_lyrics
  B. Base-tier cross-model compat
       --models xl_base,xl_sft,sft,base --use-cases only_lyrics
  C. Use-case sweep on representative model
       --models xl_base --use-cases only_lyrics,remix,full,tiny,navg4,early
  D. n_avg sensitivity
       --models xl_base --use-cases navg1,navg2,navg4

Usage (jieyue, GPU 1):
    cd /root/data/repo/gongjunmin/workspace/ACE-Step-1.5
    conda activate acestep_v15_train
    CUDA_VISIBLE_DEVICES=1 python scripts/flow_edit_matrix.py \\
        --models turbo,xl_turbo --use-cases only_lyrics \\
        --output-dir flow_edit_matrix_outputs/

Outputs:
  {output_dir}/baseline_{model}.{ext}      generated once per model, reused
  {output_dir}/edit_{model}_{use_case}.{ext}
  {output_dir}/run_log.jsonl               one line per generation
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler


# Registered config_path → DiT variant alias
MODEL_CONFIGS = {
    "turbo":     "acestep-v15-turbo",
    "xl_turbo":  "acestep-v15-xl-turbo",
    "xl_base":   "acestep-v15-xl-base",
    "xl_sft":    "acestep-v15-xl-sft",
    "sft":       "acestep-v15-sft",
    "base":      "acestep-v15-base",
}


# Source prompt — same as the local 3-WAV baseline so listening test
# stays comparable.  Anime-pop, Chinese vocals, 160s, B minor, 100 BPM.
SRC_CAPTION = (
    "An explosive, high-energy pop-rock track with a strong anime theme song "
    "feel.  The song kicks off with a catchy, synthesized brass fanfare over "
    "a driving rock beat with punchy drums and a solid bassline.  A powerful, "
    "clear male vocal enters with a theatrical and energetic delivery."
)
SRC_LYRICS = (
    "[Verse]\n黑夜里的风吹过耳畔\n甜蜜时光转瞬即万\n"
    "[Chorus]\n心电感应在震动间\n拥抱未来勇敢冒险\n"
)
SRC_BPM = 100
SRC_DURATION = 160.0
SRC_KEYSCALE = "B minor"
SRC_LANG = "zh"
SRC_TIMESIG = "4"
SRC_SEED = 42


# Target lyrics for "change-the-words" cases (only_lyrics, navg4).
TGT_LYRICS = (
    "[Verse]\n阳光洒落在街道上\n微风轻拂心情舒畅\n"
    "[Chorus]\n那旋律飞向远方\n带着希望和梦想\n"
)
# Caption variants for "change-the-style" cases.
LOFI = ("A mellow lo-fi hip-hop track with warm vinyl crackle, jazzy piano "
        "chords, soft brushed drums, and dreamy atmospheric pads.")
ORCH = ("A cinematic orchestral piece with sweeping strings, dramatic brass, "
        "and timpani.")
ROCK = ("A stadium-rock arena anthem with crunchy distorted guitars and huge "
        "gang vocals on the chorus.")

# Each preset: (n_min, n_max, n_avg, target_caption, target_lyrics).
USE_CASES = {
    "only_lyrics": (0.6, 1.0, 1, None, TGT_LYRICS),     # keep melody
    "remix":       (0.2, 0.4, 1, LOFI, None),           # keep lyrics
    "full":        (0.0, 1.0, 1, ORCH, "[Instrumental]"),
    "tiny":        (0.4, 0.5, 1, None, None),           # near-identity
    "navg4":       (0.6, 1.0, 4, None, TGT_LYRICS),     # n_avg sensitivity
    "early":       (0.0, 0.3, 1, ROCK, None),           # early-only edit
}


def _baseline_params(model: str) -> GenerationParams:
    return GenerationParams(
        task_type="text2music",
        caption=SRC_CAPTION,
        lyrics=SRC_LYRICS,
        bpm=SRC_BPM,
        keyscale=SRC_KEYSCALE,
        timesignature=SRC_TIMESIG,
        vocal_language=SRC_LANG,
        duration=SRC_DURATION,
        inference_steps=8 if "turbo" in model else 32,
        guidance_scale=1.0 if "turbo" in model else 7.0,
        shift=3.0,
        seed=SRC_SEED,
        thinking=False,
    )


def _edit_params(model: str, src_audio: str, uc: str) -> GenerationParams:
    n_min, n_max, n_avg, tgt_cap, tgt_lyr = USE_CASES[uc]
    p = _baseline_params(model)
    p.task_type = "edit"
    p.src_audio = src_audio
    p.edit_target_caption = tgt_cap if tgt_cap is not None else SRC_CAPTION
    p.edit_target_lyrics = tgt_lyr if tgt_lyr is not None else SRC_LYRICS
    p.edit_n_min, p.edit_n_max, p.edit_n_avg = n_min, n_max, n_avg
    return p


def _run_one(dit, llm, params: GenerationParams, save_dir: str, label: str):
    cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[SRC_SEED])
    t0 = time.time()
    res = generate_music(dit, llm, params, cfg, save_dir=save_dir)
    return {
        "label": label,
        "ok": bool(res.audio_paths),
        "audio": res.audio_paths[0] if res.audio_paths else None,
        "elapsed_s": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="comma-sep: turbo,xl_turbo,xl_base,xl_sft,sft,base")
    ap.add_argument("--use-cases", required=True,
                    help=f"comma-sep: {','.join(USE_CASES)}")
    ap.add_argument("--output-dir", default="flow_edit_matrix_outputs")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    use_cases = [u.strip() for u in args.use_cases.split(",") if u.strip()]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    log = (out / "run_log.jsonl").open("a")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for model in models:
        cfg_path = MODEL_CONFIGS[model]
        print(f"\n=== {model} (config={cfg_path}) ===")
        dit = AceStepHandler()
        llm = LLMHandler()
        _, ok = dit.initialize_service(
            project_root=project_root, config_path=cfg_path,
            device=args.device, use_flash_attention=False,
            compile_model=False, offload_to_cpu=False,
            offload_dit_to_cpu=False, quantization=None,
        )
        if not ok:
            print(f"  init failed for {model}; skipping"); continue

        baseline_path = out / f"baseline_{model}.flac"
        if not baseline_path.exists():
            r = _run_one(dit, llm, _baseline_params(model), str(out),
                         f"baseline_{model}")
            log.write(json.dumps(r) + "\n"); log.flush()
            print(f"  baseline → {r['audio']} ({r['elapsed_s']}s)")
            if r["audio"]:
                Path(r["audio"]).rename(baseline_path)

        for uc in use_cases:
            tag = f"edit_{model}_{uc}"
            r = _run_one(dit, llm,
                         _edit_params(model, str(baseline_path), uc),
                         str(out), tag)
            log.write(json.dumps(r) + "\n"); log.flush()
            print(f"  {uc:14s} → {r['audio']} ({r['elapsed_s']}s)")
            if r["audio"]:
                Path(r["audio"]).rename(out / f"{tag}.flac")

    log.close()


if __name__ == "__main__":
    main()
