"""Core audio generation with progressive UI yields.

Contains the main ``generate_with_progress`` generator that drives the
Gradio generate button: validates GPU limits, calls the inference
pipeline, saves audio files, and optionally runs auto-scoring and
auto-LRC in a single streaming pass.
"""
import json
import os
import time as time_module
import uuid

import gradio as gr
import torch
from loguru import logger

from acestep.audio_utils import save_audio
from acestep.core.generation.handler.retake_session_save import save_generation_session_artifacts
from acestep.gpu_config import (
    check_duration_limit,
    check_batch_size_limit,
    get_global_gpu_config,
)
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.ui.gradio.i18n import t
from acestep.ui.gradio.events.generation_handlers import parse_and_validate_timesteps
from acestep.ui.gradio.events.results.generation_info import (
    DEFAULT_RESULTS_DIR,
    _build_generation_info,
)
from acestep.ui.gradio.events.results.generation_task_type import resolve_no_fsq_task_type
from acestep.ui.gradio.events.results.audio_playback_updates import (
    build_audio_slot_update,
)
from acestep.ui.gradio.events.results.feature_cache import (
    build_storable_extra_outputs,
    feature_duration_seconds,
    load_sample_feature_data,
    persist_feature_cache,
)
from acestep.ui.gradio.events.results.scoring import calculate_score_handler
from acestep.ui.gradio.events.results.lrc_utils import lrc_to_vtt_file


def generate_with_progress(
    dit_handler, llm_handler,
    captions, lyrics, bpm, key_scale, time_signature, vocal_language,
    inference_steps, guidance_scale, random_seed_checkbox, seed,
    reference_audio, audio_duration, batch_size_input, src_audio,
    text2music_audio_code_string, repainting_start, repainting_end,
    instruction_display_gen, audio_cover_strength, cover_noise_strength, task_type,
    no_fsq, use_adg, cfg_interval_start, cfg_interval_end, shift, infer_method,
    sampler_mode, velocity_norm_threshold, velocity_ema_factor,
    dcw_enabled, dcw_mode, dcw_scaler, dcw_high_scaler, dcw_wavelet,
    custom_timesteps, audio_format, mp3_bitrate, mp3_sample_rate, lm_temperature,
    think_checkbox, lm_cfg_scale, lm_top_k, lm_top_p, lm_negative_prompt,
    use_cot_metas, use_cot_caption, use_cot_language, is_format_caption,
    constrained_decoding_debug,
    allow_lm_batch,
    auto_score,
    auto_lrc,
    score_scale,
    lm_batch_chunk_size,
    enable_normalization,
    normalization_db,
    fade_in_duration,
    fade_out_duration,
    latent_shift,
    latent_rescale,
    repaint_mode,
    repaint_strength,
    source_session_dir="",
    source_track_index=1,
    source_latent_mix_ratio=0.3,
    retake_variance=0.0,
    retake_seed="",
    flow_edit_morph=False,
    flow_edit_source_caption="",
    flow_edit_source_lyrics="",
    flow_edit_n_min=0.0,
    flow_edit_n_max=1.0,
    flow_edit_n_avg=1,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate audio with progress tracking.

    This is a Gradio generator that yields partial UI updates as each
    sample is processed, enabling progressive display of results.

    Yields:
        Tuple of Gradio component updates for the 52-output generate event.
    """
    # GPU memory validation
    gpu_config = get_global_gpu_config()
    lm_initialized = llm_handler.llm_initialized if llm_handler else False

    # Save-memory mode: force-disable features that require intermediate tensors
    if gpu_config.save_memory_mode:
        auto_score = False
        auto_lrc = False

    if audio_duration is not None and audio_duration > 0:
        is_valid, warning_msg = check_duration_limit(audio_duration, gpu_config, lm_initialized)
        if not is_valid:
            gr.Warning(warning_msg)
            max_dur = gpu_config.max_duration_with_lm if lm_initialized else gpu_config.max_duration_without_lm
            audio_duration = min(audio_duration, max_dur)
            logger.warning(f"Duration clamped to {audio_duration}s due to GPU memory limits")

    if batch_size_input is not None and batch_size_input > 0:
        is_valid, warning_msg = check_batch_size_limit(int(batch_size_input), gpu_config, lm_initialized)
        if not is_valid:
            gr.Warning(warning_msg)
            max_bs = gpu_config.max_batch_size_with_lm if lm_initialized else gpu_config.max_batch_size_without_lm
            batch_size_input = min(int(batch_size_input), max_bs)
            logger.warning(f"Batch size clamped to {batch_size_input} due to GPU memory limits")

    # Skip Phase 1 metas COT if sample is already formatted
    actual_use_cot_metas = use_cot_metas
    if is_format_caption and use_cot_metas:
        actual_use_cot_metas = False
        logger.info("[generate_with_progress] Skipping Phase 1 metas COT: is_format_caption=True")
        gr.Info(t("messages.skipping_metas_cot"))

    parsed_timesteps, _has_ts_warn, _ = parse_and_validate_timesteps(custom_timesteps, inference_steps)
    actual_inference_steps = len(parsed_timesteps) - 1 if parsed_timesteps is not None else inference_steps

    task_type = resolve_no_fsq_task_type(task_type, bool(no_fsq))

    # text2music never uses src_audio EXCEPT when flow_edit_morph is on:
    # the morph overlay needs the source audio for ``zt_src``/``zt_tar``
    # formation in the V_delta integration.  Without this guard the UI
    # silently zeroed src_audio for Custom mode and the backend's morph
    # check then errored with "Flow-edit morph requires a source audio".
    if task_type == "text2music" and not flow_edit_morph:
        src_audio = None

    # Defensive guard: cover/repaint/extract/lego tasks should never use
    # stale audio codes from the text2music_audio_code_string textbox.
    # Only text2music (Custom mode) with thinking disabled should pass codes.
    if task_type != "text2music":
        text2music_audio_code_string = ""

    gen_params = GenerationParams(
        task_type=task_type,
        instruction=instruction_display_gen,
        reference_audio=reference_audio,
        src_audio=src_audio,
        audio_codes=text2music_audio_code_string if not think_checkbox else "",
        caption=captions or "",
        lyrics=lyrics or "",
        instrumental=False,
        vocal_language=vocal_language,
        bpm=bpm,
        keyscale=key_scale,
        timesignature=time_signature,
        duration=audio_duration,
        inference_steps=actual_inference_steps,
        guidance_scale=guidance_scale,
        use_adg=use_adg,
        cfg_interval_start=cfg_interval_start,
        cfg_interval_end=cfg_interval_end,
        shift=shift,
        infer_method=infer_method,
        sampler_mode=sampler_mode,
        velocity_norm_threshold=velocity_norm_threshold,
        velocity_ema_factor=velocity_ema_factor,
        dcw_enabled=dcw_enabled,
        dcw_mode=dcw_mode,
        dcw_scaler=dcw_scaler,
        dcw_high_scaler=dcw_high_scaler,
        dcw_wavelet=dcw_wavelet,
        timesteps=parsed_timesteps,
        repainting_start=repainting_start,
        repainting_end=repainting_end,
        audio_cover_strength=audio_cover_strength,
        cover_noise_strength=cover_noise_strength,
        thinking=think_checkbox,
        lm_temperature=lm_temperature,
        lm_cfg_scale=lm_cfg_scale,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        lm_negative_prompt=lm_negative_prompt,
        use_cot_metas=actual_use_cot_metas,
        use_cot_caption=use_cot_caption,
        use_cot_language=use_cot_language,
        use_constrained_decoding=True,
        enable_normalization=enable_normalization,
        normalization_db=normalization_db,
        fade_in_duration=fade_in_duration if fade_in_duration else 0.0,
        fade_out_duration=fade_out_duration if fade_out_duration else 0.0,
        latent_shift=latent_shift,
        latent_rescale=latent_rescale,
        repaint_mode=repaint_mode if repaint_mode else "auto",
        repaint_strength=float(repaint_strength) if repaint_strength is not None else 0.5,
        source_session_dir=source_session_dir or None,
        source_track_index=int(source_track_index) if source_track_index else 1,
        source_latent_mix_ratio=(
            float(source_latent_mix_ratio) if source_latent_mix_ratio is not None else 0.3
        ),
        retake_variance=float(retake_variance) if retake_variance is not None else 0.0,
        # Empty textbox -> None; otherwise a string is fine (handler.prepare_seeds parses it).
        retake_seed=(retake_seed.strip() or None) if isinstance(retake_seed, str) else retake_seed,
        flow_edit_morph=bool(flow_edit_morph),
        flow_edit_source_caption=flow_edit_source_caption or "",
        flow_edit_source_lyrics=flow_edit_source_lyrics or "",
        flow_edit_n_min=float(flow_edit_n_min) if flow_edit_n_min is not None else 0.0,
        flow_edit_n_max=float(flow_edit_n_max) if flow_edit_n_max is not None else 1.0,
        flow_edit_n_avg=int(flow_edit_n_avg) if flow_edit_n_avg is not None else 1,
    )

    if isinstance(seed, str) and seed.strip():
        seed_list = [int(s.strip()) for s in seed.split(",")] if "," in seed else [int(seed.strip())]
    else:
        seed_list = None

    gen_config = GenerationConfig(
        batch_size=batch_size_input,
        allow_lm_batch=allow_lm_batch,
        use_random_seed=random_seed_checkbox,
        seeds=seed_list,
        lm_batch_chunk_size=lm_batch_chunk_size,
        constrained_decoding_debug=constrained_decoding_debug,
        audio_format=audio_format,
        mp3_bitrate=mp3_bitrate,
        mp3_sample_rate=mp3_sample_rate,
    )

    result = generate_music(
        dit_handler,
        llm_handler,
        params=gen_params,
        config=gen_config,
        progress=progress,
    )
    _persist_gradio_source_session(
        result=result,
        task_type=task_type,
        audio_codes=text2music_audio_code_string,
        think_enabled=bool(think_checkbox),
        lm_initialized=bool(lm_initialized),
        flow_edit_morph=bool(flow_edit_morph),
    )
    _persist_gradio_feature_cache(result)

    audio_outputs = [None] * 8
    all_audio_paths: list = []
    final_codes_list = [""] * 8
    final_scores_list = [""] * 8
    final_lrcs_list = [""] * 8
    final_subtitles_list = [None] * 8

    seed_value_for_ui = result.extra_outputs.get("seed_value", "")
    lm_generated_metadata = result.extra_outputs.get("lm_metadata", {})
    time_costs = result.extra_outputs.get("time_costs", {}).copy()

    audio_conversion_start_time = time_module.time()
    total_auto_score_time = 0.0
    total_auto_lrc_time = 0.0

    updated_audio_codes = text2music_audio_code_string if not think_checkbox else ""  # noqa: F841

    generation_info = _build_generation_info(
        lm_metadata=lm_generated_metadata,
        time_costs=time_costs,
        seed_value=seed_value_for_ui,
        inference_steps=inference_steps,
        num_audios=len(result.audios) if result.success else 0,
        audio_format=audio_format,
    )

    if not result.success:
        yield (
            (None,) * 8
            + (None, generation_info, result.status_message, gr.skip())
            + (gr.skip(),) * 8  # scores
            + (gr.skip(),) * 8  # codes_display
            + (gr.skip(),) * 8  # details_accordion
            + (gr.skip(),) * 8  # lrc_display
            + (None, is_format_caption, None, None)
        )
        return

    audios = result.audios
    progress(0.99, "Preparing audio files...")

    # Clear all scores/codes/lrc displays
    clear_scores = [gr.update(value="", visible=True) for _ in range(8)]
    clear_codes = [gr.update(value="", visible=True) for _ in range(8)]
    clear_lrcs = [gr.update(value="", visible=True) for _ in range(8)]
    clear_accordions = [gr.skip() for _ in range(8)]
    # Keep existing players mounted during generation to avoid browser volume reset.
    dump_audio = [gr.skip()] * 8

    yield (
        *dump_audio,
        None, generation_info, "Preparing generation...", gr.skip(),
        *clear_scores, *clear_codes, *clear_accordions, *clear_lrcs,
        lm_generated_metadata, is_format_caption, None, None,
    )
    time_module.sleep(0.1)

    for i in range(8):
        if i >= len(audios):
            continue

        key = audios[i]["key"]
        audio_tensor = audios[i]["tensor"]
        sample_rate = audios[i]["sample_rate"]
        audio_params = _audio_params_with_session_marker(
            audios[i]["params"],
            result.extra_outputs,
            i + 1,
        )

        timestamp = int(time_module.time())
        temp_dir = os.path.join(DEFAULT_RESULTS_DIR, f"batch_{timestamp}")
        temp_dir = os.path.abspath(temp_dir).replace("\\", "/")
        os.makedirs(temp_dir, exist_ok=True)
        json_path = os.path.join(temp_dir, f"{key}.json").replace("\\", "/")

        ext = "wav" if audio_format == "wav32" else audio_format
        audio_path = os.path.join(temp_dir, f"{key}.{ext}").replace("\\", "/")

        saved_path = save_audio(
            audio_data=audio_tensor, output_path=audio_path,
            sample_rate=sample_rate, format=audio_format, channels_first=True,
            mp3_bitrate=mp3_bitrate, mp3_sample_rate=mp3_sample_rate,
        )
        if saved_path:
            audio_path = saved_path.replace("\\", "/")

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(audio_params, f, indent=2, ensure_ascii=False)

        audio_outputs[i] = audio_path
        all_audio_paths.append(audio_path)
        all_audio_paths.append(json_path)

        code_str = audio_params.get("audio_codes", "")
        final_codes_list[i] = code_str

        scores_ui_updates = [gr.skip() for _ in range(8)]
        score_str = "Done!"

        if auto_score:
            auto_score_start = time_module.time()
            sample_tensor_data = _extract_sample_tensor(result.extra_outputs, i)
            score_str = calculate_score_handler(
                llm_handler, code_str, captions, lyrics, lm_generated_metadata,
                bpm, key_scale, time_signature, audio_duration, vocal_language,
                score_scale, dit_handler, sample_tensor_data, inference_steps,
            )
            total_auto_score_time += time_module.time() - auto_score_start

        scores_ui_updates[i] = score_str
        final_scores_list[i] = score_str

        if auto_lrc:
            auto_lrc_start = time_module.time()
            _run_auto_lrc(
                dit_handler, result.extra_outputs, i,
                audio_duration, vocal_language, inference_steps,
                final_lrcs_list, final_subtitles_list,
            )
            total_auto_lrc_time += time_module.time() - auto_lrc_start

        # STEP 1: yield audio + clear LRC
        cur_audio = [gr.skip()] * 8
        cur_audio[i] = build_audio_slot_update(gr, audio_path)
        cur_codes = [gr.skip()] * 8
        cur_codes[i] = gr.update(value=code_str, visible=True)
        cur_accordions = [gr.skip()] * 8
        lrc_clear = [gr.skip()] * 8
        lrc_clear[i] = gr.update(value="", visible=True)

        yield (
            *cur_audio,
            all_audio_paths, generation_info, f"Encoding & Ready: {i + 1}/{len(audios)}", seed_value_for_ui,
            *scores_ui_updates, *cur_codes, *cur_accordions, *lrc_clear,
            lm_generated_metadata, is_format_caption, None, None,
        )
        time_module.sleep(0.05)

        # STEP 2: set actual LRC (triggers .change() for subtitles)
        if final_lrcs_list[i]:
            skip8 = [gr.skip()] * 8
            lrc_set = [gr.skip()] * 8
            lrc_set[i] = gr.update(value=final_lrcs_list[i], visible=True)
            yield (
                *skip8,
                gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                *skip8, *skip8, *skip8, *lrc_set,
                gr.skip(), gr.skip(), None, None,
            )

        time_module.sleep(0.05)

    # Final timing
    audio_conversion_time = time_module.time() - audio_conversion_start_time
    if audio_conversion_time > 0:
        time_costs['audio_conversion_time'] = audio_conversion_time
    if total_auto_score_time > 0:
        time_costs['auto_score_time'] = total_auto_score_time
    if total_auto_lrc_time > 0:
        time_costs['auto_lrc_time'] = total_auto_lrc_time
    if 'pipeline_total_time' in time_costs:
        time_costs['pipeline_total_time'] += audio_conversion_time + total_auto_score_time + total_auto_lrc_time

    generation_info = _build_generation_info(
        lm_metadata=lm_generated_metadata,
        time_costs=time_costs,
        seed_value=seed_value_for_ui,
        inference_steps=inference_steps,
        num_audios=len(result.audios),
        audio_format=audio_format,
    )

    audio_playback_updates = []
    for idx in range(8):
        path = audio_outputs[idx]
        if path:
            audio_playback_updates.append(build_audio_slot_update(gr, path))
            logger.info(f"[generate_with_progress] Audio {idx + 1} path: {path}")
        else:
            audio_playback_updates.append(build_audio_slot_update(gr, None))

    final_codes_display = [gr.skip()] * 8
    final_accordions = [gr.skip()] * 8

    extra_to_store = build_storable_extra_outputs(
        result.extra_outputs,
        final_lrcs_list,
        final_subtitles_list,
    )
    for k, v in extra_to_store.items():
        if isinstance(v, torch.Tensor) and v.is_cuda:
            extra_to_store[k] = v.cpu()

    yield (
        *audio_playback_updates,
        all_audio_paths, generation_info, "Generation Complete", seed_value_for_ui,
        *final_scores_list, *final_codes_display, *final_accordions, *final_lrcs_list,
        lm_generated_metadata, is_format_caption,
        extra_to_store,
        final_codes_list,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_sample_tensor(extra_outputs, sample_idx):
    """Slice per-sample tensor data from *extra_outputs* for scoring.

    Returns ``None`` when data is missing or incomplete.
    """
    try:
        return load_sample_feature_data(extra_outputs, sample_idx)
    except Exception as e:
        print(f"[Auto Score] Failed to prepare tensor data for sample {sample_idx}: {e}")
        return None


def _audio_params_with_session_marker(
    audio_params: dict,
    extra_outputs: dict,
    track_index: int,
) -> dict:
    """Return output params annotated with hidden source-session identity.

    Args:
        audio_params: Per-audio parameter dictionary to write beside the WAV.
        extra_outputs: Generation extra outputs that may contain a session dir.
        track_index: One-based index of the audio inside the generated batch.

    Returns:
        A shallow copy of ``audio_params`` with session metadata when available.
    """
    params = dict(audio_params or {})
    session_dir = str((extra_outputs or {}).get("session_output_dir") or "").strip()
    if session_dir:
        params["session_output_dir"] = session_dir
        params["session_track_index"] = int(track_index)
    return params


def _persist_gradio_source_session(
    *,
    result,
    task_type: str,
    audio_codes: str,
    think_enabled: bool,
    lm_initialized: bool,
    flow_edit_morph: bool,
) -> None:
    """Save hidden source-session artifacts for Gradio text2music outputs.

    The artifacts are used by "Send To Repaint" to enable session-backed
    most-natural repaint without exposing filesystem fields in the UI.
    """
    if not _should_persist_gradio_source_session(
        task_type=task_type,
        audio_codes=audio_codes,
        think_enabled=think_enabled,
        lm_initialized=lm_initialized,
        flow_edit_morph=flow_edit_morph,
    ):
        return
    if not getattr(result, "success", False):
        return
    session_dir = _build_gradio_source_session_dir()
    try:
        save_generation_session_artifacts(result=result, session_dir=session_dir)
    except (OSError, ValueError) as exc:
        logger.warning("[gradio_repaint] Could not save source session artifacts: {}", exc)
        return
    result.extra_outputs["session_output_dir"] = session_dir


def _persist_gradio_feature_cache(result) -> None:
    """Save score/LRC feature tensors to disk and keep only paths in history."""
    if not getattr(result, "success", False):
        return
    extra_outputs = result.extra_outputs or {}
    if extra_outputs.get("feature_cache_files"):
        return
    cache_dir = extra_outputs.get("session_output_dir") or _build_gradio_feature_cache_dir()
    try:
        persist_feature_cache(extra_outputs, cache_dir)
    except (OSError, RuntimeError) as exc:
        logger.warning("[gradio_features] Could not save score/LRC feature cache: {}", exc)


def _should_persist_gradio_source_session(
    *,
    task_type: str,
    audio_codes: str,
    think_enabled: bool,
    lm_initialized: bool,
    flow_edit_morph: bool,
) -> bool:
    """Return whether Gradio should save hidden retake source artifacts."""
    if task_type != "text2music" or flow_edit_morph:
        return False
    if audio_codes and str(audio_codes).strip():
        return True
    return think_enabled and lm_initialized


def _build_gradio_source_session_dir() -> str:
    """Build a unique hidden session directory under Gradio outputs."""
    stamp = time_module.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    path = os.path.join(DEFAULT_RESULTS_DIR, "source_sessions", f"{stamp}_{suffix}")
    return os.path.abspath(path).replace("\\", "/")


def _build_gradio_feature_cache_dir() -> str:
    """Build a unique hidden feature-cache directory under Gradio outputs."""
    stamp = time_module.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    path = os.path.join(DEFAULT_RESULTS_DIR, "feature_sessions", f"{stamp}_{suffix}")
    return os.path.abspath(path).replace("\\", "/")


def _run_auto_lrc(dit_handler, extra_outputs, sample_idx,
                  audio_duration, vocal_language, inference_steps,
                  final_lrcs_list, final_subtitles_list):
    """Run automatic LRC generation for a single sample in-place.

    Updates *final_lrcs_list* and *final_subtitles_list* at *sample_idx*.
    """
    logger.info(f"[auto_lrc] Starting LRC generation for sample {sample_idx + 1}")
    try:
        feature_data = load_sample_feature_data(extra_outputs, sample_idx)
        if feature_data is None:
            logger.warning(f"[auto_lrc] Missing required extra_outputs for sample {sample_idx + 1}")
            return

        actual_duration = audio_duration
        if actual_duration is None or actual_duration <= 0:
            actual_duration = feature_duration_seconds(feature_data)
        if actual_duration is None:
            logger.warning(f"[auto_lrc] Missing duration for sample {sample_idx + 1}")
            return

        lrc_result = dit_handler.get_lyric_timestamp(
            pred_latent=feature_data["pred_latent"],
            encoder_hidden_states=feature_data["encoder_hidden_states"],
            encoder_attention_mask=feature_data["encoder_attention_mask"],
            context_latents=feature_data["context_latents"],
            lyric_token_ids=feature_data["lyric_token_ids"],
            total_duration_seconds=float(actual_duration),
            vocal_language=vocal_language or "en",
            inference_steps=int(inference_steps),
            seed=42,
        )

        if lrc_result.get("success"):
            lrc_text = lrc_result.get("lrc_text", "")
            final_lrcs_list[sample_idx] = lrc_text
            logger.info(f"[auto_lrc] LRC text length for sample {sample_idx + 1}: {len(lrc_text)}")
            vtt_path = lrc_to_vtt_file(lrc_text, total_duration=float(actual_duration))
            final_subtitles_list[sample_idx] = vtt_path
    except Exception as e:
        logger.warning(f"[auto_lrc] Failed to generate LRC for sample {sample_idx + 1}: {e}")
