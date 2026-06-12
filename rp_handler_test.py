"""Unit tests for the RunPod handler's request parsing and audio-format logic.

These cover the pure-Python surface of ``rp_handler`` — request building,
clamping, alias resolution, LM gating and audio-format resolution — without
loading any model (``RP_HANDLER_SKIP_WARMUP=1``). The heavy/external ``loguru``
and ``runpod`` imports are stubbed so the suite runs on a CPU-only box.
"""

import os
import sys
import types
import unittest

os.environ.setdefault("RP_HANDLER_SKIP_WARMUP", "1")


def _install_stubs() -> None:
    """Stub loguru/runpod so importing rp_handler needs no GPU/runtime deps."""

    if "loguru" not in sys.modules:
        loguru = types.ModuleType("loguru")

        class _Logger:
            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        loguru.logger = _Logger()
        sys.modules["loguru"] = loguru

    if "runpod" not in sys.modules:
        runpod = types.ModuleType("runpod")
        runpod.serverless = types.SimpleNamespace(start=lambda *a, **k: None)
        sys.modules["runpod"] = runpod


_install_stubs()
import rp_handler as h  # noqa: E402  (after stubs are installed)


class BuildRequestTests(unittest.TestCase):
    """Behavior tests for ``_build_request`` parsing, aliases, and clamps."""

    def setUp(self):
        # Default to a pure-DiT worker; individual tests flip this as needed.
        h._lm_available = False

    def test_rejects_input_without_any_intent(self):
        """A request with no prompt, lyrics, or sample query is invalid."""

        with self.assertRaises(ValueError):
            h._build_request({})

    def test_requires_dict_input(self):
        """Non-dict input is rejected with a clear error."""

        with self.assertRaises(ValueError):
            h._build_request(["not", "a", "dict"])

    def test_caption_alias_maps_to_prompt(self):
        """The documented `caption` alias resolves to `prompt`."""

        req = h._build_request({"caption": "lofi chill"})
        self.assertEqual("lofi chill", req.prompt)

    def test_keyscale_camel_alias_resolves(self):
        """`keyScale` (camelCase alias) resolves to the canonical `key_scale`."""

        req = h._build_request({"prompt": "x", "keyScale": "D Major"})
        self.assertEqual("D Major", req.key_scale)

    def test_duration_alias_and_clamp_to_max(self):
        """`duration` alias maps to audio_duration and clamps to MAX_DURATION."""

        req = h._build_request({"prompt": "x", "duration": 10_000})
        self.assertEqual(h.MAX_DURATION, req.audio_duration)

    def test_duration_floor(self):
        """A positive sub-floor duration is raised to the 10s minimum."""

        req = h._build_request({"prompt": "x", "audio_duration": 3})
        self.assertEqual(10.0, req.audio_duration)

    def test_batch_size_clamped_to_max(self):
        """Oversized batch_size is clamped to MAX_BATCH_SIZE."""

        req = h._build_request({"prompt": "x", "batch_size": 99})
        self.assertEqual(h.MAX_BATCH_SIZE, req.batch_size)

    def test_steps_default_uses_env_tuned_value_when_omitted(self):
        """Omitting steps falls back to the base-DiT DEFAULT_STEPS, not turbo 8."""

        req = h._build_request({"prompt": "x"})
        self.assertEqual(h.DEFAULT_STEPS, req.inference_steps)

    def test_legacy_steps_key_is_honoured(self):
        """The old rp_handler `steps` key still maps to inference_steps."""

        req = h._build_request({"prompt": "x", "steps": 8})
        self.assertEqual(8, req.inference_steps)

    def test_canonical_inference_steps_wins_over_legacy(self):
        """`inference_steps` takes priority over the legacy `steps` key."""

        req = h._build_request({"prompt": "x", "inference_steps": 12, "steps": 8})
        self.assertEqual(12, req.inference_steps)

    def test_steps_clamped_to_max(self):
        """Explicit steps above MAX_STEPS are clamped."""

        req = h._build_request({"prompt": "x", "inference_steps": 100_000})
        self.assertEqual(h.MAX_STEPS, req.inference_steps)

    def test_thinking_gated_off_when_lm_unavailable(self):
        """thinking=True is forced off on a pure-DiT (no LM) worker."""

        h._lm_available = False
        req = h._build_request({"prompt": "x", "thinking": True})
        self.assertFalse(req.thinking)

    def test_thinking_enabled_when_lm_available(self):
        """thinking=True is honoured when the LM loaded."""

        h._lm_available = True
        req = h._build_request({"prompt": "x", "thinking": True})
        self.assertTrue(req.thinking)

    def test_explicit_seed_implies_deterministic(self):
        """A non-negative seed without use_random_seed disables random seeding."""

        req = h._build_request({"prompt": "x", "seed": 1234})
        self.assertFalse(req.use_random_seed)
        self.assertEqual(1234, req.seed)

    def test_explicit_use_random_seed_is_respected_over_seed(self):
        """An explicit use_random_seed=True is not overridden by a seed value."""

        req = h._build_request({"prompt": "x", "seed": 1234, "use_random_seed": True})
        self.assertTrue(req.use_random_seed)

    def test_negative_seed_keeps_random(self):
        """A negative seed leaves random seeding enabled (the default)."""

        req = h._build_request({"prompt": "x", "seed": -1})
        self.assertTrue(req.use_random_seed)

    def test_instrumental_flag_forces_sentinel_lyrics(self):
        """instrumental=true with no lyrics sets the [Instrumental] sentinel."""

        req = h._build_request({"prompt": "x", "instrumental": True})
        self.assertEqual("[Instrumental]", req.lyrics)

    def test_vocal_language_defaults_to_unknown_when_omitted(self):
        """Omitting vocal_language defaults to 'unknown' so the LM auto-detects."""

        req = h._build_request({"prompt": "x"})
        self.assertEqual("unknown", req.vocal_language)

    def test_vocal_language_explicit_is_respected(self):
        """An explicit vocal_language is honoured over the auto-detect default."""

        req = h._build_request({"prompt": "x", "vocal_language": "hi"})
        self.assertEqual("hi", req.vocal_language)

    def test_sample_mode_without_prompt_is_accepted(self):
        """sample_mode/sample_query is sufficient intent (no prompt/lyrics needed)."""

        req = h._build_request({"sample_mode": True, "sample_query": "a soft ballad"})
        self.assertTrue(req.sample_mode)
        self.assertEqual("a soft ballad", req.sample_query)


class SplitAudioFormatTests(unittest.TestCase):
    """``_split_audio_format`` -> (local_encode_format, cloudinary_target)."""

    def test_local_formats_encode_directly_no_transcode(self):
        """libsndfile-supported formats encode locally; no Cloudinary transcode."""

        for fmt in ("wav", "wav32", "flac", "mp3", "opus"):
            self.assertEqual((fmt, None), h._split_audio_format(fmt))

    def test_aac_encodes_wav_and_transcodes_via_cloudinary(self):
        """AAC has no in-worker encoder: encode WAV, let Cloudinary make AAC."""

        self.assertEqual(("wav", "aac"), h._split_audio_format("aac"))

    def test_unknown_defaults_to_mp3(self):
        """An unrecognised/empty format defaults to mp3 (no transcode)."""

        self.assertEqual(("mp3", None), h._split_audio_format("totally-bogus"))
        self.assertEqual(("mp3", None), h._split_audio_format(None))

    def test_case_insensitive(self):
        """Format matching is case-insensitive."""

        self.assertEqual(("flac", None), h._split_audio_format("FLAC"))
        self.assertEqual(("wav", "aac"), h._split_audio_format("AAC"))


class BuildGenerationMetadataTests(unittest.TestCase):
    """``_build_generation_metadata`` echo of the LM-generated song blueprint."""

    def _result(self, lm_metadata):
        """A minimal result stand-in exposing extra_outputs['lm_metadata']."""

        return types.SimpleNamespace(extra_outputs={"lm_metadata": lm_metadata})

    def test_lm_language_key_maps_to_vocal_language(self):
        """LM writes detected language under 'language'; the echo must surface it.

        Regression: the LM CoT parser stores the key as 'language' (not
        'vocal_language'), so reading only 'vocal_language' silently dropped it
        and the response always returned 'unknown'.
        """

        result = self._result({"language": "bn", "caption": "a soft bengali song"})
        meta = h._build_generation_metadata(result, {"params": {}}, {})
        self.assertEqual("bn", meta["vocal_language"])

    def test_canonical_vocal_language_key_still_works(self):
        """A source using the 'vocal_language' key is still honoured."""

        meta = h._build_generation_metadata(
            self._result({}), {"params": {"vocal_language": "ja"}}, {}
        )
        self.assertEqual("ja", meta["vocal_language"])

    def test_lm_language_wins_over_request_default(self):
        """LM 'language' (priority source) beats the raw-request fallback."""

        result = self._result({"language": "fr"})
        meta = h._build_generation_metadata(result, {"params": {}}, {"vocal_language": "en"})
        self.assertEqual("fr", meta["vocal_language"])

    def test_pure_dit_mode_falls_back_to_unknown(self):
        """No LM metadata and no user language -> 'unknown' (pure-DiT mode)."""

        meta = h._build_generation_metadata(self._result(None), {"params": {}}, {})
        self.assertEqual("unknown", meta["vocal_language"])


class ColdStartOverrideWarningTests(unittest.TestCase):
    """``_warn_cold_start_only_overrides`` must never raise on mismatches."""

    def test_mismatched_model_does_not_raise(self):
        """A model/lm mismatch only warns; building must still succeed."""

        req = h._build_request(
            {
                "prompt": "x",
                "model": "some-other-model",
                "lm_model_path": "acestep-5Hz-lm-9000B",
                "lm_backend": "pt",
            }
        )
        # _build_request already invoked the warning path; assert it built fine.
        self.assertEqual("some-other-model", req.model)


if __name__ == "__main__":
    unittest.main()
