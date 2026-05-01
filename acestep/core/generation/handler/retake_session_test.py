"""Tests for session-backed repaint retake artifact helpers."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from acestep.core.generation.handler.retake_session import (
    build_retake_generation_inputs,
    load_retake_source_track,
    normalize_repaint_mode_alias,
    resolve_repaint_mode,
    save_generation_session_artifacts,
)
from acestep.core.generation.handler.retake_session_save import (
    can_save_generation_session_artifacts,
)


class RetakeSessionTests(unittest.TestCase):
    """Verify retake source session loading and mode resolution."""

    def test_auto_mode_uses_retake_only_when_session_is_present(self):
        """Auto mode should preserve legacy balanced repaint without session."""
        self.assertEqual("balanced", resolve_repaint_mode("auto", None))
        self.assertEqual("most natural", resolve_repaint_mode("auto", "/tmp/session"))
        self.assertEqual("most natural", resolve_repaint_mode("most natural", "/tmp/session"))
        self.assertEqual("aggressive", resolve_repaint_mode("aggressive", "/tmp/session"))

    def test_retake_alias_normalizes_to_most_natural_public_label(self):
        """Legacy retake strings should map to the new user-facing label."""
        self.assertEqual("most natural", normalize_repaint_mode_alias("retake"))
        self.assertEqual("most natural", normalize_repaint_mode_alias("most_natural"))

    def test_load_source_track_requires_audio_codes(self):
        """Retake source tracks must include saved audio codes."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "01_params.json").write_text("{}", encoding="utf-8")
            np.save(root / "01_latents.npy", np.ones((4, 2), dtype=np.float32))

            with self.assertRaises(ValueError):
                load_retake_source_track(tmp, 1)

    def test_load_source_track_returns_codes_and_latents(self):
        """Valid source artifacts should load into retake source mapping."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            params = {
                "audio_codes": "<|audio_code_1|>",
                "cot_caption": "source caption",
                "lyrics": "la",
            }
            (root / "01_params.json").write_text(json.dumps(params), encoding="utf-8")
            np.save(root / "01_latents.npy", np.ones((50, 3), dtype=np.float32))

            source = load_retake_source_track(tmp, 1)

            self.assertEqual("<|audio_code_1|>", source["audio_codes"])
            self.assertEqual((50, 3), tuple(source["latents"].shape))
            self.assertAlmostEqual(2.0, source["duration"])

    def test_build_generation_inputs_prefers_overrides_then_source_metadata(self):
        """Caller text overrides source text, while missing fields use source data."""
        source = {
            "audio_codes": "<|audio_code_1|>",
            "duration": 4.0,
            "params": {"cot_caption": "source caption", "cot_bpm": 120, "lyrics": "old"},
            "lm_metadata": {"caption": "lm caption", "keyscale": "C major"},
        }

        values = build_retake_generation_inputs(source, {"caption": "override"})

        self.assertEqual("override", values["caption"])
        self.assertEqual("old", values["lyrics"])
        self.assertEqual(120, values["bpm"])
        self.assertEqual("C major", values["keyscale"])
        self.assertEqual("<|audio_code_1|>", values["audio_codes"])

    def test_save_session_artifacts_requires_complete_retake_inputs(self):
        """Reusable session persistence should not write incomplete artifacts."""
        result = SimpleNamespace(
            success=True,
            audios=[{"params": {"audio_codes": "<|audio_code_1|>"}}],
            extra_outputs={},
        )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "pred_latents"):
                save_generation_session_artifacts(result=result, session_dir=tmp)

    def test_save_session_artifacts_writes_loadable_track(self):
        """Saved generation artifacts should be directly loadable for retake."""
        result = SimpleNamespace(
            success=True,
            audios=[{"params": {"audio_codes": "<|audio_code_1|>", "caption": "source"}}],
            extra_outputs={"pred_latents": torch.ones(1, 6, 3)},
        )

        with tempfile.TemporaryDirectory() as tmp:
            save_generation_session_artifacts(result=result, session_dir=tmp)
            source = load_retake_source_track(tmp, 1)

            self.assertEqual("<|audio_code_1|>", source["audio_codes"])
            self.assertEqual((6, 3), tuple(source["latents"].shape))

    def test_can_save_session_artifacts_is_capability_based(self):
        """Any task result with audio codes and final latents can be a source."""
        result = SimpleNamespace(
            success=True,
            audios=[{"params": {"task_type": "cover", "audio_codes": "<|audio_code_1|>"}}],
            extra_outputs={"pred_latents": torch.ones(1, 6, 3)},
        )

        self.assertTrue(can_save_generation_session_artifacts(result))

    def test_can_save_session_artifacts_rejects_missing_codes(self):
        """Uploaded-audio style outputs without codes should not expose retake."""
        result = SimpleNamespace(
            success=True,
            audios=[{"params": {"task_type": "cover"}}],
            extra_outputs={"pred_latents": torch.ones(1, 6, 3)},
        )

        self.assertFalse(can_save_generation_session_artifacts(result))

    def test_can_save_session_artifacts_rejects_missing_latents(self):
        """Retake needs saved final latents as well as audio codes."""
        result = SimpleNamespace(
            success=True,
            audios=[{"params": {"audio_codes": "<|audio_code_1|>"}}],
            extra_outputs={},
        )

        self.assertFalse(can_save_generation_session_artifacts(result))


if __name__ == "__main__":
    unittest.main()
