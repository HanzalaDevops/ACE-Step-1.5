"""Tests for Gradio generation-progress helper behavior."""

import unittest

from acestep.ui.gradio.events.results.generation_progress import (
    _audio_params_with_session_marker,
    _should_persist_gradio_source_session,
)


class GradioSourceSessionPersistenceTests(unittest.TestCase):
    """Cover hidden source-session persistence gating."""

    def test_persists_text2music_when_llm_can_generate_audio_codes(self):
        """Text2music with initialized LM should persist source-session artifacts."""
        self.assertTrue(
            _should_persist_gradio_source_session(
                task_type="text2music",
                audio_codes="",
                think_enabled=True,
                lm_initialized=True,
                flow_edit_morph=False,
            )
        )

    def test_persists_text2music_with_user_audio_codes(self):
        """Provided audio codes should allow persistence without an initialized LM."""
        self.assertTrue(
            _should_persist_gradio_source_session(
                task_type="text2music",
                audio_codes="<|audio_code_1|>",
                think_enabled=False,
                lm_initialized=False,
                flow_edit_morph=False,
            )
        )

    def test_does_not_persist_repaint_or_morph_sources(self):
        """Only plain text2music outputs become Send To Repaint source sessions."""
        self.assertFalse(
            _should_persist_gradio_source_session(
                task_type="repaint",
                audio_codes="<|audio_code_1|>",
                think_enabled=True,
                lm_initialized=True,
                flow_edit_morph=False,
            )
        )
        self.assertFalse(
            _should_persist_gradio_source_session(
                task_type="text2music",
                audio_codes="",
                think_enabled=True,
                lm_initialized=True,
                flow_edit_morph=True,
            )
        )


class AudioParamsSessionMarkerTests(unittest.TestCase):
    """Cover generated sidecar metadata used by Send To Repaint."""

    def test_adds_session_identity_to_audio_params_copy(self):
        """Session-backed source generations should mark every audio JSON."""
        original = {"audio_codes": "<|audio_code_1|>"}

        result = _audio_params_with_session_marker(
            original,
            {"session_output_dir": "/tmp/source-session"},
            2,
        )

        self.assertEqual("<|audio_code_1|>", result["audio_codes"])
        self.assertEqual("/tmp/source-session", result["session_output_dir"])
        self.assertEqual(2, result["session_track_index"])
        self.assertNotIn("session_output_dir", original)

    def test_leaves_params_unchanged_without_session_identity(self):
        """Ordinary outputs should not claim a hidden source session."""
        result = _audio_params_with_session_marker({"audio_codes": "abc"}, {}, 1)

        self.assertEqual({"audio_codes": "abc"}, result)


if __name__ == "__main__":
    unittest.main()
