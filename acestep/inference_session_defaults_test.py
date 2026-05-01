"""Tests for generated-source session default handling in inference."""

import unittest

from acestep.inference import _apply_source_session_defaults


class SourceSessionDefaultsTests(unittest.TestCase):
    """Verify session metadata only fills missing repaint inputs."""

    def test_session_defaults_fill_missing_metadata(self):
        """Generated-source repaint should reuse saved source metadata when absent."""
        result = _apply_source_session_defaults(
            {
                "bpm": "118",
                "keyscale": "C Major",
                "timesignature": "4/4",
                "duration": "27.5",
                "vocal_language": "en",
                "caption": "saved caption",
                "lyrics": "saved lyrics",
            },
            bpm=None,
            key_scale="",
            time_signature="",
            audio_duration=-1,
            vocal_language="unknown",
            caption="",
            lyrics="",
        )

        self.assertEqual((118, "C Major", "4/4", 27.5, "en", "saved caption", "saved lyrics"), result)

    def test_session_defaults_do_not_overwrite_user_edits(self):
        """Current request text should win over saved source text."""
        result = _apply_source_session_defaults(
            {
                "bpm": "118",
                "keyscale": "C Major",
                "timesignature": "4/4",
                "duration": "27.5",
                "vocal_language": "en",
                "caption": "saved caption",
                "lyrics": "saved lyrics",
            },
            bpm=96,
            key_scale="D Minor",
            time_signature="3/4",
            audio_duration=12.0,
            vocal_language="zh",
            caption="edited caption",
            lyrics="edited lyrics",
        )

        self.assertEqual((96, "D Minor", "3/4", 12.0, "zh", "edited caption", "edited lyrics"), result)


if __name__ == "__main__":
    unittest.main()
