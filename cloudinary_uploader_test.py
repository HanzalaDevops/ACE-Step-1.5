"""Unit tests for ``cloudinary_uploader`` — config detection and upload.

The Cloudinary SDK is stubbed (no network, no credentials needed) so these run
on any box. ``loguru`` is stubbed too, matching ``rp_handler_test``.
"""

import os
import sys
import types
import unittest
from unittest import mock


def _install_stubs() -> None:
    """Stub loguru and the cloudinary SDK so the module imports without deps."""

    if "loguru" not in sys.modules:
        loguru = types.ModuleType("loguru")

        class _Logger:
            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        loguru.logger = _Logger()
        sys.modules["loguru"] = loguru


_install_stubs()
import cloudinary_uploader as cu  # noqa: E402  (after stubs)


def _install_fake_cloudinary(upload_return=None, upload_side_effect=None):
    """Register a fake ``cloudinary`` + ``cloudinary.uploader`` in sys.modules.

    Returns the upload Mock so tests can assert how it was called.
    """
    fake_cloudinary = types.ModuleType("cloudinary")
    fake_cloudinary.config = mock.Mock()

    fake_uploader = types.ModuleType("cloudinary.uploader")
    upload_mock = mock.Mock(side_effect=upload_side_effect, return_value=upload_return)
    fake_uploader.upload = upload_mock
    fake_cloudinary.uploader = fake_uploader

    sys.modules["cloudinary"] = fake_cloudinary
    sys.modules["cloudinary.uploader"] = fake_uploader
    return upload_mock


class IsConfiguredTests(unittest.TestCase):
    """``is_configured`` recognises both credential styles."""

    def test_cloudinary_url_is_enough(self):
        with mock.patch.dict(os.environ, {"CLOUDINARY_URL": "cloudinary://k:s@c"}, clear=True):
            self.assertTrue(cu.is_configured())

    def test_three_discrete_vars_are_enough(self):
        env = {
            "CLOUDINARY_CLOUD_NAME": "c",
            "CLOUDINARY_API_KEY": "k",
            "CLOUDINARY_API_SECRET": "s",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(cu.is_configured())

    def test_partial_discrete_vars_are_not_enough(self):
        env = {"CLOUDINARY_CLOUD_NAME": "c", "CLOUDINARY_API_KEY": "k"}  # secret missing
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(cu.is_configured())

    def test_no_config_is_false(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(cu.is_configured())


class UploadAudioTests(unittest.TestCase):
    """``upload_audio`` config-gating, options, and error handling."""

    def test_raises_when_not_configured(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(cu.CloudinaryUploadError):
                cu.upload_audio(b"abc", "songs")

    def test_returns_secure_url_and_metadata(self):
        upload_mock = _install_fake_cloudinary(
            upload_return={
                "secure_url": "https://res.cloudinary.com/c/video/upload/songs/x.mp3",
                "public_id": "songs/x",
                "format": "mp3",
                "bytes": 123,
            }
        )
        with mock.patch.dict(os.environ, {"CLOUDINARY_URL": "cloudinary://k:s@c"}, clear=True):
            result = cu.upload_audio(b"audio-bytes", "songs/user123")

        self.assertEqual("https://res.cloudinary.com/c/video/upload/songs/x.mp3", result["url"])
        self.assertEqual("songs/x", result["public_id"])
        self.assertEqual("mp3", result["format"])
        # Audio uploads under the "video" resource type, into the given folder.
        _, kwargs = upload_mock.call_args
        self.assertEqual("video", kwargs["resource_type"])
        self.assertEqual("songs/user123", kwargs["folder"])
        self.assertNotIn("format", kwargs)  # no transcode requested

    def test_aac_target_passes_format_for_transcode(self):
        upload_mock = _install_fake_cloudinary(
            upload_return={"secure_url": "https://x/y.aac", "public_id": "y", "format": "aac"}
        )
        with mock.patch.dict(os.environ, {"CLOUDINARY_URL": "cloudinary://k:s@c"}, clear=True):
            cu.upload_audio(b"wav-bytes", "songs", target_format="aac")

        _, kwargs = upload_mock.call_args
        self.assertEqual("aac", kwargs["format"])

    def test_empty_folder_is_omitted(self):
        upload_mock = _install_fake_cloudinary(
            upload_return={"secure_url": "https://x/y.mp3", "public_id": "y", "format": "mp3"}
        )
        with mock.patch.dict(os.environ, {"CLOUDINARY_URL": "cloudinary://k:s@c"}, clear=True):
            cu.upload_audio(b"bytes", "")

        _, kwargs = upload_mock.call_args
        self.assertNotIn("folder", kwargs)

    def test_api_error_is_wrapped(self):
        _install_fake_cloudinary(upload_side_effect=RuntimeError("boom"))
        with mock.patch.dict(os.environ, {"CLOUDINARY_URL": "cloudinary://k:s@c"}, clear=True):
            with self.assertRaises(cu.CloudinaryUploadError):
                cu.upload_audio(b"bytes", "songs")

    def test_missing_url_in_response_is_an_error(self):
        _install_fake_cloudinary(upload_return={"public_id": "y", "format": "mp3"})  # no url
        with mock.patch.dict(os.environ, {"CLOUDINARY_URL": "cloudinary://k:s@c"}, clear=True):
            with self.assertRaises(cu.CloudinaryUploadError):
                cu.upload_audio(b"bytes", "songs")


if __name__ == "__main__":
    unittest.main()
