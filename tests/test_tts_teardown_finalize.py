#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for two-tier audio-context finalization in TTSService.

The full ``stop_frame_timeout_s`` is a TTFB guard for the *first* audio chunk
(intentionally large to avoid silent greetings). Once audio has been received it
is unnecessary and only delays the EndFrame -> telephony stop -> caller hangup at
end of call, so a shorter ``audio_done_timeout_s`` is used for end-of-turn
detection. The first-chunk (TTFB) wait must keep the full timeout.
"""

import unittest
from typing import AsyncGenerator

from pipecat.frames.frames import Frame
from pipecat.services.tts_service import TTSService


class _MiniTTS(TTSService):
    """Minimal concrete TTSService for exercising base-class finalization logic."""

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        if False:  # pragma: no cover - never yields; we test timeout selection only
            yield None


class TestTwoTierFinalization(unittest.IsolatedAsyncioTestCase):
    def _svc(self, **kwargs):
        return _MiniTTS(sample_rate=16000, **kwargs)

    def test_full_timeout_before_first_audio(self):
        # TTFB guard: no audio yet → keep the full timeout (silent-greeting guard).
        svc = self._svc(stop_frame_timeout_s=8.0, audio_done_timeout_s=1.0)
        self.assertEqual(svc._audio_context_timeout(audio_received=False), 8.0)

    def test_short_timeout_after_audio(self):
        # End-of-turn detection: audio received → use the short timeout.
        svc = self._svc(stop_frame_timeout_s=8.0, audio_done_timeout_s=1.0)
        self.assertEqual(svc._audio_context_timeout(audio_received=True), 1.0)

    def test_disabled_second_tier_uses_full(self):
        # audio_done_timeout_s=None → original single-tier behavior.
        svc = self._svc(stop_frame_timeout_s=8.0, audio_done_timeout_s=None)
        self.assertEqual(svc._audio_context_timeout(audio_received=True), 8.0)
        self.assertEqual(svc._audio_context_timeout(audio_received=False), 8.0)

    def test_short_timeout_never_exceeds_full(self):
        # If the full timeout is configured below the post-audio value, never lengthen.
        svc = self._svc(stop_frame_timeout_s=0.5, audio_done_timeout_s=1.0)
        self.assertEqual(svc._audio_context_timeout(audio_received=True), 0.5)


if __name__ == "__main__":
    unittest.main()
