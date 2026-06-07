#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for end-of-call audio-context finalization in TTSService.

At end of call the EndFrame (and therefore the telephony stop event / caller
hangup) is ordered behind the goodbye's TTSStoppedFrame, which is emitted when
the audio context finalizes. The full stop_frame_timeout_s is a TTFB guard for
the *first* audio chunk (intentionally large to avoid silent greetings). Once
audio has flowed and an EndFrame is processed, the context should finalize via a
much shorter timeout so the stop isn't delayed by the full idle wait.
"""

import asyncio
import unittest
from typing import AsyncGenerator

from pipecat.frames.frames import Frame
from pipecat.services.tts_service import TTSService


class _MiniTTS(TTSService):
    """Minimal concrete TTSService for exercising base-class finalization logic."""

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        if False:  # pragma: no cover - never yields; we drive contexts directly
            yield None


class TestTeardownFinalization(unittest.IsolatedAsyncioTestCase):
    def _svc(self):
        svc = _MiniTTS(sample_rate=16000, stop_frame_timeout_s=8.0)
        svc._teardown_stop_frame_timeout_s = 1.0
        return svc

    def test_full_timeout_when_not_tearing_down(self):
        svc = self._svc()
        self.assertEqual(svc._audio_context_timeout(audio_received=False), 8.0)
        self.assertEqual(svc._audio_context_timeout(audio_received=True), 8.0)

    def test_ttfb_guard_preserved_before_first_audio(self):
        # Tearing down but no audio yet → keep the full timeout so a slow first
        # chunk (TTFB spike) isn't cut off. This is the silent-greeting guard.
        svc = self._svc()
        svc._tearing_down = True
        self.assertEqual(svc._audio_context_timeout(audio_received=False), 8.0)

    def test_short_timeout_when_tearing_down_after_audio(self):
        svc = self._svc()
        svc._tearing_down = True
        self.assertEqual(svc._audio_context_timeout(audio_received=True), 1.0)

    def test_short_timeout_never_exceeds_configured_full(self):
        # If someone configures a full timeout below the teardown value, the
        # teardown timeout must not lengthen finalization.
        svc = _MiniTTS(sample_rate=16000, stop_frame_timeout_s=0.5)
        svc._tearing_down = True
        self.assertEqual(svc._audio_context_timeout(audio_received=True), 0.5)

    async def test_begin_teardown_sets_flag_and_nudges_active_context(self):
        svc = self._svc()
        svc._playing_context_id = "ctx"
        svc._audio_contexts["ctx"] = asyncio.Queue()

        await svc._begin_teardown_finalization()

        self.assertTrue(svc._tearing_down)
        self.assertEqual(
            svc._audio_contexts["ctx"].get_nowait(), TTSService._CONTEXT_KEEPALIVE
        )

    async def test_begin_teardown_without_active_context_is_safe(self):
        svc = self._svc()
        svc._playing_context_id = None
        # Should not raise even with no active context.
        await svc._begin_teardown_finalization()
        self.assertTrue(svc._tearing_down)


if __name__ == "__main__":
    unittest.main()
