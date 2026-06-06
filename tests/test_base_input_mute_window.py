#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the post-bot-stop mute window in BaseInputTransport.

The mute window drops inbound mic frames for a short period after the bot stops
speaking, to stop the bot's own echo from triggering a false VAD turn-start
(see log_review_findings.md, Cases 4/5). It must only be armed when the bot
stops *naturally* — on a barge-in the user is already mid-utterance, so arming
the window would drop ~600ms of live user speech and corrupt the transcript.
"""

import asyncio
import time
import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    InterruptionFrame,
)
from pipecat.tests.utils import run_test
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams


def _make_transport() -> BaseInputTransport:
    return BaseInputTransport(TransportParams(audio_in_enabled=True))


class TestPostBotStopMuteWindow(unittest.IsolatedAsyncioTestCase):
    async def test_natural_stop_arms_window(self):
        """Bot finishing its own turn arms the echo-suppression window."""
        transport = _make_transport()

        await run_test(
            transport,
            frames_to_send=[BotStartedSpeakingFrame(), BotStoppedSpeakingFrame()],
            expected_down_frames=[BotStartedSpeakingFrame, BotStoppedSpeakingFrame],
        )

        self.assertGreater(transport._post_bot_stop_mute_until_ts, 0.0)
        self.assertFalse(transport._interrupted_since_bot_start)

    async def test_bargein_stop_does_not_arm_window(self):
        """A barge-in (InterruptionFrame before stop) must NOT arm the window."""
        transport = _make_transport()

        await run_test(
            transport,
            frames_to_send=[
                BotStartedSpeakingFrame(),
                InterruptionFrame(),
                BotStoppedSpeakingFrame(),
            ],
            expected_down_frames=[
                BotStartedSpeakingFrame,
                InterruptionFrame,
                BotStoppedSpeakingFrame,
            ],
        )

        self.assertEqual(transport._post_bot_stop_mute_until_ts, 0.0)
        self.assertTrue(transport._interrupted_since_bot_start)

    async def test_interruption_disarms_already_armed_window(self):
        """An interruption arriving after the window is armed disarms it.

        Defensive: covers the case where BotStoppedSpeakingFrame is processed
        before the InterruptionFrame reaches this head transport.
        """
        transport = _make_transport()

        await run_test(
            transport,
            frames_to_send=[
                BotStartedSpeakingFrame(),
                BotStoppedSpeakingFrame(),
                InterruptionFrame(),
            ],
            expected_down_frames=[
                BotStartedSpeakingFrame,
                BotStoppedSpeakingFrame,
                InterruptionFrame,
            ],
        )

        self.assertEqual(transport._post_bot_stop_mute_until_ts, 0.0)
        self.assertTrue(transport._interrupted_since_bot_start)

    async def test_new_bot_turn_clears_interruption_flag(self):
        """A fresh bot turn forgets the prior turn's interruption and re-arms."""
        transport = _make_transport()

        await run_test(
            transport,
            frames_to_send=[
                BotStartedSpeakingFrame(),
                InterruptionFrame(),
                BotStoppedSpeakingFrame(),  # barge-in: not armed
                BotStartedSpeakingFrame(),  # new turn: clears flag
                BotStoppedSpeakingFrame(),  # natural stop: armed
            ],
        )

        self.assertFalse(transport._interrupted_since_bot_start)
        self.assertGreater(transport._post_bot_stop_mute_until_ts, 0.0)

    async def test_push_audio_frame_drops_while_armed_passes_when_disarmed(self):
        """The armed window drops mic frames; a disarmed window lets them through."""
        transport = _make_transport()
        transport._audio_in_queue = asyncio.Queue()
        audio = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)

        # Armed window: frame is dropped (never enqueued for STT/recording).
        transport._post_bot_stop_mute_until_ts = time.monotonic() + 10.0
        await transport.push_audio_frame(audio)
        self.assertEqual(transport._audio_in_queue.qsize(), 0)

        # Disarmed (barge-in cleared it): frame passes through.
        transport._post_bot_stop_mute_until_ts = 0.0
        await transport.push_audio_frame(audio)
        self.assertEqual(transport._audio_in_queue.qsize(), 1)


if __name__ == "__main__":
    unittest.main()
