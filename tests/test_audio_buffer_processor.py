#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import struct
import unittest
import unittest.mock

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class _PassthroughResampler:
    async def resample(
        self, audio: bytes, in_rate: int, out_rate: int
    ) -> bytes:  # pragma: no cover - trivial
        return audio


async def _make_processor(*, buffer_size: int = 0) -> AudioBufferProcessor:
    """Create and start a processor ready to record.

    Calls setup() and sends a StartFrame through the public process_frame path so that
    the processor is fully initialised (task manager set, sample rate configured,
    __started flag set) without needing a full pipeline.
    """
    processor = AudioBufferProcessor(sample_rate=16000, num_channels=2, buffer_size=buffer_size)
    processor._input_resampler = _PassthroughResampler()
    processor._output_resampler = _PassthroughResampler()

    loop = asyncio.get_event_loop()
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=loop))
    await processor.setup(FrameProcessorSetup(clock=SystemClock(), task_manager=task_manager))

    await processor.process_frame(
        StartFrame(audio_out_sample_rate=16000), FrameDirection.DOWNSTREAM
    )
    await processor.start_recording()
    return processor


async def _capture_track_audio(processor: AudioBufferProcessor) -> tuple[bytes, bytes]:
    """Flush the processor and return (user_track, bot_track) from on_track_audio_data."""
    captured = {}
    event = asyncio.Event()

    async def on_track_audio_data(_, user, bot, sample_rate, num_channels):
        captured["user"] = user
        captured["bot"] = bot
        event.set()

    processor.add_event_handler("on_track_audio_data", on_track_audio_data)
    await processor.stop_recording()
    await asyncio.wait_for(event.wait(), timeout=1)
    return captured["user"], captured["bot"]


class TestAudioBufferProcessor(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.processor = await _make_processor(buffer_size=4)

    async def asyncTearDown(self):
        if getattr(self.processor, "_recording", False):
            await self.processor.stop_recording()
        await self.processor.cleanup()

    async def test_flush_user_audio_pads_bot_track(self):
        user_audio = struct.pack("<hh", 1000, -1000)
        audio_event = asyncio.Event()
        track_event = asyncio.Event()
        captured = {}

        async def on_audio_data(_, audio: bytes, sample_rate: int, num_channels: int):
            captured["merged"] = (audio, sample_rate, num_channels)
            audio_event.set()

        async def on_track_audio_data(
            _, user: bytes, bot: bytes, sample_rate: int, num_channels: int
        ):
            captured["tracks"] = (user, bot, sample_rate, num_channels)
            track_event.set()

        self.processor.add_event_handler("on_audio_data", on_audio_data)
        self.processor.add_event_handler("on_track_audio_data", on_track_audio_data)

        frame = InputAudioRawFrame(audio=user_audio, sample_rate=16000, num_channels=1)
        await self.processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        await asyncio.wait_for(audio_event.wait(), timeout=1)
        await asyncio.wait_for(track_event.wait(), timeout=1)

        merged_audio, merged_sr, merged_channels = captured["merged"]
        user_track, bot_track, track_sr, track_channels = captured["tracks"]

        self.assertEqual(merged_sr, 16000)
        self.assertEqual(merged_channels, 2)
        self.assertEqual(track_sr, 16000)
        self.assertEqual(track_channels, 2)
        self.assertEqual(user_track, user_audio)
        self.assertEqual(bot_track, b"\x00" * len(user_audio))
        self.assertEqual(len(merged_audio), len(user_audio) * 2)
        self.assertEqual(merged_audio[0:2], user_audio[0:2])
        self.assertEqual(merged_audio[2:4], b"\x00\x00")
        self.assertEqual(merged_audio[4:6], user_audio[2:4])
        self.assertEqual(merged_audio[6:8], b"\x00\x00")
        self.assertEqual(len(self.processor._user_audio_buffer), 0)
        self.assertEqual(len(self.processor._bot_audio_buffer), 0)

    async def test_flush_bot_audio_pads_user_track(self):
        bot_audio = struct.pack("<hh", -800, 400)
        audio_event = asyncio.Event()
        track_event = asyncio.Event()
        captured = {}

        async def on_audio_data(_, audio: bytes, sample_rate: int, num_channels: int):
            captured["merged"] = (audio, sample_rate, num_channels)
            audio_event.set()

        async def on_track_audio_data(
            _, user: bytes, bot: bytes, sample_rate: int, num_channels: int
        ):
            captured["tracks"] = (user, bot, sample_rate, num_channels)
            track_event.set()

        self.processor.add_event_handler("on_audio_data", on_audio_data)
        self.processor.add_event_handler("on_track_audio_data", on_track_audio_data)

        frame = OutputAudioRawFrame(audio=bot_audio, sample_rate=16000, num_channels=1)
        await self.processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        await asyncio.wait_for(audio_event.wait(), timeout=1)
        await asyncio.wait_for(track_event.wait(), timeout=1)

        merged_audio, merged_sr, merged_channels = captured["merged"]
        user_track, bot_track, track_sr, track_channels = captured["tracks"]

        self.assertEqual(merged_sr, 16000)
        self.assertEqual(merged_channels, 2)
        self.assertEqual(track_sr, 16000)
        self.assertEqual(track_channels, 2)
        self.assertEqual(user_track, b"\x00" * len(bot_audio))
        self.assertEqual(bot_track, bot_audio)
        self.assertEqual(len(merged_audio), len(bot_audio) * 2)
        self.assertEqual(merged_audio[0:2], b"\x00\x00")
        self.assertEqual(merged_audio[2:4], bot_audio[0:2])
        self.assertEqual(merged_audio[4:6], b"\x00\x00")
        self.assertEqual(merged_audio[6:8], bot_audio[2:4])
        self.assertEqual(len(self.processor._user_audio_buffer), 0)
        self.assertEqual(len(self.processor._bot_audio_buffer), 0)


class TestSilenceInjectionGuards(unittest.IsolatedAsyncioTestCase):
    """Tests that silence is not injected mid-utterance (fix for crackling artifacts).

    Each test verifies the audio alignment in the flushed tracks to confirm that
    silence is only added by _align_track_buffers at flush time (end of the buffer),
    never injected mid-stream while the affected track is actively producing audio.
    """

    async def test_no_silence_injected_into_bot_buffer_while_bot_speaking(self):
        """Bot audio must appear at the start of the bot track, not after mid-stream silence.

        Timeline:
          1. User sends 4 bytes  (bot not speaking → normal sync, no-op since bot is at 0)
          2. Bot starts speaking
          3. User sends 4 more bytes  (bot speaking → sync skipped; bot stays at 0)
          4. Bot sends 4 bytes of known audio

        Expected final bot track (8 bytes total after _align_track_buffers at flush):
          [bot_audio][silence_padding]  ← audio first, silence only at the end

        With the bug the bot track would be:
          [silence_injected_mid_stream][bot_audio]  ← silence inserted before the audio
        """
        p = await _make_processor()

        bot_audio = b"\xaa\xbb\xcc\xdd"

        await p.process_frame(
            InputAudioRawFrame(audio=b"\x01\x02\x03\x04", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await p.process_frame(
            InputAudioRawFrame(audio=b"\x05\x06\x07\x08", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(
            OutputAudioRawFrame(audio=bot_audio, sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

        _, bot_track = await _capture_track_audio(p)
        await p.cleanup()

        # Audio must appear at the beginning of the bot track (not after injected silence).
        self.assertEqual(bot_track[:4], bot_audio)
        self.assertEqual(bot_track[4:], b"\x00" * 4)

    async def test_no_silence_injected_into_user_buffer_while_user_speaking(self):
        """User audio must appear at the start of the user track, not after mid-stream silence.

        Timeline:
          1. Bot sends 4 bytes  (user not speaking → normal sync, no-op since user is at 0)
          2. User starts speaking
          3. Bot sends 4 more bytes  (user speaking → sync skipped; user stays at 0)
          4. User sends 4 bytes of known audio

        Expected final user track (8 bytes total after _align_track_buffers at flush):
          [user_audio][silence_padding]  ← audio first, silence only at the end

        With the bug the user track would be:
          [silence_injected_mid_stream][user_audio]
        """
        p = await _make_processor()

        user_audio = b"\xaa\xbb\xcc\xdd"

        await p.process_frame(
            OutputAudioRawFrame(audio=b"\x01\x02\x03\x04", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await p.process_frame(
            OutputAudioRawFrame(audio=b"\x05\x06\x07\x08", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(
            InputAudioRawFrame(audio=user_audio, sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

        user_track, _ = await _capture_track_audio(p)
        await p.cleanup()

        self.assertEqual(user_track[:4], user_audio)
        self.assertEqual(user_track[4:], b"\x00" * 4)

    async def test_silence_resumes_into_bot_buffer_after_bot_stops_speaking(self):
        """After bot stops speaking, the bot buffer is synced again on user audio arrival.

        Timeline:
          1. User sends 4 bytes  (user=4, bot=0)
          2. Bot starts speaking
          3. User sends 4 more bytes  (sync skipped; user=8, bot=0)
          4. Bot stops speaking
          5. User sends 4 more bytes  (sync resumes; bot gets 8 bytes silence, user=12)

        Expected final bot track (12 bytes): 8 bytes silence then no more audio (bot never
        sent audio, _align_track_buffers pads bot to 12).
        The key assertion: bot has 8 bytes of silence at positions 0-7, confirming that
        the sync at step 5 did inject 8 bytes (positions 0-7 of the bot buffer).
        """
        p = await _make_processor()

        await p.process_frame(
            InputAudioRawFrame(audio=b"\x01\x02\x03\x04", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await p.process_frame(
            InputAudioRawFrame(audio=b"\x05\x06\x07\x08", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await p.process_frame(
            InputAudioRawFrame(audio=b"\x09\x0a\x0b\x0c", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

        _, bot_track = await _capture_track_audio(p)
        await p.cleanup()

        # The sync at step 5 targets len(user)=8, so bot must have 8 bytes of silence
        # written before user's third chunk was added.
        self.assertEqual(bot_track[:8], b"\x00" * 8)

    async def test_silence_resumes_into_user_buffer_after_user_stops_speaking(self):
        """After user stops speaking, the user buffer is synced again on bot audio arrival.

        Timeline:
          1. Bot sends 4 bytes  (user=0, bot=4)
          2. User starts speaking
          3. Bot sends 4 more bytes  (sync skipped; user=0, bot=8)
          4. User stops speaking
          5. Bot sends 4 more bytes  (sync resumes; user gets 8 bytes silence, bot=12)

        Expected: user track has 8 bytes of silence at positions 0-7.
        """
        p = await _make_processor()

        await p.process_frame(
            OutputAudioRawFrame(audio=b"\x01\x02\x03\x04", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await p.process_frame(
            OutputAudioRawFrame(audio=b"\x05\x06\x07\x08", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await p.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await p.process_frame(
            OutputAudioRawFrame(audio=b"\x09\x0a\x0b\x0c", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )

        user_track, _ = await _capture_track_audio(p)
        await p.cleanup()

        self.assertEqual(user_track[:8], b"\x00" * 8)


class _FakeMonotonic:
    """Controllable replacement for time.monotonic()."""

    def __init__(self, start: float = 1000.0):
        self.value = start

    def __call__(self) -> float:
        return self.value


class TestTimelineRecording(unittest.IsolatedAsyncioTestCase):
    """Clock-anchored recording: inbound placed by carrier timestamp, outbound by
    monotonic arrival. This decouples the recording from frame-arrival cadence so
    bursty delivery after an event-loop stall can no longer chop tracks with
    injected silence.

    All assertions use 16kHz/16-bit mono tracks: 1 second = 32000 bytes, so
    20ms = 640 bytes.
    """

    SR = 16000  # bytes per 20ms = SR * 0.02 * 2 = 640

    async def _make_timeline_processor(self, clock: _FakeMonotonic, *, buffer_size: int = 0):
        # Patch the module clock before start_recording so _rec_start_mono is the
        # fake origin, and keep it patched for the lifetime of the returned
        # processor (caller stays inside the patch context).
        processor = await _make_processor(buffer_size=buffer_size)
        # _make_processor already called start_recording() under the patch, but to
        # be explicit and robust set the recording origin to the fake start.
        processor._rec_start_mono = clock.value
        return processor

    async def test_inbound_placed_by_carrier_timestamp_with_gap_as_silence(self):
        """Burst-delivered user frames land at spaced positions per carrier ts, and a
        real gap in carrier timestamps becomes exactly that much silence."""
        clock = _FakeMonotonic(1000.0)
        with unittest.mock.patch(
            "pipecat.processors.audio.audio_buffer_processor.time.monotonic", new=clock
        ):
            p = await self._make_timeline_processor(clock)

            f1 = b"\x11" * 640  # carrier 0.00 - 0.02
            f2 = b"\x22" * 640  # carrier 0.02 - 0.04
            f3 = b"\x33" * 640  # carrier 0.06 - 0.08 (20ms gap after f2)

            # All three arrive in the same instant (a burst after a stall):
            # the fake clock never advances.
            for audio, ts in ((f1, 0.00), (f2, 0.02), (f3, 0.06)):
                frame = InputAudioRawFrame(audio=audio, sample_rate=self.SR, num_channels=1)
                frame.metadata["recording_ts"] = ts
                await p.process_frame(frame, FrameDirection.DOWNSTREAM)

            self.assertTrue(p._timeline_enabled)
            user_track, _ = await _capture_track_audio(p)
            await p.cleanup()

        # f1, f2 contiguous; 640 bytes (20ms) of silence for the carrier gap; then f3.
        self.assertEqual(user_track, f1 + f2 + (b"\x00" * 640) + f3)

    async def test_outbound_placed_by_monotonic_arrival(self):
        """Bot frames are positioned by monotonic arrival relative to recording start,
        so a real-time gap becomes silence and consecutive frames stay contiguous."""
        clock = _FakeMonotonic(1000.0)
        with unittest.mock.patch(
            "pipecat.processors.audio.audio_buffer_processor.time.monotonic", new=clock
        ):
            p = await self._make_timeline_processor(clock)

            # Latch timeline mode with a timestamped inbound frame at t=0.
            latch = InputAudioRawFrame(audio=b"\x01" * 640, sample_rate=self.SR, num_channels=1)
            latch.metadata["recording_ts"] = 0.0
            await p.process_frame(latch, FrameDirection.DOWNSTREAM)

            b1 = b"\xaa" * 640  # arrives at t=0.04 (40ms after start)
            clock.value = 1000.04
            await p.process_frame(
                OutputAudioRawFrame(audio=b1, sample_rate=self.SR, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )
            b2 = b"\xbb" * 640  # arrives at t=0.06 (contiguous with b1)
            clock.value = 1000.06
            await p.process_frame(
                OutputAudioRawFrame(audio=b2, sample_rate=self.SR, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )

            _, bot_track = await _capture_track_audio(p)
            await p.cleanup()

        # 40ms (1280 bytes) leading silence (bot silent until t=0.04), then b1, b2 back-to-back.
        self.assertEqual(bot_track, (b"\x00" * 1280) + b1 + b2)

    async def test_legacy_mode_unchanged_without_timestamps(self):
        """Inbound frames with no carrier timestamp keep the legacy append behavior:
        burst frames are concatenated (no timeline gaps), timeline stays disabled."""
        p = await _make_processor(buffer_size=0)

        f1 = b"\x11" * 640
        f2 = b"\x22" * 640
        f3 = b"\x33" * 640
        for audio in (f1, f2, f3):
            await p.process_frame(
                InputAudioRawFrame(audio=audio, sample_rate=self.SR, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )

        self.assertFalse(p._timeline_enabled)
        user_track, _ = await _capture_track_audio(p)
        await p.cleanup()

        # Legacy: contiguous append, no carrier-driven gap silence.
        self.assertEqual(user_track, f1 + f2 + f3)


if __name__ == "__main__":
    unittest.main()
