#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio buffer processor for managing and synchronizing audio streams.

This module provides an AudioBufferProcessor that handles buffering and synchronization
of audio from both user input and bot output sources, with support for various audio
configurations and event-driven processing.
"""

import time
from typing import Optional

from loguru import logger

from pipecat.audio.utils import create_stream_resampler, interleave_stereo_audio, mix_audio
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AudioBufferProcessor(FrameProcessor):
    """Processes and buffers audio frames from both input (user) and output (bot) sources.

    This processor manages audio buffering and synchronization, providing both merged and
    track-specific audio access through event handlers. It supports various audio configurations
    including sample rate conversion and mono/stereo output.

    Events:

    - on_audio_data: Triggered when buffer_size is reached, providing merged audio
    - on_track_audio_data: Triggered when buffer_size is reached, providing separate tracks
    - on_user_turn_audio_data: Triggered when user turn has ended, providing that user turn's audio
    - on_bot_turn_audio_data: Triggered when bot turn has ended, providing that bot turn's audio

    Audio handling:

    - Mono output (num_channels=1): User and bot audio are mixed
    - Stereo output (num_channels=2): User audio on left, bot audio on right
    - Automatic resampling of incoming audio to match desired sample_rate
    - Silence insertion for non-continuous audio streams
    - Buffer synchronization between user and bot audio
    """

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        num_channels: int = 1,
        buffer_size: int = 0,
        user_continuous_stream: Optional[bool] = None,
        enable_turn_audio: bool = False,
        **kwargs,
    ):
        """Initialize the audio buffer processor.

        Args:
            sample_rate: Desired output sample rate. If None, uses source rate.
            num_channels: Number of channels (1 for mono, 2 for stereo). Defaults to 1.
            buffer_size: Size of buffer before triggering events. 0 for no buffering.
            user_continuous_stream: Controls whether user audio is treated as a continuous
                stream for buffering purposes.

                .. deprecated:: 0.0.72
                    This parameter no longer has any effect and will be removed in a future version.

            enable_turn_audio: Whether turn audio event handlers should be triggered.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._audio_buffer_size_1s = 0
        self._num_channels = num_channels
        self._buffer_size = buffer_size
        self._enable_turn_audio = enable_turn_audio

        if user_continuous_stream is not None:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter `user_continuous_stream` is deprecated.",
                    DeprecationWarning,
                )

        self._user_audio_buffer = bytearray()
        self._bot_audio_buffer = bytearray()

        # --- Timeline (clock-anchored) recording state -------------------------
        # When inbound audio frames carry an authoritative capture timestamp
        # (metadata["recording_ts"], in seconds), we place each track on a real
        # time axis instead of appending by arrival order. This prevents bursty
        # frame delivery (e.g. after an event-loop stall) from chopping the bot
        # track with silence mid-utterance. Auto-enables on the first timestamped
        # inbound frame; otherwise the legacy byte-append path is used unchanged.
        self._timeline_enabled = False
        # Monotonic time captured at start_recording; the shared recording axis
        # origin for both tracks.
        self._rec_start_mono: Optional[float] = None
        # Bytes of (aligned) audio already flushed via on_audio_data. Absolute
        # timeline positions are converted to current-buffer offsets by
        # subtracting this baseline.
        self._timeline_base_bytes = 0
        # Bridge from the inbound source clock (carrier ms) to the recording
        # axis: (first_carrier_ts_seconds, axis_offset_seconds).
        self._in_ts_origin: Optional[tuple[float, float]] = None

        self._user_speaking = False
        self._bot_speaking = False
        self._user_turn_audio_buffer = bytearray()
        self._bot_turn_audio_buffer = bytearray()

        self._recording = False

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()

        self._register_event_handler("on_audio_data")
        self._register_event_handler("on_track_audio_data")
        self._register_event_handler("on_user_turn_audio_data")
        self._register_event_handler("on_bot_turn_audio_data")

    @property
    def sample_rate(self) -> int:
        """Current sample rate of the audio processor.

        Returns:
            The sample rate in Hz.
        """
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        """Number of channels in the audio output.

        Returns:
            Number of channels (1 for mono, 2 for stereo).
        """
        return self._num_channels

    def has_audio(self) -> bool:
        """Check if either user or bot audio buffers contain data.

        Returns:
            True if either buffer contains audio data.
        """
        return self._buffer_has_audio(self._user_audio_buffer) or self._buffer_has_audio(
            self._bot_audio_buffer
        )

    def merge_audio_buffers(self) -> bytes:
        """Merge user and bot audio buffers into a single audio stream.

        For mono output, audio is mixed. For stereo output, user audio is placed
        on the left channel and bot audio on the right channel.

        Returns:
            Mixed audio data as bytes.
        """
        if self._num_channels == 1:
            return mix_audio(bytes(self._user_audio_buffer), bytes(self._bot_audio_buffer))
        elif self._num_channels == 2:
            return interleave_stereo_audio(
                bytes(self._user_audio_buffer), bytes(self._bot_audio_buffer)
            )
        else:
            return b""

    async def start_recording(self):
        """Start recording audio from both user and bot.

        Initializes recording state and resets audio buffers.
        """
        self._recording = True
        self._rec_start_mono = time.monotonic()
        self._reset_recording()

    async def stop_recording(self):
        """Stop recording and trigger final audio data handlers.

        Calls audio handlers with any remaining buffered audio before stopping.
        """
        await self._call_on_audio_data_handler()
        self._reset_recording()
        self._recording = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming audio frames and manage audio buffers.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # Update output sample rate if necessary.
        if isinstance(frame, StartFrame):
            self._update_sample_rate(frame)

        if self._recording:
            await self._process_recording(frame)

        if isinstance(frame, (CancelFrame, EndFrame)):
            await self.stop_recording()

        await self.push_frame(frame, direction)

    def _update_sample_rate(self, frame: StartFrame):
        """Update the sample rate from the start frame."""
        self._sample_rate = self._init_sample_rate or frame.audio_out_sample_rate
        self._audio_buffer_size_1s = self._sample_rate * 2

    async def _process_recording(self, frame: Frame):
        """Process audio frames for recording."""
        # Track speaking state here (not just in _process_turn_recording) so the
        # silence-injection guards below work regardless of enable_turn_audio.
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False

        # Latch timeline (clock-anchored) recording on the first inbound frame
        # that carries an authoritative capture timestamp. Until then (or forever,
        # for transports that never provide one) the legacy byte-append path runs
        # unchanged.
        if (
            not self._timeline_enabled
            and isinstance(frame, InputAudioRawFrame)
            and frame.metadata.get("recording_ts") is not None
        ):
            self._timeline_enabled = True

        resampled = None
        if isinstance(frame, InputAudioRawFrame):
            resampled = await self._resample_input_audio(frame)
            # Ignoring in case we don't have audio
            if len(resampled) > 0:
                if self._timeline_enabled:
                    # Place the user chunk at its true capture time on the
                    # recording axis (immune to bursty arrival after a stall).
                    self._place_timeline_audio(
                        self._user_audio_buffer,
                        resampled,
                        self._input_timeline_position(frame),
                    )
                else:
                    # Sync bot buffer to current user position before adding user audio.
                    # We sync BEFORE extending to align both buffers at the same starting timestamp.
                    # For example, user buffer is at 100 bytes, and you receive 20 bytes of new audio
                    #  - Bot buffer sees User is at 100. Bot pads itself to 100.
                    #  - User buffer adds 20. User is now at 120.
                    #  - Outcome: At index 100-120, we have User Audio and (potentially) Bot Audio or silence. They are aligned
                    # This gives the opportunity to the bot to send audio.
                    #
                    # If we synced AFTER, we'd pad the bot buffer with silence for the same
                    # window we just gave to the user, effectively "overwriting" that time slot
                    # with silence and causing the bot's audio to flicker or cut out.
                    #
                    # Skip silence injection if the bot is actively speaking to avoid
                    # inserting silence in the middle of a bot utterance (causes crackling).
                    if not self._bot_speaking:
                        self._sync_buffer_to_position(
                            self._bot_audio_buffer, len(self._user_audio_buffer)
                        )
                    # Add user audio.
                    self._user_audio_buffer.extend(resampled)
        elif isinstance(frame, OutputAudioRawFrame):
            resampled = await self._resample_output_audio(frame)
            # Ignoring in case we don't have audio
            if len(resampled) > 0:
                if self._timeline_enabled:
                    # Output frames reach us right after the transport's
                    # real-time pacing sleep, so monotonic arrival is a faithful
                    # clock for the bot track.
                    self._place_timeline_audio(
                        self._bot_audio_buffer,
                        resampled,
                        self._output_timeline_position(),
                    )
                else:
                    # Sync user buffer to current bot position before adding bot audio.
                    # Skip silence injection if the user is actively speaking to avoid
                    # inserting silence in the middle of a user utterance (causes crackling).
                    if not self._user_speaking:
                        self._sync_buffer_to_position(
                            self._user_audio_buffer, len(self._bot_audio_buffer)
                        )
                    # Add bot audio.
                    self._bot_audio_buffer.extend(resampled)

        if self._buffer_size > 0 and (
            len(self._user_audio_buffer) >= self._buffer_size
            or len(self._bot_audio_buffer) >= self._buffer_size
        ):
            await self._call_on_audio_data_handler()
            if self._timeline_enabled:
                # Both tracks are equal length after alignment in the handler;
                # advance the flushed-bytes baseline so absolute timeline
                # positions keep mapping into the freshly reset buffers.
                self._timeline_base_bytes += len(self._user_audio_buffer)
            self._reset_primary_audio_buffers()

        # Process turn recording with preprocessed data.
        if self._enable_turn_audio:
            await self._process_turn_recording(frame, resampled)

    def _sync_buffer_to_position(self, buffer: bytearray, target_position: int):
        """Pad buffer with silence if it's behind the target position.

        This ensures both buffers stay synchronized by padding the lagging
        buffer before new audio is added to the other buffer.

        Args:
            buffer: The buffer to potentially pad.
            target_position: The position (in bytes) the buffer should reach.
        """
        current_len = len(buffer)
        if current_len < target_position:
            silence_needed = target_position - current_len
            buffer.extend(b"\x00" * silence_needed)

    def _seconds_to_byte_position(self, pos_sec: float) -> int:
        """Convert a time offset (seconds) to an absolute byte position.

        Track buffers hold mono 16-bit PCM (2 bytes/sample), so positions are
        sample-aligned (always even). Negative offsets are clamped to 0.
        """
        sample_pos = max(0, round(pos_sec * self._sample_rate))
        return sample_pos * 2

    def _input_timeline_position(self, frame: InputAudioRawFrame) -> int:
        """Absolute byte position for an inbound (user) audio chunk.

        Uses the carrier capture timestamp (metadata["recording_ts"], seconds)
        bridged to the recording axis via the first inbound frame, so spacing is
        faithful to real time regardless of arrival jitter or event-loop stalls.
        Frames without a timestamp are appended at the current position.
        """
        carrier_ts = frame.metadata.get("recording_ts")
        if carrier_ts is None:
            return self._timeline_base_bytes + len(self._user_audio_buffer)
        if self._in_ts_origin is None:
            rec_start = self._rec_start_mono if self._rec_start_mono is not None else time.monotonic()
            axis_offset = time.monotonic() - rec_start
            self._in_ts_origin = (carrier_ts, axis_offset)
        origin_ts, axis_offset = self._in_ts_origin
        pos_sec = axis_offset + (carrier_ts - origin_ts)
        return self._seconds_to_byte_position(pos_sec)

    def _output_timeline_position(self) -> int:
        """Absolute byte position for an outbound (bot) audio chunk.

        Output frames are pushed downstream immediately after the transport's
        real-time pacing sleep, so monotonic arrival relative to the recording
        start is a faithful real-time clock for the bot track.
        """
        rec_start = self._rec_start_mono if self._rec_start_mono is not None else time.monotonic()
        return self._seconds_to_byte_position(time.monotonic() - rec_start)

    def _place_timeline_audio(self, buffer: bytearray, data: bytes, abs_pos_bytes: int):
        """Place audio at an absolute timeline position in the current chunk.

        Pads with silence when the target is ahead of the buffer (a real gap in
        the stream) and never moves backwards: a frame whose timeline position
        falls before the current buffer end (rounding/jitter) is appended
        contiguously rather than overwriting existing audio.
        """
        local_pos = abs_pos_bytes - self._timeline_base_bytes
        target = max(local_pos, len(buffer))
        if len(buffer) < target:
            buffer.extend(b"\x00" * (target - len(buffer)))
        buffer.extend(data)

    async def _process_turn_recording(self, frame: Frame, resampled_audio: Optional[bytes] = None):
        """Process frames for turn-based audio recording."""
        # Speaking state (_user_speaking / _bot_speaking) is maintained by
        # _process_recording so it is always up-to-date here.
        if isinstance(frame, UserStoppedSpeakingFrame):
            await self._call_event_handler(
                "on_user_turn_audio_data", self._user_turn_audio_buffer, self.sample_rate, 1
            )
            self._user_turn_audio_buffer = bytearray()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._call_event_handler(
                "on_bot_turn_audio_data", self._bot_turn_audio_buffer, self.sample_rate, 1
            )
            self._bot_turn_audio_buffer = bytearray()

        if isinstance(frame, InputAudioRawFrame) and resampled_audio:
            self._user_turn_audio_buffer.extend(resampled_audio)
            # In the case of the user, we need to keep a short buffer of audio
            # since VAD notification of when the user starts speaking comes
            # later.
            if (
                not self._user_speaking
                and len(self._user_turn_audio_buffer) > self._audio_buffer_size_1s
            ):
                discarded = len(self._user_turn_audio_buffer) - self._audio_buffer_size_1s
                self._user_turn_audio_buffer = self._user_turn_audio_buffer[discarded:]
        elif self._bot_speaking and isinstance(frame, OutputAudioRawFrame) and resampled_audio:
            self._bot_turn_audio_buffer.extend(resampled_audio)

    async def _call_on_audio_data_handler(self):
        """Call the audio data event handlers with buffered audio."""
        if not self._recording:
            return

        if len(self._user_audio_buffer) == 0 and len(self._bot_audio_buffer) == 0:
            return

        # Final alignment before we send the audio
        self._align_track_buffers()

        # Call original handler with merged audio
        merged_audio = self.merge_audio_buffers()
        await self._call_event_handler(
            "on_audio_data", merged_audio, self._sample_rate, self._num_channels
        )

        # Call new handler with separate tracks
        await self._call_event_handler(
            "on_track_audio_data",
            bytes(self._user_audio_buffer),
            bytes(self._bot_audio_buffer),
            self._sample_rate,
            self._num_channels,
        )

    def _buffer_has_audio(self, buffer: bytearray) -> bool:
        """Check if a buffer contains audio data."""
        return buffer is not None and len(buffer) > 0

    def _reset_recording(self):
        """Reset recording state and buffers."""
        self._reset_all_audio_buffers()
        # Reset timeline bookkeeping. The enabled latch and origin are cleared so
        # a fresh recording re-derives them; _rec_start_mono is set by
        # start_recording() just before this call.
        self._timeline_enabled = False
        self._timeline_base_bytes = 0
        self._in_ts_origin = None

    def _reset_all_audio_buffers(self):
        """Reset all audio buffers to empty state."""
        self._reset_primary_audio_buffers()
        self._reset_turn_audio_buffers()

    def _reset_primary_audio_buffers(self):
        """Clear user and bot buffers while preserving turn buffers and timestamps."""
        self._user_audio_buffer = bytearray()
        self._bot_audio_buffer = bytearray()

    def _reset_turn_audio_buffers(self):
        """Clear user and bot turn buffers while preserving primary buffers and timestamps."""
        self._user_turn_audio_buffer = bytearray()
        self._bot_turn_audio_buffer = bytearray()

    def _align_track_buffers(self):
        """Pad the shorter track with silence so both tracks stay in sync."""
        user_len = len(self._user_audio_buffer)
        bot_len = len(self._bot_audio_buffer)
        if user_len == bot_len:
            return

        target_len = max(user_len, bot_len)
        if user_len < target_len:
            self._sync_buffer_to_position(self._user_audio_buffer, target_len)
        if bot_len < target_len:
            self._sync_buffer_to_position(self._bot_audio_buffer, target_len)

    async def _resample_input_audio(self, frame: InputAudioRawFrame) -> bytes:
        """Resample audio frame to the target sample rate."""
        return await self._input_resampler.resample(
            frame.audio, frame.sample_rate, self._sample_rate
        )

    async def _resample_output_audio(self, frame: OutputAudioRawFrame) -> bytes:
        """Resample audio frame to the target sample rate."""
        return await self._output_resampler.resample(
            frame.audio, frame.sample_rate, self._sample_rate
        )
