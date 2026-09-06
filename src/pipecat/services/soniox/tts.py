#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Soniox text-to-speech service implementation.

Soniox real-time TTS is a WebSocket API that multiplexes several independent
*streams* over one connection, each identified by a client-chosen ``stream_id``.
That maps directly onto Pipecat's audio contexts: one Soniox stream per audio
context, opened with a config message and closed with ``text_end``.

See https://soniox.com/docs/api-reference/tts/websocket-api.
"""

import base64
import json
from dataclasses import dataclass, field
from typing import AsyncGenerator, ClassVar, Dict, Optional, Set

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Soniox, you need to `pip install pipecat-ai[soniox]`.")
    raise Exception(f"Missing module: {e}")


SONIOX_TTS_WEBSOCKET_URL = "wss://tts-rt.soniox.com/tts-websocket"

# Sample rates the API accepts. Anything else is rejected outright, so an
# unsupported pipeline rate is snapped to the nearest of these and the audio
# frames carry that rate instead — the output transport resamples from there.
SONIOX_SUPPORTED_SAMPLE_RATES = (8000, 16000, 24000, 44100, 48000)

# A single connection hosts at most this many concurrent streams. Contexts are
# closed as they finish, so this is a guard against leaking stream ids rather
# than a limit we expect to reach.
SONIOX_MAX_CONCURRENT_STREAMS = 5


def language_to_soniox_language(language: Language) -> Optional[str]:
    """Convert a Language enum to a Soniox TTS language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Soniox language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.AF: "af",
        Language.AR: "ar",
        Language.AZ: "az",
        Language.BE: "be",
        Language.BG: "bg",
        Language.BN: "bn",
        Language.BS: "bs",
        Language.CA: "ca",
        Language.CS: "cs",
        Language.CY: "cy",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.ET: "et",
        Language.EU: "eu",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.GL: "gl",
        Language.GU: "gu",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HU: "hu",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KA: "ka",
        Language.KK: "kk",
        Language.KN: "kn",
        Language.KO: "ko",
        Language.LT: "lt",
        Language.LV: "lv",
        Language.MK: "mk",
        Language.ML: "ml",
        Language.MR: "mr",
        Language.MS: "ms",
        Language.NL: "nl",
        Language.NO: "no",
        Language.PA: "pa",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.SQ: "sq",
        Language.SR: "sr",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TE: "te",
        Language.TH: "th",
        Language.TL: "tl",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.VI: "vi",
        Language.ZH: "zh",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


def nearest_supported_sample_rate(sample_rate: int) -> int:
    """Snap a pipeline sample rate onto the closest rate Soniox accepts."""
    return min(SONIOX_SUPPORTED_SAMPLE_RATES, key=lambda rate: abs(rate - sample_rate))


@dataclass
class SonioxTTSSettings(TTSSettings):
    """Settings for SonioxTTSService.

    Parameters:
        speed: Speaking rate from 0.7 to 1.3, where 1.0 is normal speed.
        reduce_silence: Whether the model should trim long silences.
        client_reference_id: Optional client-defined identifier recorded with
            each request, for correlating a call with Soniox's own logs.
    """

    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    reduce_silence: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    client_reference_id: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[Dict[str, str]] = {"voice_id": "voice"}


class SonioxTTSService(WebsocketTTSService):
    """Soniox TTS service with WebSocket streaming.

    Text is streamed into the service incrementally, so audio for the first
    words comes back while the rest of the sentence is still being written.
    Each Pipecat audio context is one Soniox stream: the first ``run_tts`` call
    for a context sends the config message, subsequent calls only append text,
    and ``flush_audio`` closes it with ``text_end``.
    """

    Settings = SonioxTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = SONIOX_TTS_WEBSOCKET_URL,
        sample_rate: Optional[int] = None,
        audio_format: str = "pcm_s16le",
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the Soniox TTS service.

        Args:
            api_key: Soniox API key. Sent in the per-stream config message —
                Soniox does not authenticate the socket itself.
            url: WebSocket URL for the Soniox real-time TTS API.
            sample_rate: Audio sample rate. Snapped to the nearest rate Soniox
                supports; if None, the pipeline's rate is used.
            audio_format: Raw output encoding. Only PCM formats produce frames
                the pipeline can consume directly.
            settings: Runtime-updatable settings (model, voice, language, speed).
            **kwargs: Additional arguments passed to the parent service.
        """
        default_settings = self.Settings(
            model="tts-rt-v2",
            voice="Adrian",
            language=Language.EN,
            speed=None,
            reduce_silence=None,
            client_reference_id=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            pause_frame_processing=False,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url

        # Output format — init-only, not runtime-updatable.
        self._audio_format = audio_format
        self._output_sample_rate = 0  # Set in start() from self.sample_rate

        # Stream ids for which a config message has already been sent on the
        # current websocket. Cleared on disconnect: a new socket needs a new
        # config message for every stream.
        self._configured_streams: Set[str] = set()

        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as the Soniox service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to a Soniox language code.

        Args:
            language: The language to convert.

        Returns:
            The Soniox language code, or None if not supported.
        """
        return language_to_soniox_language(language)

    def _build_config_msg(self, stream_id: str) -> str:
        msg = {
            "api_key": self._api_key,
            "stream_id": stream_id,
            "model": self._settings.model,
            "voice": self._settings.voice,
            "audio_format": self._audio_format,
            "sample_rate": self._output_sample_rate,
        }

        if self._settings.language:
            msg["language"] = self._settings.language
        if self._settings.speed is not None:
            msg["speed"] = self._settings.speed
        if self._settings.reduce_silence is not None:
            msg["reduce_silence"] = self._settings.reduce_silence
        if self._settings.client_reference_id:
            msg["client_reference_id"] = self._settings.client_reference_id

        return json.dumps(msg)

    def _build_text_msg(self, stream_id: str, text: str, text_end: bool = False) -> str:
        return json.dumps({"stream_id": stream_id, "text": text, "text_end": text_end})

    async def start(self, frame: StartFrame):
        """Start the Soniox TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._output_sample_rate = nearest_supported_sample_rate(self.sample_rate)
        if self._output_sample_rate != self.sample_rate:
            logger.debug(
                f"{self}: pipeline sample rate {self.sample_rate} is not supported by "
                f"Soniox; requesting {self._output_sample_rate} instead"
            )
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Soniox TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Soniox TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Soniox TTS")
            self._websocket = await websocket_connect(self._url)
            # A fresh socket knows nothing about the streams the old one had
            # configured, so every context must send its config message again.
            self._configured_streams.clear()
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Soniox TTS")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._configured_streams.clear()
            await self.remove_active_audio_context()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _cancel_stream(self, stream_id: str):
        """Cancel one Soniox stream and forget it, tolerating a dead socket."""
        self._configured_streams.discard(stream_id)
        try:
            await self._get_websocket().send(json.dumps({"stream_id": stream_id, "cancel": True}))
        except Exception as e:
            logger.debug(f"{self}: failed to cancel Soniox stream {stream_id}: {e}")

    async def _cancel_orphaned_streams(self):
        """Cancel streams whose audio context is already gone.

        Not every context reaches ``flush_audio``: one finalized by the base
        class's TTFB or idle timeout is dropped without it, so no ``text_end``
        is ever sent, Soniox never answers ``audio_end``, and the stream stays
        open — metered for the rest of the call, and counting against the
        per-connection cap. Sweeping them here is what keeps that cap out of
        reach; reconnecting to shed them would take the live utterance down
        with them.
        """
        orphans = [
            stream_id
            for stream_id in self._configured_streams
            if not self.audio_context_available(stream_id)
        ]
        for stream_id in orphans:
            logger.debug(f"{self}: cancelling orphaned Soniox stream {stream_id}")
            await self._cancel_stream(stream_id)

    async def _ensure_stream_configured(self, stream_id: str):
        """Open the Soniox stream for this context if it isn't open already."""
        if stream_id in self._configured_streams:
            return

        await self._cancel_orphaned_streams()

        if len(self._configured_streams) >= SONIOX_MAX_CONCURRENT_STREAMS:
            # Every remaining stream still has a live context, so there is
            # nothing safe to shed. Soniox fails just this stream rather than
            # the connection, and that arrives on the error path — which is a
            # better outcome than cutting off whatever is currently speaking.
            logger.warning(
                f"{self}: {len(self._configured_streams)} Soniox streams open on this "
                f"connection (limit {SONIOX_MAX_CONCURRENT_STREAMS}); "
                f"stream {stream_id} may be rejected"
            )

        await self._get_websocket().send(self._build_config_msg(stream_id))
        self._configured_streams.add(stream_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the Soniox stream when the bot is interrupted."""
        await self.stop_all_metrics()
        if context_id and context_id in self._configured_streams:
            await self._cancel_stream(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def flush_audio(self, context_id: Optional[str] = None):
        """Close the Soniox stream so it emits the tail of the utterance.

        Soniox holds back the final audio until it is told no more text is
        coming, so this is what ends the turn rather than an optimization.

        Args:
            context_id: The specific context to flush. If None, falls back to
                the currently active context.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id or not self._websocket:
            return
        if flush_id not in self._configured_streams:
            # Nothing was ever sent for this context — there is no Soniox stream
            # to close, and a text_end for an unknown stream is an error.
            return
        logger.trace(f"{self}: flushing audio")
        await self._websocket.send(self._build_text_msg(flush_id, "", text_end=True))

    async def _process_messages(self):
        async for message in self._get_websocket():
            msg = json.loads(message)
            stream_id = msg.get("stream_id")

            if msg.get("error_code") is not None:
                await self.stop_all_metrics()
                await self.push_error(
                    error_msg=(
                        f"Soniox TTS error {msg.get('error_code')} "
                        f"({msg.get('error_type')}): {msg.get('error_message')}"
                    )
                )
                # Tear down only the stream that actually failed. Soniox fails
                # one stream without closing the connection, so an error for a
                # stream we have already finished — a late error for a cancelled
                # one, say — must not touch whatever is speaking now.
                if stream_id and stream_id in self._configured_streams:
                    self._configured_streams.discard(stream_id)
                    if self.audio_context_available(stream_id):
                        await self.append_to_audio_context(
                            stream_id, TTSStoppedFrame(context_id=stream_id)
                        )
                        await self.remove_audio_context(stream_id)
                continue

            if not stream_id or not self.audio_context_available(stream_id):
                continue

            if msg.get("audio"):
                audio = base64.b64decode(msg["audio"])
                if audio:
                    # TTFB is stopped by the base class on the first frame it
                    # dequeues for playout, so it is deliberately not stopped
                    # here: doing both would report this service's TTFB against
                    # a different clock than every other TTS service's.
                    await self.append_to_audio_context(
                        stream_id,
                        TTSAudioRawFrame(
                            audio=audio,
                            sample_rate=self._output_sample_rate,
                            num_channels=1,
                            context_id=stream_id,
                        ),
                    )

            # A normal stream ends twice over: `audio_end` on the last audio
            # chunk, then a separate `terminated` message. Only the first may
            # close the context — `remove_audio_context` marks the context for
            # deletion rather than removing it outright, so the guard above
            # still reports it as available and a second close would queue a
            # duplicate TTSStoppedFrame behind the deletion marker. The set
            # membership is the guard, and it also makes a `terminated` that
            # follows a cancelled stream a no-op.
            if (msg.get("audio_end") or msg.get("terminated")) and (
                stream_id in self._configured_streams
            ):
                self._configured_streams.discard(stream_id)
                await self.stop_ttfb_metrics()
                await self.append_to_audio_context(stream_id, TTSStoppedFrame(context_id=stream_id))
                await self.remove_audio_context(stream_id)

    async def _receive_messages(self):
        while True:
            await self._process_messages()
            logger.debug(f"{self} Soniox connection was disconnected, reconnecting")
            await self._connect_websocket()

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Soniox's streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        if not self._is_streaming_tokens:
            logger.debug(f"{self}: Generating TTS [{text}]")
        else:
            logger.trace(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                await self._ensure_stream_configured(context_id)
                await self._get_websocket().send(self._build_text_msg(context_id, text))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
