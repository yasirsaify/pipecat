#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SonioxTTSService.

These cover the wire protocol rather than the network: a fake websocket records
what the service sends and replays what Soniox would send back.
"""

import base64
import json

import pytest

from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame
from pipecat.services.soniox.tts import (
    SONIOX_MAX_CONCURRENT_STREAMS,
    SonioxTTSService,
    SonioxTTSSettings,
    nearest_supported_sample_rate,
)
from pipecat.transcriptions.language import Language


class FakeWebsocket:
    """Records outgoing messages and replays a scripted inbound stream."""

    def __init__(self, inbound=None):
        self.sent = []
        self.closed = False
        self._inbound = list(inbound or [])

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def close(self):
        self.closed = True

    def __aiter__(self):
        async def gen():
            for msg in self._inbound:
                yield json.dumps(msg)

        return gen()

    @property
    def state(self):
        from websockets.protocol import State

        return State.OPEN

    def sent_of_kind(self, key):
        return [m for m in self.sent if key in m]


def _make_service(**kwargs):
    service = SonioxTTSService(api_key="test-key", **kwargs)
    service._output_sample_rate = 8000
    return service


async def _drain(agen):
    return [frame async for frame in agen]


def test_nearest_supported_sample_rate():
    """Telephony and pipeline rates map onto rates Soniox actually accepts."""
    assert nearest_supported_sample_rate(8000) == 8000
    assert nearest_supported_sample_rate(16000) == 16000
    assert nearest_supported_sample_rate(24000) == 24000
    # 22050 is a common pipeline rate Soniox rejects outright.
    assert nearest_supported_sample_rate(22050) == 24000


def test_config_message_shape():
    """The config message carries every field Soniox requires, and no None."""
    service = _make_service(
        settings=SonioxTTSSettings(model="tts-rt-v2", voice="Mina", language=Language.HI, speed=1.2)
    )
    msg = json.loads(service._build_config_msg("ctx-1"))

    assert msg == {
        "api_key": "test-key",
        "stream_id": "ctx-1",
        "model": "tts-rt-v2",
        "voice": "Mina",
        "audio_format": "pcm_s16le",
        "sample_rate": 8000,
        "language": "hi",
        "speed": 1.2,
    }


def test_config_message_omits_unset_optionals():
    """Optional controls are omitted rather than sent as null.

    Soniox validates the config message strictly; a null `speed` is a 400 that
    kills the stream before any audio is produced.
    """
    service = _make_service()
    msg = json.loads(service._build_config_msg("ctx-1"))

    assert "speed" not in msg
    assert "reduce_silence" not in msg
    assert "client_reference_id" not in msg


@pytest.mark.asyncio
async def test_run_tts_sends_config_once_then_text_only(monkeypatch):
    """A stream is configured on its first chunk and only appended to after.

    Soniox rejects a second config message for a live stream_id, so re-sending
    it on every token would break token-streaming mode entirely.
    """
    service = _make_service()
    ws = FakeWebsocket()
    service._websocket = ws

    await _drain(service.run_tts("Hello", context_id="ctx-1"))
    await _drain(service.run_tts(" there", context_id="ctx-1"))

    assert len(ws.sent_of_kind("api_key")) == 1
    assert ws.sent[0]["stream_id"] == "ctx-1"
    assert ws.sent[1] == {"stream_id": "ctx-1", "text": "Hello", "text_end": False}
    assert ws.sent[2] == {"stream_id": "ctx-1", "text": " there", "text_end": False}


@pytest.mark.asyncio
async def test_run_tts_configures_each_new_context(monkeypatch):
    """Each audio context gets its own Soniox stream."""
    service = _make_service()
    ws = FakeWebsocket()
    service._websocket = ws

    await _drain(service.run_tts("one", context_id="ctx-1"))
    await _drain(service.run_tts("two", context_id="ctx-2"))

    configs = ws.sent_of_kind("api_key")
    assert [c["stream_id"] for c in configs] == ["ctx-1", "ctx-2"]


@pytest.mark.asyncio
async def test_flush_audio_ends_the_stream():
    """flush_audio closes the stream with text_end, which is what releases the tail audio."""
    service = _make_service()
    ws = FakeWebsocket()
    service._websocket = ws

    await _drain(service.run_tts("Hello", context_id="ctx-1"))
    ws.sent.clear()

    await service.flush_audio(context_id="ctx-1")

    assert ws.sent == [{"stream_id": "ctx-1", "text": "", "text_end": True}]


@pytest.mark.asyncio
async def test_flush_audio_skips_unopened_context():
    """A context that never reached run_tts has no Soniox stream to close.

    Sending text_end for an unknown stream_id is an error response, which would
    tear down the turn for no reason.
    """
    service = _make_service()
    ws = FakeWebsocket()
    service._websocket = ws

    await service.flush_audio(context_id="never-opened")

    assert ws.sent == []


@pytest.mark.asyncio
async def test_interruption_cancels_the_stream():
    """An interruption cancels the Soniox stream so it stops billing and speaking."""
    service = _make_service()
    ws = FakeWebsocket()
    service._websocket = ws

    await _drain(service.run_tts("Hello", context_id="ctx-1"))
    ws.sent.clear()

    await service.on_audio_context_interrupted("ctx-1")

    assert ws.sent == [{"stream_id": "ctx-1", "cancel": True}]
    # The stream is gone; a later chunk on the same id must reconfigure.
    assert "ctx-1" not in service._configured_streams


@pytest.mark.asyncio
async def test_process_messages_emits_audio_and_closes_context(monkeypatch):
    """Base64 audio becomes frames; audio_end closes the context."""
    audio = b"\x01\x02\x03\x04"
    ws = FakeWebsocket(
        inbound=[
            {"stream_id": "ctx-1", "audio": base64.b64encode(audio).decode(), "audio_end": False},
            {"stream_id": "ctx-1", "audio": "", "audio_end": True},
        ]
    )
    service = _make_service()
    service._websocket = ws
    service._configured_streams.add("ctx-1")

    appended = []
    removed = []

    monkeypatch.setattr(service, "audio_context_available", lambda ctx: ctx == "ctx-1")

    async def fake_append(ctx, frame):
        appended.append((ctx, frame))

    async def fake_remove(ctx):
        removed.append(ctx)

    async def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(service, "append_to_audio_context", fake_append)
    monkeypatch.setattr(service, "remove_audio_context", fake_remove)
    monkeypatch.setattr(service, "stop_ttfb_metrics", noop)

    await service._process_messages()

    audio_frames = [f for _, f in appended if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) == 1
    assert audio_frames[0].audio == audio
    assert audio_frames[0].sample_rate == 8000

    assert any(isinstance(f, TTSStoppedFrame) for _, f in appended)
    assert removed == ["ctx-1"]
    assert "ctx-1" not in service._configured_streams


@pytest.mark.asyncio
async def test_terminated_after_audio_end_does_not_close_twice(monkeypatch):
    """A normal stream ends twice over — `audio_end`, then `terminated`.

    `remove_audio_context` only *marks* a context for deletion, so it still
    reports as available; without a guard the second message would queue a
    duplicate TTSStoppedFrame behind the deletion marker.
    """
    ws = FakeWebsocket(
        inbound=[
            {"stream_id": "ctx-1", "audio": "", "audio_end": True},
            {"stream_id": "ctx-1", "terminated": True},
        ]
    )
    service = _make_service()
    service._websocket = ws
    service._configured_streams.add("ctx-1")

    appended = []
    removed = []

    # Mirrors the base class: the context stays "available" after removal until
    # the audio-context task drains the deletion marker.
    monkeypatch.setattr(service, "audio_context_available", lambda ctx: True)

    async def fake_append(ctx, frame):
        appended.append((ctx, frame))

    async def fake_remove(ctx):
        removed.append(ctx)

    async def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(service, "append_to_audio_context", fake_append)
    monkeypatch.setattr(service, "remove_audio_context", fake_remove)
    monkeypatch.setattr(service, "stop_ttfb_metrics", noop)

    await service._process_messages()

    stopped = [f for _, f in appended if isinstance(f, TTSStoppedFrame)]
    assert len(stopped) == 1
    assert removed == ["ctx-1"]


@pytest.mark.asyncio
async def test_terminated_after_a_cancelled_stream_is_ignored(monkeypatch):
    """An interruption already tore the context down; `terminated` must not
    queue a TTSStoppedFrame into the next turn's context."""
    ws = FakeWebsocket(inbound=[{"stream_id": "ctx-1", "terminated": True}])
    service = _make_service()
    service._websocket = ws  # note: ctx-1 deliberately not in _configured_streams

    appended = []
    monkeypatch.setattr(service, "audio_context_available", lambda ctx: True)

    async def fake_append(ctx, frame):
        appended.append((ctx, frame))

    monkeypatch.setattr(service, "append_to_audio_context", fake_append)

    await service._process_messages()

    assert appended == []


@pytest.mark.asyncio
async def test_process_messages_ignores_empty_audio(monkeypatch):
    """An empty audio payload must not become a zero-length frame."""
    ws = FakeWebsocket(inbound=[{"stream_id": "ctx-1", "audio": "", "audio_end": False}])
    service = _make_service()
    service._websocket = ws

    appended = []
    monkeypatch.setattr(service, "audio_context_available", lambda ctx: True)

    async def fake_append(ctx, frame):
        appended.append((ctx, frame))

    monkeypatch.setattr(service, "append_to_audio_context", fake_append)

    await service._process_messages()

    assert appended == []


@pytest.mark.asyncio
async def test_process_messages_surfaces_errors(monkeypatch):
    """A stream-scoped error is reported and does not leave the stream marked open."""
    ws = FakeWebsocket(
        inbound=[
            {
                "stream_id": "ctx-1",
                "error_code": 400,
                "error_type": "invalid_request",
                "error_message": "Missing model",
            }
        ]
    )
    service = _make_service()
    service._websocket = ws
    service._configured_streams.add("ctx-1")

    errors = []
    appended = []
    removed = []

    async def fake_push_error(error_msg, **kwargs):
        errors.append(error_msg)

    async def fake_append(ctx, frame):
        appended.append((ctx, frame))

    async def fake_remove(ctx):
        removed.append(ctx)

    async def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(service, "push_error", fake_push_error)
    monkeypatch.setattr(service, "append_to_audio_context", fake_append)
    monkeypatch.setattr(service, "remove_audio_context", fake_remove)
    monkeypatch.setattr(service, "audio_context_available", lambda ctx: True)
    monkeypatch.setattr(service, "stop_all_metrics", noop)

    await service._process_messages()

    assert len(errors) == 1
    assert "invalid_request" in errors[0]
    assert "Missing model" in errors[0]
    assert any(isinstance(f, TTSStoppedFrame) for _, f in appended)
    assert removed == ["ctx-1"]
    assert "ctx-1" not in service._configured_streams


@pytest.mark.asyncio
async def test_orphaned_streams_are_cancelled_not_reconnected(monkeypatch):
    """A context finalized by the base class's TTFB/idle timeout never reaches
    flush_audio, so its Soniox stream is never closed and stays metered for the
    rest of the call. Opening the next stream sweeps those away.

    Reconnecting to shed them (the obvious alternative) would run
    remove_active_audio_context and take the live utterance down with it.
    """
    service = _make_service()
    ws = FakeWebsocket()
    service._websocket = ws
    service._configured_streams.update({"dead-1", "dead-2", "live-1"})

    monkeypatch.setattr(service, "audio_context_available", lambda ctx: ctx == "live-1")

    await service._ensure_stream_configured("new-1")

    cancels = sorted(m["stream_id"] for m in ws.sent if m.get("cancel"))
    assert cancels == ["dead-1", "dead-2"]
    # The live stream is untouched and the new one is configured.
    assert service._configured_streams == {"live-1", "new-1"}
    assert [m["stream_id"] for m in ws.sent_of_kind("api_key")] == ["new-1"]


@pytest.mark.asyncio
async def test_stream_cap_does_not_tear_down_live_streams(monkeypatch):
    """At the cap with every stream live there is nothing safe to shed.

    Soniox fails just the new stream rather than the connection, which arrives
    on the error path — a better outcome than cutting off whatever is speaking.
    """
    service = _make_service()
    ws = FakeWebsocket()
    service._websocket = ws
    live = {f"live-{i}" for i in range(SONIOX_MAX_CONCURRENT_STREAMS)}
    service._configured_streams.update(live)

    monkeypatch.setattr(service, "audio_context_available", lambda ctx: True)

    disconnected = []

    async def fake_disconnect():
        disconnected.append(True)

    monkeypatch.setattr(service, "_disconnect_websocket", fake_disconnect)

    await service._ensure_stream_configured("new-1")

    assert disconnected == [], "must not reconnect: it would kill the live utterance"
    assert ws.sent_of_kind("cancel") == []
    assert service._configured_streams == live | {"new-1"}


@pytest.mark.asyncio
async def test_disconnect_clears_configured_streams():
    """A new socket knows nothing about old streams, so config must be re-sent."""
    service = _make_service()
    ws = FakeWebsocket()
    service._websocket = ws
    service._configured_streams.add("ctx-1")

    async def noop(*args, **kwargs):
        pass

    service.stop_all_metrics = noop
    service.remove_active_audio_context = noop
    service._call_event_handler = noop

    await service._disconnect_websocket()

    assert service._configured_streams == set()
    assert ws.closed


@pytest.mark.asyncio
async def test_error_for_a_finished_stream_does_not_disturb_the_live_one(monkeypatch):
    """Soniox fails one stream without closing the connection, so a late error
    for a stream we already finished must not tear down what is speaking now.

    The earlier version reset the active audio context here, nulling the
    playback cursor of an unrelated, live context.
    """
    ws = FakeWebsocket(
        inbound=[
            {
                "stream_id": "ctx-old",
                "error_code": 400,
                "error_type": "invalid_request",
                "error_message": "too late",
            }
        ]
    )
    service = _make_service()
    service._websocket = ws
    service._configured_streams.add("ctx-live")  # ctx-old already finished

    errors = []
    appended = []
    removed = []
    resets = []

    async def fake_push_error(error_msg, **kwargs):
        errors.append(error_msg)

    async def fake_append(ctx, frame):
        appended.append((ctx, frame))

    async def fake_remove(ctx):
        removed.append(ctx)

    async def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(service, "push_error", fake_push_error)
    monkeypatch.setattr(service, "append_to_audio_context", fake_append)
    monkeypatch.setattr(service, "remove_audio_context", fake_remove)
    monkeypatch.setattr(service, "audio_context_available", lambda ctx: True)
    monkeypatch.setattr(service, "stop_all_metrics", noop)
    monkeypatch.setattr(service, "reset_active_audio_context", lambda: resets.append(True))

    await service._process_messages()

    # The error is still surfaced...
    assert len(errors) == 1
    # ...but nothing about the live stream is touched.
    assert appended == []
    assert removed == []
    assert resets == []
    assert service._configured_streams == {"ctx-live"}


def test_init_time_regional_language_is_resolved_to_a_base_code():
    """Soniox takes a bare ISO code; a regional one is rejected.

    TTSService.__init__ converts after apply_update, so settings passed at
    construction are normalised the same way a runtime update would be.
    """
    service = _make_service(settings=SonioxTTSSettings(language=Language.EN_US))
    assert service._settings.language == "en"
    assert json.loads(service._build_config_msg("ctx-1"))["language"] == "en"
