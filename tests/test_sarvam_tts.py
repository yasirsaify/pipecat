#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SarvamTTSService."""

import asyncio
from types import SimpleNamespace

import pytest
from websockets.protocol import State

from pipecat.services.sarvam.tts import SarvamTTSService


def _make_service():
    return SarvamTTSService(api_key="test-key")


async def _drain(agen):
    """Collect everything yielded by an async generator."""
    return [frame async for frame in agen]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ['"', "   ", '"  "', "।", "...", "—"])
async def test_run_tts_skips_unsynthesizable_text(monkeypatch, text):
    """Punctuation/whitespace-only chunks must not be sent to Sarvam.

    Sarvam returns "400: Text must contain at least one character from the
    allowed languages." for such fragments and tears down the websocket, so
    run_tts should skip them entirely. Sentence aggregation can isolate a
    trailing closing quote as its own chunk, which previously broke the
    end-of-call goodbye line.
    """
    service = _make_service()

    sent = []
    connected = []

    async def fake_send_text(t):
        sent.append(t)

    async def fake_connect():
        connected.append(True)

    monkeypatch.setattr(service, "_send_text", fake_send_text)
    monkeypatch.setattr(service, "_connect", fake_connect)

    frames = await _drain(service.run_tts(text, context_id="ctx-1"))

    assert sent == [], f"unsynthesizable text {text!r} should not be sent"
    assert connected == [], "should not even open a connection for skipped text"
    assert frames == [], "no frames should be produced for skipped text"


@pytest.mark.asyncio
async def test_run_tts_sends_synthesizable_text(monkeypatch):
    """Text with at least one alphanumeric character is sent for synthesis."""
    service = _make_service()

    sent = []

    async def fake_send_text(t):
        sent.append(t)

    async def fake_connect():
        return None

    async def fake_metrics(_t):
        return None

    monkeypatch.setattr(service, "_send_text", fake_send_text)
    monkeypatch.setattr(service, "_connect", fake_connect)
    monkeypatch.setattr(service, "start_tts_usage_metrics", fake_metrics)

    await _drain(service.run_tts("धन्यवाद, आपका दिन शुभ हो।", context_id="ctx-1"))

    assert sent == ["धन्यवाद, आपका दिन शुभ हो।"]


# --------------------------------------------------------------------------- #
# Background connect (start() schedules connect; run_tts/teardown gate on it)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_ensure_connected_awaits_and_clears_task():
    """_ensure_connected awaits the pending connect once, then is a no-op."""
    service = _make_service()
    connected = []

    async def fake_connect():
        connected.append(True)

    service._start_connect_task = asyncio.ensure_future(fake_connect())

    await service._ensure_connected()
    assert connected == [True]
    assert service._start_connect_task is None

    # Second call must not blow up or re-await anything.
    await service._ensure_connected()
    assert connected == [True]


@pytest.mark.asyncio
async def test_run_tts_awaits_background_connect_before_sending(monkeypatch):
    """The first run_tts must wait for the start()-scheduled connect before
    sending text, so the utterance never races the handshake."""
    service = _make_service()
    order = []

    async def fake_connect():
        # Simulate handshake latency so a non-gated send would interleave.
        await asyncio.sleep(0.01)
        order.append("connected")

    async def fake_send_text(t):
        order.append("sent")

    async def fake_metrics(_t):
        return None

    monkeypatch.setattr(service, "_send_text", fake_send_text)
    monkeypatch.setattr(service, "start_tts_usage_metrics", fake_metrics)
    # Pretend the socket is already open once connect finishes, so run_tts does
    # not trigger its own reconnect path.
    service._websocket = SimpleNamespace(state=State.OPEN)
    service._start_connect_task = asyncio.ensure_future(fake_connect())

    await _drain(service.run_tts("नमस्ते", context_id="ctx-1"))

    assert order == ["connected", "sent"]
    assert service._start_connect_task is None


@pytest.mark.asyncio
async def test_cancel_start_connect_cancels_pending_task(monkeypatch):
    """Teardown must cancel a still-pending background connect."""
    service = _make_service()
    started = asyncio.Event()

    async def never_finishes():
        started.set()
        await asyncio.Event().wait()

    async def fake_cancel_task(task, *args, **kwargs):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    monkeypatch.setattr(service, "cancel_task", fake_cancel_task)
    task = asyncio.ensure_future(never_finishes())
    service._start_connect_task = task
    await started.wait()

    await service._cancel_start_connect()

    assert task.cancelled()
    assert service._start_connect_task is None
