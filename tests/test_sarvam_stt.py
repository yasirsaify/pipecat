#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SarvamSTTService background-connect behavior."""

import asyncio
from types import SimpleNamespace

import pytest

from pipecat.services.sarvam.stt import SarvamSTTService


def _make_service():
    return SarvamSTTService(api_key="test-key")


async def _drain(agen):
    """Collect everything yielded by an async generator."""
    return [frame async for frame in agen]


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

    await service._ensure_connected()
    assert connected == [True]


@pytest.mark.asyncio
async def test_run_stt_awaits_background_connect_before_sending(monkeypatch):
    """The first run_stt must wait for the start()-scheduled connect before
    sending audio, so the user's opening words aren't dropped."""
    service = _make_service()
    order = []

    async def fake_connect():
        await asyncio.sleep(0.01)
        order.append("connected")
        # Connect installs the socket client; mimic that here.
        service._socket_client = SimpleNamespace(transcribe=record_send, translate=record_send)

    async def record_send(**kwargs):
        order.append("sent")

    service._start_connect_task = asyncio.ensure_future(fake_connect())

    await _drain(service.run_stt(b"\x00\x01"))

    assert order == ["connected", "sent"]
    assert service._start_connect_task is None


@pytest.mark.asyncio
async def test_run_stt_yields_none_when_connect_left_no_socket():
    """If the background connect failed (no socket client), run_stt must not
    raise — it yields None and returns."""
    service = _make_service()

    async def fake_connect():
        # Simulate a failed connect that leaves no socket client.
        return None

    service._socket_client = None
    service._start_connect_task = asyncio.ensure_future(fake_connect())

    frames = await _drain(service.run_stt(b"\x00\x01"))
    assert frames == [None]


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
