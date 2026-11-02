"""Cached images get random names. A manifest keeps what the user called them."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_discord_attachment_download import _PNG_BYTES, _make_adapter


def _attachment(filename):
    return SimpleNamespace(
        url=f"https://cdn.discordapp.com/attachments/fake/{filename}",
        filename=filename,
        content_type="image/png",
        size=len(_PNG_BYTES),
        read=AsyncMock(return_value=_PNG_BYTES),
    )


async def _upload(monkeypatch, *attachments, message_id=1566737254695829574):
    class _FakeDMChannel:
        id = 100
        name = "dm"

    monkeypatch.setattr("plugins.platforms.discord.adapter.discord.DMChannel", _FakeDMChannel)
    adapter = _make_adapter()
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter.handle_message = AsyncMock()
    await adapter._handle_message(
        SimpleNamespace(
            id=message_id,
            content="",
            attachments=list(attachments),
            mentions=[],
            reference=None,
            created_at=datetime.now(timezone.utc),
            channel=_FakeDMChannel(),
            author=SimpleNamespace(id=42, display_name="U", name="U"),
        )
    )
    return adapter.handle_message.call_args[0][0]


def _rows():
    from gateway.platforms.base import get_image_manifest_path

    path = get_image_manifest_path()
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.asyncio
async def test_uploaded_name_can_be_read_back_from_the_manifest(monkeypatch):
    event = await _upload(monkeypatch, _attachment("Crono.png"))

    cached = event.media_urls[0]
    assert "Crono" not in Path(cached).name
    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["path"] == cached
    assert rows[0]["filename"] == "Crono.png"
    assert rows[0]["message_id"] == "1566737254695829574"
    assert datetime.fromisoformat(rows[0]["time"])


@pytest.mark.asyncio
async def test_manifest_sits_beside_the_cache_so_cleanup_cannot_delete_it(monkeypatch):
    from gateway.platforms.base import (
        cleanup_image_cache,
        get_image_cache_dir,
        get_image_manifest_path,
    )

    await _upload(monkeypatch, _attachment("Crono.png"))
    assert get_image_manifest_path().parent == get_image_cache_dir().parent
    cleanup_image_cache(max_age_hours=-1)
    assert get_image_manifest_path().exists()


@pytest.mark.asyncio
async def test_each_image_in_a_message_gets_its_own_row(monkeypatch):
    await _upload(monkeypatch, _attachment("Crono.png"), _attachment("Marle.png"))
    assert [r["filename"] for r in _rows()] == ["Crono.png", "Marle.png"]


@pytest.mark.asyncio
async def test_hostile_filename_stays_one_harmless_row(monkeypatch):
    await _upload(monkeypatch, _attachment("../../etc/pass\nwd.png"))
    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["filename"] == "passwd.png"


@pytest.mark.asyncio
async def test_a_manifest_write_failure_does_not_lose_the_image(monkeypatch):
    def _boom():
        raise OSError("read only")

    monkeypatch.setattr("gateway.platforms.base.get_image_manifest_path", _boom)
    event = await _upload(monkeypatch, _attachment("Crono.png"))
    assert Path(event.media_urls[0]).exists()
