"""Cached images get random names. A manifest keeps what the user called them."""

from __future__ import annotations

import json
import os
import time
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


def _record(name, age_hours=0):
    from gateway.platforms.base import get_image_cache_dir, record_image_origin

    cached = get_image_cache_dir() / name
    cached.write_bytes(b"x")
    if age_hours:
        stamp = time.time() - age_hours * 3600
        os.utime(cached, (stamp, stamp))
    record_image_origin(str(cached), name, "1")
    return cached


def test_cleanup_drops_rows_for_the_images_it_deleted():
    from gateway.platforms.base import cleanup_image_cache

    old = _record("old.png", age_hours=48)
    fresh = _record("fresh.png")

    assert cleanup_image_cache(max_age_hours=24) == 1

    assert not old.exists() and fresh.exists()
    assert [r["filename"] for r in _rows()] == ["fresh.png"]


def test_cleanup_with_no_manifest_creates_none():
    from gateway.platforms.base import cleanup_image_cache, get_image_manifest_path

    cleanup_image_cache(max_age_hours=24)
    assert not get_image_manifest_path().exists()


def test_cleanup_does_not_rewrite_a_manifest_with_nothing_to_drop():
    from gateway.platforms.base import cleanup_image_cache, get_image_manifest_path

    _record("fresh.png")
    before = get_image_manifest_path().stat().st_mtime_ns
    cleanup_image_cache(max_age_hours=24)
    assert get_image_manifest_path().stat().st_mtime_ns == before


def test_cleanup_drops_rows_it_cannot_read_and_keeps_the_rest():
    from gateway.platforms.base import cleanup_image_cache, get_image_manifest_path

    _record("fresh.png")
    with get_image_manifest_path().open("a", encoding="utf-8") as fh:
        fh.write("not json\n")
    cleanup_image_cache(max_age_hours=24)
    assert [r["filename"] for r in _rows()] == ["fresh.png"]


def test_a_manifest_prune_failure_does_not_stop_cleanup(monkeypatch):
    from gateway.platforms.base import cleanup_image_cache

    def _boom():
        raise OSError("read only")

    old = _record("old.png", age_hours=48)
    monkeypatch.setattr("gateway.platforms.base.get_image_manifest_path", _boom)
    assert cleanup_image_cache(max_age_hours=24) == 1
    assert not old.exists()
