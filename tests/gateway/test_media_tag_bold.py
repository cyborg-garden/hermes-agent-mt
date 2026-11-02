"""Models wrap MEDIA tags in markdown bold. The file must still be delivered."""

from __future__ import annotations

import pytest

from gateway.platforms.base import BasePlatformAdapter


@pytest.mark.parametrize(
    "text",
    [
        "**MEDIA:/tmp/art/x.png**",
        "*MEDIA:/tmp/art/x.png*",
        "***MEDIA:/tmp/art/x.png***",
        "Here it is **MEDIA:/tmp/art/x.png** done",
        "**MEDIA: /tmp/art/x.png**",
        "MEDIA:/tmp/art/x.png",
    ],
)
def test_wrapped_tag_is_delivered_and_leaves_no_stray_asterisks(text):
    media, cleaned = BasePlatformAdapter.extract_media(text)
    assert media == [("/tmp/art/x.png", False)]
    assert "MEDIA" not in cleaned
    assert "*" not in cleaned


def test_bold_text_around_a_tag_is_left_alone():
    media, cleaned = BasePlatformAdapter.extract_media("**look** MEDIA:/tmp/x.png **now**")
    assert media == [("/tmp/x.png", False)]
    assert cleaned.split() == ["**look**", "**now**"]


def test_wrapped_tag_for_a_file_type_outside_the_known_list(tmp_path):
    target = tmp_path / "notes.weirdext"
    target.write_text("hi")
    media, cleaned = BasePlatformAdapter.extract_media(f"**MEDIA:{target}**")
    assert media and media[0][0] == str(target)
    assert "*" not in cleaned
