"""Tests for direct and quoted image detection and URL backfill safety."""

from __future__ import annotations

from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.provider import ProviderRequest
from astrbot_plugin_multimodal_router.media import (
    backfill_image_urls_if_empty,
    message_has_image,
)


def test_detects_direct_and_quoted_images() -> None:
    plain_event = AstrMessageEvent([Plain("hello")])
    assert message_has_image(plain_event) is False

    direct_event = AstrMessageEvent([Plain("look"), Image(url="https://img/a.png")])
    assert message_has_image(direct_event, include_quoted=False) is True

    quoted_event = AstrMessageEvent(
        [
            Reply(id="100", chain=[Plain("old"), Image(url="https://img/quoted.png")]),
            Plain("what is this?"),
        ]
    )
    assert message_has_image(quoted_event, include_quoted=True) is True
    assert message_has_image(quoted_event, include_quoted=False) is False


def test_backfill_does_not_duplicate_astrbot_prepared_images() -> None:
    event = AstrMessageEvent([Image(url="https://platform.invalid/raw.png")])
    req = ProviderRequest(
        prompt="describe",
        image_urls=["/tmp/astrbot_compressed.jpg", "/tmp/astrbot_compressed.jpg"],
    )

    urls = backfill_image_urls_if_empty(event, req)

    assert urls == ["/tmp/astrbot_compressed.jpg"]
    assert req.image_urls == ["/tmp/astrbot_compressed.jpg"]


def test_backfill_populates_when_req_image_urls_is_empty() -> None:
    event = AstrMessageEvent(
        [
            Image(url="https://platform.invalid/1.png"),
            Reply(id="1", chain=[Image(path="/data/quoted.png")]),
        ]
    )
    req = ProviderRequest(prompt="compare", image_urls=[])

    urls = backfill_image_urls_if_empty(event, req, include_quoted=True)

    assert urls == ["https://platform.invalid/1.png", "/data/quoted.png"]
    assert req.image_urls == ["https://platform.invalid/1.png", "/data/quoted.png"]
