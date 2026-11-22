"""Tests for route decisions and bounded confirmation tracking."""

from __future__ import annotations

import asyncio

from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Image, Plain, Reply
from astrbot_plugin_multimodal_router.config import RouterConfig
from astrbot_plugin_multimodal_router.routing import (
    EXTRA_ROUTED,
    EXTRA_SELECTED_PROVIDER,
    EXTRA_SUMMARY_PROVIDER_ID,
    EXTRA_TARGET_PROVIDER_ID,
    ConfirmationTracker,
    apply_route_decision,
    decide_route,
)

from tests.conftest import FakeContext, FakeProvider


def test_confirmation_tracker_counts_and_evicts_lru() -> None:
    tracker = ConfirmationTracker(max_sessions=2)
    tracker.arm("s1", 2)
    tracker.arm("s2", 1)
    assert tracker.remaining("s1") == 2

    # Adding s3 evicts least recently used s1.
    tracker.arm("s3", 1)
    assert tracker.remaining("s1") == 0
    assert tracker.remaining("s2") == 1
    assert tracker.remaining("s3") == 1

    assert tracker.consume("s2") is True
    assert tracker.consume("s2") is False

    # Arming with 0 clears the session.
    tracker.arm("s3", 0)
    assert tracker.consume("s3") is False


def test_decide_route_selects_answer_and_summary_models() -> None:
    ds = FakeProvider("ds", modalities=["text"])
    opus = FakeProvider("opus5", modalities=["text", "image"])
    kimi = FakeProvider("kimi", modalities=["text", "image"])
    ctx = FakeContext([ds, opus, kimi], using_provider=ds)
    cfg = RouterConfig.from_mapping(
        {
            "target_provider_id": "opus5",
            "summary_provider_id": "kimi",
        }
    )
    event = AstrMessageEvent([Plain("这是什么"), Image(url="https://img/1.png")])

    decision = asyncio.run(decide_route(ctx, event, cfg))

    assert decision.should_route is True
    assert decision.reason == "routed"
    assert decision.target_provider_id == "opus5"
    assert decision.summary_provider_id == "kimi"

    apply_route_decision(event, decision)
    assert event.get_extra(EXTRA_SELECTED_PROVIDER) == "opus5"
    assert event.get_extra(EXTRA_ROUTED) is True
    assert event.get_extra(EXTRA_TARGET_PROVIDER_ID) == "opus5"
    assert event.get_extra(EXTRA_SUMMARY_PROVIDER_ID) == "kimi"


def test_decide_route_skips_when_disabled_no_image_or_vision_capable() -> None:
    ds = FakeProvider("ds", modalities=["text"])
    opus = FakeProvider("opus5", modalities=["text", "image"])
    ctx_text = FakeContext([ds, opus], using_provider=ds)
    ctx_vision = FakeContext([ds, opus], using_provider=opus)

    img_event = AstrMessageEvent([Image(url="https://img/1.png")])
    text_event = AstrMessageEvent([Plain("pure text")])

    disabled_cfg = RouterConfig.from_mapping(
        {"enabled": False, "target_provider_id": "opus5"}
    )
    assert (
        asyncio.run(decide_route(ctx_text, img_event, disabled_cfg)).reason
        == "disabled"
    )

    normal_cfg = RouterConfig.from_mapping({"target_provider_id": "opus5"})
    assert (
        asyncio.run(decide_route(ctx_text, text_event, normal_cfg)).reason == "no_image"
    )
    assert (
        asyncio.run(decide_route(ctx_vision, img_event, normal_cfg)).reason
        == "already_target"
    )

    other_vision = FakeProvider("gpt4o", modalities=["text", "image"])
    ctx_other_vision = FakeContext(
        [ds, opus, other_vision], using_provider=other_vision
    )
    assert (
        asyncio.run(decide_route(ctx_other_vision, img_event, normal_cfg)).reason
        == "current_supports_vision"
    )


def test_decide_route_handles_quoted_images_flag() -> None:
    ds = FakeProvider("ds", modalities=["text"])
    opus = FakeProvider("opus5", modalities=["text", "image"])
    ctx = FakeContext([ds, opus], using_provider=ds)
    quoted_event = AstrMessageEvent(
        [Reply(id="99", chain=[Image(url="https://img/quoted.png")]), Plain("解释一下")]
    )

    cfg_on = RouterConfig.from_mapping(
        {"target_provider_id": "opus5", "route_quoted_images": True}
    )
    cfg_off = RouterConfig.from_mapping(
        {"target_provider_id": "opus5", "route_quoted_images": False}
    )

    assert asyncio.run(decide_route(ctx, quoted_event, cfg_on)).should_route is True
    assert asyncio.run(decide_route(ctx, quoted_event, cfg_off)).should_route is False
