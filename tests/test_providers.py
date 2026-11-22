"""Tests for provider capability inspection and /mmproviders diagnostics."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from astrbot.api.event import AstrMessageEvent
from astrbot_plugin_multimodal_router.config import RouterConfig
from astrbot_plugin_multimodal_router.providers import (
    format_providers_report,
    inspect_vision_capability,
    resolve_current_provider,
)

from tests.conftest import FakeContext, FakeProvider


def test_vision_capability_respects_astrbot_modalities_semantics() -> None:
    text_only = FakeProvider(
        "ds", model="deepseek-chat", modalities=["text", "tool_use"]
    )
    cap_text = inspect_vision_capability(text_only)
    assert cap_text.supported is False
    assert cap_text.source == "modalities"

    vision = FakeProvider("opus5", model="claude-opus-5", modalities=["text", "image"])
    cap_vision = inspect_vision_capability(vision)
    assert cap_vision.supported is True
    assert cap_vision.source == "modalities"

    legacy_empty = FakeProvider("legacy", model="legacy-model", modalities=[])
    cap_empty = inspect_vision_capability(legacy_empty)
    assert cap_empty.supported is True
    assert cap_empty.source == "modalities_legacy_empty"


def test_vision_whitelist_overrides_modalities() -> None:
    provider = FakeProvider("custom", model="my-model", modalities=["text"])
    cap = inspect_vision_capability(provider, frozenset({"custom"}))
    assert cap.supported is True
    assert cap.source == "whitelist"


def test_heuristic_avoids_vl_substring_false_positives() -> None:
    false_positive_candidate = SimpleNamespace(
        id="devlog-bot",
        capabilities=["devlog", "available", "eval"],
    )
    assert inspect_vision_capability(false_positive_candidate).supported is False

    true_vl = SimpleNamespace(
        id="kimi-vl",
        capabilities=["kimi-vl-a3b"],
    )
    cap = inspect_vision_capability(true_vl)
    assert cap.supported is True
    assert cap.source == "heuristic"


def test_resolve_current_provider_prefers_selected_then_async_then_sync() -> None:
    ds = FakeProvider("ds", modalities=["text"])
    opus = FakeProvider("opus5", modalities=["text", "image"])

    ctx_async = FakeContext(
        [ds, opus], using_provider=ds, use_async_using_provider=True
    )
    event = AstrMessageEvent()
    assert asyncio.run(resolve_current_provider(ctx_async, event)) is ds

    event.set_extra("selected_provider", "opus5")
    assert asyncio.run(resolve_current_provider(ctx_async, event)) is opus

    ctx_sync = FakeContext([ds], using_provider=ds, use_async_using_provider=False)
    plain_event = AstrMessageEvent()
    assert asyncio.run(resolve_current_provider(ctx_sync, plain_event)) is ds


def test_format_providers_report_shows_dual_models_and_warnings() -> None:
    ds = FakeProvider("ds", model="deepseek-chat", modalities=["text"])
    opus = FakeProvider("opus5", model="claude-opus-5", modalities=["text", "image"])
    kimi = FakeProvider("kimi", model="kimi-vl", modalities=["text", "image"])
    ctx = FakeContext(
        [ds, opus, kimi],
        using_provider=ds,
        native_caption_provider_id="native-cap",
    )
    cfg = RouterConfig.from_mapping(
        {
            "target_provider_id": "opus5",
            "summary_provider_id": "kimi",
        }
    )

    report = asyncio.run(format_providers_report(ctx, AstrMessageEvent(), cfg))

    assert "id=ds, model=deepseek-chat" in report
    assert "[当前会话]" in report
    assert "[临时回答]" in report
    assert "[图片总结]" in report
    assert "图片总结模型 (summary_provider_id): kimi (并行)" in report
    assert "原生图片转述模型已启用 (native-cap)" in report
