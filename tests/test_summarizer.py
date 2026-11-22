"""Tests for dedicated image summarization, context sanitization, and history injection."""

from __future__ import annotations

import asyncio

from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import ImageURLPart, Message, TextPart
from astrbot_plugin_multimodal_router.config import RouterConfig
from astrbot_plugin_multimodal_router.summarizer import (
    IMAGE_CONTEXT_OPEN_TAG,
    ImageSummarizer,
    build_summary_snapshot,
    clean_summary_text,
    extract_recent_text_context,
    inject_summary_into_legacy_contexts,
    inject_summary_into_run_messages,
)

from tests.conftest import FakeContext, FakeProvider


def test_extract_recent_text_context_strips_tools_system_and_images() -> None:
    contexts = [
        {"role": "system", "content": "secret persona"},
        {"role": "user", "content": "第一句问题"},
        {"role": "assistant", "content": "第一句回答"},
        {"role": "tool", "tool_call_id": "1", "content": "raw tool json"},
        Message(
            role="user",
            content=[
                TextPart(text="看看这张图"),
                ImageURLPart(image_url={"url": "https://img/old.png"}),
                TextPart(text="<image_context>旧摘要</image_context>"),
            ],
        ),
        Message(role="assistant", content=[TextPart(text="这是旧回复")]),
    ]

    digest = extract_recent_text_context(contexts, max_messages=3)

    assert "secret persona" not in digest
    assert "raw tool json" not in digest
    assert "old.png" not in digest
    assert "旧摘要" not in digest
    assert "助手: 第一句回答" in digest
    assert "用户: 看看这张图" in digest
    assert "助手: 这是旧回复" in digest


def test_clean_summary_text_strips_think_tags_and_truncates() -> None:
    raw = "<think>我先分析一下图片</think>图里是一张咖啡收据，总价 38 元。"
    cleaned = clean_summary_text(raw, max_chars=12)
    assert cleaned == "图里是一张咖啡收据，总价"


def test_summarizer_uses_separate_summary_provider_without_forwarding_model() -> None:
    opus = FakeProvider("opus5", model="claude-opus-5", modalities=["text", "image"])
    kimi = FakeProvider(
        "kimi",
        model="kimi-vl",
        modalities=["text", "image"],
        response_text="一张显示股价走势的折线图。",
    )
    # Test text_chat fallback path (use_llm_generate=False) to verify model is never forwarded.
    ctx = FakeContext([opus, kimi], using_provider=opus, use_llm_generate=False)
    cfg = RouterConfig.from_mapping(
        {
            "target_provider_id": "opus5",
            "summary_provider_id": "kimi",
        }
    )
    req = ProviderRequest(
        prompt="帮我分析趋势",
        image_urls=["/tmp/chart.png"],
        model="claude-opus-5",
        session_id="sess-1",
    )
    snapshot = build_summary_snapshot(req, cfg)
    assert snapshot is not None

    summarizer = ImageSummarizer()
    summary = asyncio.run(summarizer.generate_summary(ctx, snapshot, cfg, "opus5"))

    assert summary == "一张显示股价走势的折线图。"
    assert len(opus.calls) == 0
    assert len(kimi.calls) == 1
    assert kimi.calls[0]["model"] is None
    assert kimi.calls[0]["image_urls"] == ["/tmp/chart.png"]


def test_summarizer_falls_back_to_target_provider_when_summary_missing() -> None:
    opus = FakeProvider(
        "opus5",
        model="claude-opus-5",
        modalities=["text", "image"],
        response_text="由 opus5 生成的回退摘要。",
    )
    ctx = FakeContext([opus], using_provider=opus, use_llm_generate=True)
    cfg = RouterConfig.from_mapping(
        {
            "target_provider_id": "opus5",
            "summary_provider_id": "missing-kimi",
        }
    )
    snapshot = build_summary_snapshot(
        ProviderRequest(prompt="看图", image_urls=["/tmp/a.png"]), cfg
    )
    assert snapshot is not None

    summary = asyncio.run(
        ImageSummarizer().generate_summary(ctx, snapshot, cfg, "opus5")
    )

    assert summary == "由 opus5 生成的回退摘要。"
    assert len(ctx.llm_generate_calls) == 1
    assert ctx.llm_generate_calls[0]["chat_provider_id"] == "opus5"


def test_background_summary_finishing_early_is_only_called_once() -> None:
    async def scenario() -> None:
        kimi = FakeProvider("kimi", response_text="快速总结完成。")
        opus = FakeProvider("opus5")
        ctx = FakeContext([opus, kimi], using_provider=opus, use_llm_generate=True)
        cfg = RouterConfig.from_mapping(
            {"target_provider_id": "opus5", "summary_provider_id": "kimi"}
        )
        snapshot = build_summary_snapshot(
            ProviderRequest(prompt="看图", image_urls=["/tmp/fast.png"]), cfg
        )
        assert snapshot is not None

        summarizer = ImageSummarizer()
        task = summarizer.start_background_summary(
            "t1",
            context=ctx,
            snapshot=snapshot,
            config=cfg,
            routed_target_id="opus5",
        )
        # Wait until the fast summary task has already finished before main answer completes.
        await asyncio.sleep(0.02)
        assert task.done() is True

        result = await summarizer.await_or_generate(
            "t1",
            context=ctx,
            snapshot=snapshot,
            config=cfg,
            routed_target_id="opus5",
        )
        assert result == "快速总结完成。"
        assert len(kimi.calls) == 1
        assert summarizer.active_task_count() == 0

    asyncio.run(scenario())


def test_inject_summary_into_run_messages_attaches_to_image_user_message() -> None:
    image_user_msg = Message(
        role="user",
        content=[
            TextPart(text="帮我查一下图里书的价格"),
            ImageURLPart(image_url={"url": "/tmp/book.png"}),
        ],
    )
    steering_user_msg = Message(role="user", content=[TextPart(text="顺便看下作者")])
    messages = [
        Message(role="system", content="你是助手"),
        image_user_msg,
        Message(role="assistant", content=[TextPart(text="正在查询")]),
        steering_user_msg,
        Message(role="assistant", content=[TextPart(text="价格是 59 元")]),
    ]

    assert (
        inject_summary_into_run_messages(messages, "图片是一本 Python 书籍封面。")
        is True
    )
    # No mid-conversation system message added (safe for Anthropic/Gemini).
    assert sum(1 for msg in messages if msg.role == "system") == 1
    # Attached specifically to the user message containing the image.
    assert len(image_user_msg.content) == 3
    injected = image_user_msg.content[-1]
    assert isinstance(injected, TextPart)
    assert IMAGE_CONTEXT_OPEN_TAG in injected.text
    assert "Python 书籍封面" in injected.text
    # Steering message stays untouched.
    assert len(steering_user_msg.content) == 1
    # Idempotent if called again.
    assert inject_summary_into_run_messages(messages, "重复调用") is False


def test_inject_summary_into_legacy_contexts_uses_user_role() -> None:
    req = ProviderRequest(contexts=[{"role": "assistant", "content": "hi"}])
    assert inject_summary_into_legacy_contexts(req, "legacy summary") is True
    assert req.contexts[-1]["role"] == "user"
    assert "legacy summary" in req.contexts[-1]["content"]
