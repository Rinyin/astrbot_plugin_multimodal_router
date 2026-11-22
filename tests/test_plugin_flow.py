"""End-to-end plugin lifecycle tests covering Issue #1 dual-model flow and edge cases."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Image, Plain
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.core.agent.message import ImageURLPart, Message, TextPart
from astrbot_plugin_multimodal_router.config import DEFAULT_CONFIRM_PROMPT
from astrbot_plugin_multimodal_router.main import MultimodalRouterPlugin
from astrbot_plugin_multimodal_router.routing import EXTRA_SELECTED_PROVIDER
from astrbot_plugin_multimodal_router.summarizer import IMAGE_CONTEXT_OPEN_TAG

from tests.conftest import FakeContext, FakeProvider


def test_full_issue1_dual_model_flow_and_followup_confirm_turns() -> None:
    async def scenario() -> None:
        ds = FakeProvider("ds", model="deepseek-chat", modalities=["text"])
        opus = FakeProvider(
            "opus5", model="claude-opus-5", modalities=["text", "image"]
        )
        kimi = FakeProvider(
            "kimi",
            model="kimi-vl",
            modalities=["text", "image"],
            response_text="图片是一张显示总价 128 元的餐厅小票。",
        )
        ctx = FakeContext([ds, opus, kimi], using_provider=ds, use_llm_generate=True)
        plugin = MultimodalRouterPlugin(
            ctx,
            {
                "target_provider_id": "opus5",
                "summary_provider_id": "kimi",
                "enable_image_summary": True,
                "parallel_image_summary": True,
                "confirm_turns": 2,
            },
        )

        # Turn 1: User sends an image while session default is pure-text DeepSeek.
        img_event = AstrMessageEvent(
            [Plain("帮我算一下人均"), Image(url="https://cdn.invalid/receipt.png")],
            umo="group:123",
        )
        await plugin.prepare_provider_selection(img_event)
        assert img_event.get_extra(EXTRA_SELECTED_PROVIDER) == "opus5"

        req1 = ProviderRequest(
            prompt="帮我算一下人均",
            session_id="group:123",
            image_urls=["/tmp/prepared_receipt.jpg"],
            system_prompt="你是日常助手",
            model="claude-opus-5",
        )
        await plugin.on_llm_request(img_event, req1)
        # Routed multimodal turn must NOT duplicate images or consume confirm_turns.
        assert req1.image_urls == ["/tmp/prepared_receipt.jpg"]
        assert DEFAULT_CONFIRM_PROMPT not in req1.system_prompt

        run_context = SimpleNamespace(
            run_end_reason="completed",
            messages=[
                Message(role="system", content=req1.system_prompt),
                Message(
                    role="user",
                    content=[
                        TextPart(text="帮我算一下人均"),
                        ImageURLPart(image_url={"url": "/tmp/prepared_receipt.jpg"}),
                    ],
                ),
                Message(
                    role="assistant",
                    content=[TextPart(text="共 128 元，两人人均 64 元。")],
                ),
            ],
        )

        # Agent begins -> parallel Kimi summary task starts.
        await plugin.on_agent_begin(img_event, run_context)
        # Agent finishes -> summary awaited and injected into run_context.messages.
        await plugin.on_agent_done(
            img_event,
            run_context,
            LLMResponse(
                role="assistant", completion_text="共 128 元，两人人均 64 元。"
            ),
        )

        assert len(opus.calls) == 0  # Expensive opus5 never called for image summary!
        assert len(kimi.calls) == 1  # Cheap kimi generated the summary exactly once!
        assert ctx.llm_generate_calls[0]["chat_provider_id"] == "kimi"

        user_msg = run_context.messages[1]
        assert isinstance(user_msg.content[-1], TextPart)
        assert IMAGE_CONTEXT_OPEN_TAG in user_msg.content[-1].text
        assert "128 元" in user_msg.content[-1].text

        # Turn 2 & Turn 3: Pure text follow-ups back on DeepSeek receive confirm prompt.
        for turn_idx in range(2):
            follow_event = AstrMessageEvent(
                [Plain(f"追问 {turn_idx}")], umo="group:123"
            )
            await plugin.prepare_provider_selection(follow_event)
            assert follow_event.get_extra(EXTRA_SELECTED_PROVIDER) is None
            follow_req = ProviderRequest(
                prompt=f"追问 {turn_idx}", system_prompt="base"
            )
            await plugin.on_llm_request(follow_event, follow_req)
            assert DEFAULT_CONFIRM_PROMPT in follow_req.system_prompt

        # Turn 4: Confirmation window expired.
        turn4_event = AstrMessageEvent([Plain("普通闲聊")], umo="group:123")
        turn4_req = ProviderRequest(prompt="普通闲聊", system_prompt="base")
        await plugin.on_llm_request(turn4_event, turn4_req)
        assert DEFAULT_CONFIRM_PROMPT not in turn4_req.system_prompt

        await plugin.terminate()

    asyncio.run(scenario())


def test_failed_or_cancelled_answer_cancels_parallel_summary_and_does_not_arm_confirm() -> (
    None
):
    async def scenario() -> None:
        started = asyncio.Event()
        cancelled = False

        async def slow_kimi(_call):
            nonlocal cancelled
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled = True
                raise

        ds = FakeProvider("ds", modalities=["text"])
        opus = FakeProvider("opus5", modalities=["text", "image"])
        kimi = FakeProvider(
            "kimi",
            modalities=["text", "image"],
            text_chat_side_effect=slow_kimi,
        )
        ctx = FakeContext([ds, opus, kimi], using_provider=ds, use_llm_generate=True)
        plugin = MultimodalRouterPlugin(
            ctx,
            {
                "target_provider_id": "opus5",
                "summary_provider_id": "kimi",
                "parallel_image_summary": True,
                "confirm_turns": 3,
            },
        )

        event = AstrMessageEvent([Image(url="https://img/1.png")], umo="group:cancel")
        await plugin.prepare_provider_selection(event)
        req = ProviderRequest(prompt="看图", image_urls=["/tmp/1.png"])
        await plugin.on_llm_request(event, req)

        run_context = SimpleNamespace(run_end_reason="error", messages=[])
        await plugin.on_agent_begin(event, run_context)
        await started.wait()

        # Answering provider errored out.
        await plugin.on_agent_done(
            event,
            run_context,
            LLMResponse(role="err", completion_text="upstream error"),
        )
        await asyncio.sleep(0.01)

        assert cancelled is True
        assert plugin._summarizer.active_task_count() == 0
        assert plugin._confirm_tracker.remaining("group:cancel") == 0

        await plugin.terminate()

    asyncio.run(scenario())


def test_disable_image_summary_skips_summary_provider_entirely() -> None:
    async def scenario() -> None:
        ds = FakeProvider("ds", modalities=["text"])
        opus = FakeProvider("opus5", modalities=["text", "image"])
        kimi = FakeProvider("kimi", modalities=["text", "image"])
        ctx = FakeContext([ds, opus, kimi], using_provider=ds, use_llm_generate=True)
        plugin = MultimodalRouterPlugin(
            ctx,
            {
                "target_provider_id": "opus5",
                "summary_provider_id": "kimi",
                "enable_image_summary": False,
                "confirm_turns": 0,
            },
        )

        event = AstrMessageEvent([Image(url="https://img/1.png")], umo="group:nosum")
        await plugin.prepare_provider_selection(event)
        assert event.get_extra(EXTRA_SELECTED_PROVIDER) == "opus5"

        req = ProviderRequest(prompt="看图", image_urls=["/tmp/1.png"])
        await plugin.on_llm_request(event, req)
        run_context = SimpleNamespace(run_end_reason="completed", messages=[])
        await plugin.on_agent_begin(event, run_context)
        await plugin.on_agent_done(
            event,
            run_context,
            LLMResponse(role="assistant", completion_text="回答完成"),
        )

        assert len(kimi.calls) == 0
        assert len(opus.calls) == 0
        assert plugin._confirm_tracker.remaining("group:nosum") == 0

        await plugin.terminate()

    asyncio.run(scenario())


def test_mmproviders_admin_command_yields_report() -> None:
    async def scenario() -> None:
        ds = FakeProvider("ds", model="deepseek-chat", modalities=["text"])
        opus = FakeProvider(
            "opus5", model="claude-opus-5", modalities=["text", "image"]
        )
        kimi = FakeProvider("kimi", model="kimi-vl", modalities=["text", "image"])
        ctx = FakeContext([ds, opus, kimi], using_provider=ds)
        plugin = MultimodalRouterPlugin(
            ctx,
            {
                "target_provider_id": "opus5",
                "summary_provider_id": "kimi",
            },
        )

        results = []
        async for out in plugin.list_providers(AstrMessageEvent()):
            results.append(out)

        assert len(results) == 1
        assert "opus5" in results[0]
        assert "kimi" in results[0]
        assert "[临时回答]" in results[0]
        assert "[图片总结]" in results[0]

    asyncio.run(scenario())


def test_legacy_on_llm_response_fallback_works_without_agent_done() -> None:
    async def scenario() -> None:
        ds = FakeProvider("ds", modalities=["text"])
        opus = FakeProvider("opus5", modalities=["text", "image"])
        kimi = FakeProvider(
            "kimi", modalities=["text", "image"], response_text="旧版摘要"
        )
        ctx = FakeContext([ds, opus, kimi], using_provider=ds, use_llm_generate=False)
        plugin = MultimodalRouterPlugin(
            ctx,
            {
                "target_provider_id": "opus5",
                "summary_provider_id": "kimi",
            },
        )

        event = AstrMessageEvent([Image(url="https://img/legacy.png")], umo="legacy:1")
        await plugin.prepare_provider_selection(event)
        req = ProviderRequest(
            prompt="看看",
            image_urls=["https://img/legacy.png"],
            contexts=[],
        )
        event.set_extra("provider_request", req)
        await plugin.on_llm_request(event, req)
        await plugin.on_llm_response(
            event, LLMResponse(role="assistant", completion_text="回答")
        )

        assert len(kimi.calls) == 1
        assert len(req.contexts) == 1
        assert req.contexts[0]["role"] == "user"
        assert "旧版摘要" in req.contexts[0]["content"]
        assert plugin._confirm_tracker.remaining("legacy:1") == 3

        await plugin.terminate()

    asyncio.run(scenario())


def test_blank_summary_provider_runs_serially_using_target_provider() -> None:
    async def scenario() -> None:
        ds = FakeProvider("ds", modalities=["text"])
        opus = FakeProvider(
            "opus5",
            modalities=["text", "image"],
            response_text="同一模型串行生成的摘要。",
        )
        ctx = FakeContext([ds, opus], using_provider=ds, use_llm_generate=True)
        plugin = MultimodalRouterPlugin(
            ctx,
            {
                "target_provider_id": "opus5",
                "summary_provider_id": "",
                "parallel_image_summary": True,
            },
        )

        event = AstrMessageEvent([Image(url="https://img/same.png")], umo="group:same")
        await plugin.prepare_provider_selection(event)
        req = ProviderRequest(prompt="看图", image_urls=["/tmp/same.png"])
        await plugin.on_llm_request(event, req)

        run_context = SimpleNamespace(
            run_end_reason="completed",
            messages=[
                Message(role="system", content="sys"),
                Message(
                    role="user",
                    content=[
                        TextPart(text="看图"),
                        ImageURLPart(image_url={"url": "/tmp/same.png"}),
                    ],
                ),
                Message(role="assistant", content=[TextPart(text="好的")]),
            ],
        )

        # Same provider must NOT start a concurrent task in on_agent_begin.
        await plugin.on_agent_begin(event, run_context)
        assert plugin._summarizer.active_task_count() == 0
        assert len(opus.calls) == 0

        await plugin.on_agent_done(
            event,
            run_context,
            LLMResponse(role="assistant", completion_text="好的"),
        )
        assert len(opus.calls) == 1
        assert "同一模型串行生成的摘要" in run_context.messages[1].content[-1].text

        await plugin.terminate()

    asyncio.run(scenario())


def test_summary_provider_error_never_breaks_turn_completion() -> None:
    async def scenario() -> None:
        def boom(_call):
            raise RuntimeError("summary upstream down")

        ds = FakeProvider("ds", modalities=["text"])
        opus = FakeProvider("opus5", modalities=["text", "image"])
        kimi = FakeProvider(
            "kimi",
            modalities=["text", "image"],
            text_chat_side_effect=boom,
        )
        ctx = FakeContext([ds, opus, kimi], using_provider=ds, use_llm_generate=True)
        plugin = MultimodalRouterPlugin(
            ctx,
            {
                "target_provider_id": "opus5",
                "summary_provider_id": "kimi",
                "confirm_turns": 2,
            },
        )

        event = AstrMessageEvent([Image(url="https://img/err.png")], umo="group:err")
        await plugin.prepare_provider_selection(event)
        req = ProviderRequest(prompt="看图", image_urls=["/tmp/err.png"])
        await plugin.on_llm_request(event, req)

        user_msg = Message(
            role="user",
            content=[
                TextPart(text="看图"),
                ImageURLPart(image_url={"url": "/tmp/err.png"}),
            ],
        )
        run_context = SimpleNamespace(
            run_end_reason="completed",
            messages=[
                Message(role="system", content="sys"),
                user_msg,
                Message(role="assistant", content=[TextPart(text="主回答正常")]),
            ],
        )

        await plugin.on_agent_begin(event, run_context)
        await plugin.on_agent_done(
            event,
            run_context,
            LLMResponse(role="assistant", completion_text="主回答正常"),
        )

        # Turn completed cleanly: no crash, confirm window armed, user message uncorrupted.
        assert len(user_msg.content) == 2
        assert plugin._confirm_tracker.remaining("group:err") == 2

        await plugin.terminate()

    asyncio.run(scenario())
