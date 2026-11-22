"""AstrBot plugin entry point for temporary multimodal routing and image summary."""

from __future__ import annotations

import uuid
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

from .config import DEFAULT_CONFIRM_PROMPT, PLUGIN_NAME, RouterConfig
from .media import backfill_image_urls_if_empty
from .providers import format_providers_report
from .routing import (
    EXTRA_REQUEST_SNAPSHOT,
    EXTRA_ROUTED,
    EXTRA_SUMMARY_DONE,
    EXTRA_SUMMARY_TASK_KEY,
    EXTRA_TARGET_PROVIDER_ID,
    ConfirmationTracker,
    append_system_prompt,
    apply_route_decision,
    decide_route,
    session_key,
)
from .summarizer import (
    ImageSummarizer,
    SummarySnapshot,
    build_summary_snapshot,
    inject_summary_into_legacy_contexts,
    inject_summary_into_run_messages,
)

_HAS_WAITING_LLM_HOOK = hasattr(filter, "on_waiting_llm_request")
_HAS_AGENT_LIFECYCLE_HOOKS = hasattr(filter, "on_agent_begin") and hasattr(
    filter, "on_agent_done"
)


def _pre_llm_route_hook():
    if _HAS_WAITING_LLM_HOOK:
        return filter.on_waiting_llm_request(priority=100)
    return filter.event_message_type(filter.EventMessageType.ALL, priority=100)


def _agent_begin_hook():
    if _HAS_AGENT_LIFECYCLE_HOOKS:
        return filter.on_agent_begin(priority=100)
    return lambda fn: fn


def _agent_done_hook():
    if _HAS_AGENT_LIFECYCLE_HOOKS:
        return filter.on_agent_done(priority=100)
    return lambda fn: fn


def _legacy_llm_response_hook():
    if not _HAS_AGENT_LIFECYCLE_HOOKS:
        return filter.on_llm_response(priority=100)
    return lambda fn: fn


def _is_successful_agent_response(run_context: Any, response: Any) -> bool:
    if response is None or getattr(response, "role", None) != "assistant":
        return False
    end_reason = getattr(run_context, "run_end_reason", None)
    return end_reason not in {"cancelled", "error"}


class MultimodalRouterPlugin(Star):
    """Route image requests to a multimodal provider and optionally summarize with a lighter model."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._confirm_tracker = ConfirmationTracker()
        self._summarizer = ImageSummarizer()

    def _load_config(self) -> RouterConfig:
        return RouterConfig.from_mapping(self.config)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("mmproviders")
    async def list_providers(self, event: AstrMessageEvent):
        """列出当前加载的 LLM providers、视觉能力与多模态路由配置诊断。"""
        try:
            report = await format_providers_report(
                self.context, event, self._load_config()
            )
            yield event.plain_result(report)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"{PLUGIN_NAME}: failed to list providers: {exc}", exc_info=True
            )
            yield event.plain_result("列出 provider 失败，请查看 AstrBot 日志。")

    @_pre_llm_route_hook()
    async def prepare_provider_selection(self, event: AstrMessageEvent) -> None:
        """Select the multimodal answering provider before AstrBot builds the LLM request."""
        try:
            if not _HAS_WAITING_LLM_HOOK and not getattr(
                event, "is_at_or_wake_command", True
            ):
                return
            if event.get_extra(EXTRA_ROUTED):
                return

            config = self._load_config()
            decision = await decide_route(self.context, event, config)
            if not decision.should_route:
                return

            apply_route_decision(event, decision)
            if decision.summary_provider_id:
                logger.info(
                    f"{PLUGIN_NAME}: routed image request to answer provider "
                    f"{decision.target_provider_id!r} (summary provider="
                    f"{decision.summary_provider_id!r})."
                )
            else:
                logger.info(
                    f"{PLUGIN_NAME}: routed image request to answer provider "
                    f"{decision.target_provider_id!r} (image summary disabled)."
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"{PLUGIN_NAME}: prepare_provider_selection failed: {exc}",
                exc_info=True,
            )

    @filter.on_llm_request(priority=100)
    async def on_llm_request(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        """Snapshot routed image inputs and inject follow-up confirmation prompts on text turns."""
        try:
            config = self._load_config()
            if not config.enabled:
                return

            if event.get_extra(EXTRA_ROUTED):
                backfill_image_urls_if_empty(
                    event, req, include_quoted=config.route_quoted_images
                )
                if config.enable_image_summary:
                    snapshot = build_summary_snapshot(req, config)
                    if snapshot is not None:
                        event.set_extra(EXTRA_REQUEST_SNAPSHOT, snapshot)
                return

            if self._confirm_tracker.consume(session_key(event)):
                append_system_prompt(req, DEFAULT_CONFIRM_PROMPT)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{PLUGIN_NAME}: on_llm_request failed: {exc}", exc_info=True)

    @_agent_begin_hook()
    async def on_agent_begin(
        self,
        event: AstrMessageEvent,
        run_context: Any,
    ) -> None:
        """Optionally start image summarization in parallel when using a dedicated summary model."""
        del run_context
        try:
            if not event.get_extra(EXTRA_ROUTED):
                return
            config = self._load_config()
            if not config.should_run_summary_in_parallel:
                return

            snapshot = event.get_extra(EXTRA_REQUEST_SNAPSHOT)
            if not isinstance(snapshot, SummarySnapshot):
                return

            routed_target_id = str(event.get_extra(EXTRA_TARGET_PROVIDER_ID) or "")
            task_key = uuid.uuid4().hex
            self._summarizer.start_background_summary(
                task_key,
                context=self.context,
                snapshot=snapshot,
                config=config,
                routed_target_id=routed_target_id,
            )
            event.set_extra(EXTRA_SUMMARY_TASK_KEY, task_key)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{PLUGIN_NAME}: on_agent_begin failed: {exc}", exc_info=True)

    @_agent_done_hook()
    async def on_agent_done(
        self,
        event: AstrMessageEvent,
        run_context: Any,
        response: LLMResponse | None,
    ) -> None:
        """Persist the hidden image summary into run_context.messages and arm confirmation turns."""
        try:
            if not event.get_extra(EXTRA_ROUTED):
                return
            if event.get_extra(EXTRA_SUMMARY_DONE):
                return

            task_key = event.get_extra(EXTRA_SUMMARY_TASK_KEY)
            if not _is_successful_agent_response(run_context, response):
                self._summarizer.cancel_task(task_key)
                return

            event.set_extra(EXTRA_SUMMARY_DONE, True)
            config = self._load_config()
            self._confirm_tracker.arm(session_key(event), config.confirm_turns)

            if not config.enable_image_summary:
                self._summarizer.cancel_task(task_key)
                return

            snapshot = self._resolve_snapshot(event, config)
            if snapshot is None:
                self._summarizer.cancel_task(task_key)
                return

            routed_target_id = str(event.get_extra(EXTRA_TARGET_PROVIDER_ID) or "")
            summary = await self._summarizer.await_or_generate(
                task_key,
                context=self.context,
                snapshot=snapshot,
                config=config,
                routed_target_id=routed_target_id,
            )
            if not summary:
                return

            messages = getattr(run_context, "messages", None)
            if isinstance(messages, list) and inject_summary_into_run_messages(
                messages, summary
            ):
                logger.info(
                    f"{PLUGIN_NAME}: injected hidden image summary into run_context.messages."
                )
                return

            req = event.get_extra("provider_request")
            if inject_summary_into_legacy_contexts(req, summary):
                logger.info(
                    f"{PLUGIN_NAME}: injected hidden image summary into legacy req.contexts."
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{PLUGIN_NAME}: on_agent_done failed: {exc}", exc_info=True)

    @_legacy_llm_response_hook()
    async def on_llm_response(
        self,
        event: AstrMessageEvent,
        resp: LLMResponse,
    ) -> None:
        """Legacy AstrBot (< 4.x) fallback when on_agent_done is unavailable."""
        try:
            if not event.get_extra(EXTRA_ROUTED):
                return
            if event.get_extra(EXTRA_SUMMARY_DONE):
                return
            if resp is None or getattr(resp, "role", None) != "assistant":
                return

            event.set_extra(EXTRA_SUMMARY_DONE, True)
            config = self._load_config()
            self._confirm_tracker.arm(session_key(event), config.confirm_turns)

            if not config.enable_image_summary:
                return

            snapshot = self._resolve_snapshot(event, config)
            if snapshot is None:
                return

            routed_target_id = str(event.get_extra(EXTRA_TARGET_PROVIDER_ID) or "")
            summary = await self._summarizer.generate_summary(
                self.context,
                snapshot,
                config,
                routed_target_id,
            )
            if not summary:
                return

            req = event.get_extra("provider_request")
            if inject_summary_into_legacy_contexts(req, summary):
                logger.info(
                    f"{PLUGIN_NAME}: injected hidden image summary into req.contexts."
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{PLUGIN_NAME}: on_llm_response failed: {exc}", exc_info=True)

    def _resolve_snapshot(
        self,
        event: AstrMessageEvent,
        config: RouterConfig,
    ) -> SummarySnapshot | None:
        snapshot = event.get_extra(EXTRA_REQUEST_SNAPSHOT)
        if isinstance(snapshot, SummarySnapshot):
            return snapshot
        req = event.get_extra("provider_request")
        if req is not None:
            return build_summary_snapshot(req, config)
        return None

    async def terminate(self) -> None:
        await self._summarizer.cancel_all()
        self._confirm_tracker.clear()
