"""Routing decision logic and bounded per-session confirmation window tracking."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger

from .config import PLUGIN_NAME, RouterConfig
from .media import message_has_image
from .providers import (
    get_provider_id,
    provider_supports_vision,
    resolve_current_provider,
    resolve_provider_by_id,
)

EXTRA_SELECTED_PROVIDER = "selected_provider"
EXTRA_ROUTED = "mm_router_routed"
EXTRA_TARGET_PROVIDER_ID = "mm_router_target_provider_id"
EXTRA_SUMMARY_PROVIDER_ID = "mm_router_summary_provider_id"
EXTRA_SUMMARY_DONE = "mm_router_summary_done"
EXTRA_REQUEST_SNAPSHOT = "mm_router_request_snapshot"
EXTRA_SUMMARY_TASK_KEY = "mm_router_summary_task_key"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Outcome of evaluating whether an event should be temporarily routed."""

    should_route: bool
    reason: str
    current_provider_id: str = ""
    target_provider_id: str = ""
    summary_provider_id: str = ""


class ConfirmationTracker:
    """Bounded LRU tracker for post-route confirmation reminder turns per session."""

    def __init__(self, max_sessions: int = 2048) -> None:
        self._max_sessions = max(1, max_sessions)
        self._remaining: OrderedDict[str, int] = OrderedDict()

    def arm(self, session_id: str, turns: int) -> None:
        """Start or reset the confirmation window for a session."""
        if turns <= 0:
            self._remaining.pop(session_id, None)
            return
        self._remaining[session_id] = turns
        self._remaining.move_to_end(session_id)
        while len(self._remaining) > self._max_sessions:
            self._remaining.popitem(last=False)

    def consume(self, session_id: str) -> bool:
        """Consume one confirmation turn if active, returning True if prompt should inject."""
        remaining = self._remaining.get(session_id, 0)
        if remaining <= 0:
            self._remaining.pop(session_id, None)
            return False

        next_remaining = remaining - 1
        if next_remaining > 0:
            self._remaining[session_id] = next_remaining
            self._remaining.move_to_end(session_id)
        else:
            self._remaining.pop(session_id, None)
        return True

    def remaining(self, session_id: str) -> int:
        return self._remaining.get(session_id, 0)

    def clear(self) -> None:
        self._remaining.clear()

    def __len__(self) -> int:
        return len(self._remaining)


def session_key(event: Any) -> str:
    """Derive a stable per-session identifier from an AstrMessageEvent."""
    return str(
        getattr(event, "unified_msg_origin", None)
        or getattr(getattr(event, "message_obj", None), "session_id", None)
        or getattr(event, "session_id", None)
        or "default"
    )


async def decide_route(
    context: Any,
    event: Any,
    config: RouterConfig,
) -> RouteDecision:
    """Evaluate whether the current event needs temporary multimodal routing."""
    if not config.enabled:
        return RouteDecision(should_route=False, reason="disabled")

    if not message_has_image(event, include_quoted=config.route_quoted_images):
        return RouteDecision(should_route=False, reason="no_image")

    current_provider = await resolve_current_provider(context, event)
    if current_provider is None:
        logger.warning(f"{PLUGIN_NAME}: no active LLM provider; skip routing.")
        return RouteDecision(should_route=False, reason="no_current_provider")

    current_id = get_provider_id(current_provider)
    target_id = config.target_provider_id

    if target_id and current_id == target_id:
        return RouteDecision(
            should_route=False,
            reason="already_target",
            current_provider_id=current_id,
            target_provider_id=target_id,
        )

    if provider_supports_vision(current_provider, config.vision_provider_ids):
        return RouteDecision(
            should_route=False,
            reason="current_supports_vision",
            current_provider_id=current_id,
            target_provider_id=target_id,
        )

    if not target_id:
        logger.warning(f"{PLUGIN_NAME}: target_provider_id is empty; skip routing.")
        return RouteDecision(
            should_route=False,
            reason="missing_target",
            current_provider_id=current_id,
        )

    target_provider = await resolve_provider_by_id(context, target_id)
    if target_provider is None:
        logger.error(
            f"{PLUGIN_NAME}: target provider {target_id!r} was not found. "
            "Use /mmproviders to inspect loaded provider IDs."
        )
        return RouteDecision(
            should_route=False,
            reason="target_not_found",
            current_provider_id=current_id,
            target_provider_id=target_id,
        )

    effective_summary_id = (
        config.effective_summary_provider_id if config.enable_image_summary else ""
    )
    return RouteDecision(
        should_route=True,
        reason="routed",
        current_provider_id=current_id,
        target_provider_id=target_id,
        summary_provider_id=effective_summary_id,
    )


def apply_route_decision(event: Any, decision: RouteDecision) -> None:
    """Attach temporary routing extras to the event before AstrBot selects a provider."""
    if not decision.should_route:
        return
    event.set_extra(EXTRA_SELECTED_PROVIDER, decision.target_provider_id)
    event.set_extra(EXTRA_ROUTED, True)
    event.set_extra(EXTRA_TARGET_PROVIDER_ID, decision.target_provider_id)
    if decision.summary_provider_id:
        event.set_extra(EXTRA_SUMMARY_PROVIDER_ID, decision.summary_provider_id)


def append_system_prompt(req: Any, prompt: str) -> bool:
    """Append text to req.system_prompt without duplication."""
    if not hasattr(req, "system_prompt"):
        logger.warning(f"{PLUGIN_NAME}: ProviderRequest has no system_prompt field.")
        return False
    current = getattr(req, "system_prompt", "") or ""
    if prompt in current:
        return False
    separator = "\n\n" if current else ""
    req.system_prompt = f"{current}{separator}{prompt}"
    return True
