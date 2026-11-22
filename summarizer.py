"""Dedicated image summarization service and conversation history injection."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger

from .config import (
    DEFAULT_SUMMARY_TIMEOUT_SECONDS,
    PLUGIN_NAME,
    RouterConfig,
)
from .media import dedupe_image_urls
from .providers import resolve_provider_by_id

try:
    from astrbot.core.agent.message import TextPart as _AstrBotTextPart
except ImportError:  # pragma: no cover - fallback for stripped environments
    _AstrBotTextPart = None

IMAGE_CONTEXT_OPEN_TAG = "<image_context>"
IMAGE_CONTEXT_CLOSE_TAG = "</image_context>"

_THINK_TAG_RE = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_IMAGE_CONTEXT_RE = re.compile(
    r"<image_context>.*?</image_context>",
    re.DOTALL | re.IGNORECASE,
)
_MAX_CONTEXT_ENTRY_CHARS = 240


@dataclass(frozen=True, slots=True)
class SummarySnapshot:
    """Immutable snapshot of a routed turn's image inputs and compact text context."""

    prompt: str
    image_urls: tuple[str, ...]
    recent_context_text: str = ""
    session_id: str | None = None


def extract_recent_text_context(
    contexts: Sequence[Any] | None,
    *,
    max_messages: int,
) -> str:
    """Extract a compact, text-only digest of recent user/assistant turns.

    Tool messages, system prompts, and historical media blocks are intentionally
    omitted so lightweight vision models stay cheap and compatible.
    """
    if not contexts or max_messages <= 0:
        return ""

    entries: list[str] = []
    for raw_msg in contexts:
        role = _message_role(raw_msg)
        if role not in {"user", "assistant"}:
            continue
        text = _extract_plain_text_from_content(_message_content(raw_msg))
        text = _IMAGE_CONTEXT_RE.sub("", text).strip()
        if not text:
            continue
        if len(text) > _MAX_CONTEXT_ENTRY_CHARS:
            text = f"{text[:_MAX_CONTEXT_ENTRY_CHARS].rstrip()}…"
        speaker = "用户" if role == "user" else "助手"
        entries.append(f"{speaker}: {text}")

    if not entries:
        return ""
    return "\n".join(entries[-max_messages:])


def build_summary_snapshot(req: Any, config: RouterConfig) -> SummarySnapshot | None:
    """Capture an immutable summary snapshot from a prepared ProviderRequest."""
    image_urls = tuple(dedupe_image_urls(getattr(req, "image_urls", None)))
    if not image_urls:
        return None

    raw_prompt = getattr(req, "prompt", None)
    prompt = str(raw_prompt).strip() if raw_prompt is not None else ""
    recent_context_text = extract_recent_text_context(
        getattr(req, "contexts", None),
        max_messages=config.summary_context_turns,
    )
    session_id = getattr(req, "session_id", None)
    return SummarySnapshot(
        prompt=prompt,
        image_urls=image_urls,
        recent_context_text=recent_context_text,
        session_id=str(session_id) if session_id else None,
    )


def build_summary_user_prompt(snapshot: SummarySnapshot, max_chars: int) -> str:
    """Build the focused prompt sent to the image summary provider."""
    sections = ["请为本轮用户发送的图片生成一段客观、简洁的上下文摘要。"]
    if snapshot.recent_context_text:
        sections.append(f"最近对话参考（仅文本）：\n{snapshot.recent_context_text}")
    user_text = snapshot.prompt or "[图片]"
    sections.append(f"用户本轮文本：{user_text}")
    target_min = min(100, max(40, max_chars // 4))
    target_max = min(max_chars, max(200, max_chars // 2))
    sections.append(
        f"输出限制：约 {target_min}-{target_max} 字，不超过 {max_chars} 字；"
        "只输出图片描述正文，不要回答问题，不要附加标题或解释。"
    )
    return "\n\n".join(sections)


def clean_summary_text(raw_text: Any, max_chars: int) -> str:
    """Strip reasoning tags and normalize/truncate model summary output."""
    text = str(raw_text or "")
    text = _THINK_TAG_RE.sub("", text).strip()
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def format_image_context_block(summary: str) -> str:
    """Wrap the image summary in a clear, provider-neutral context tag."""
    return (
        f"{IMAGE_CONTEXT_OPEN_TAG}\n"
        "[图片上下文摘要，供后续纯文本对话参考]\n"
        f"{summary}\n"
        f"{IMAGE_CONTEXT_CLOSE_TAG}"
    )


def inject_summary_into_run_messages(
    messages: list[Any] | None,
    summary: str,
) -> bool:
    """Attach the hidden image summary as a TextPart on the current turn's user message.

    Writing into run_context.messages inside on_agent_done ensures AstrBot 4.x's
    ConversationHistoryObserver and save_agent_history durably persist the summary.
    Attaching to the user message (rather than inserting a mid-conversation system
    message) keeps Anthropic, Gemini, DeepSeek, and OpenAI providers fully compatible.
    """
    if not messages or not summary.strip():
        return False

    target_msg = _find_current_turn_user_message(messages)
    if target_msg is None:
        return False

    block = format_image_context_block(summary.strip())
    if _message_contains_marker(target_msg, IMAGE_CONTEXT_OPEN_TAG):
        return False

    if isinstance(target_msg, dict):
        return _append_block_to_dict_message(target_msg, block)
    return _append_block_to_object_message(target_msg, block)


def inject_summary_into_legacy_contexts(req: Any, summary: str) -> bool:
    """Legacy fallback for older AstrBot versions without on_agent_done."""
    if req is None or not summary.strip():
        return False
    contexts = getattr(req, "contexts", None)
    if not isinstance(contexts, list):
        return False

    block = format_image_context_block(summary.strip())
    if contexts:
        last = contexts[-1]
        if _message_role(last) == "user":
            if _message_contains_marker(last, IMAGE_CONTEXT_OPEN_TAG):
                return False
            if isinstance(last, dict):
                return _append_block_to_dict_message(last, block)
            return _append_block_to_object_message(last, block)

    contexts.append({"role": "user", "content": block})
    return True


async def call_compat(func: Any, kwargs: dict[str, Any]) -> Any:
    """Call a provider method with only the keyword arguments it declares."""
    signature = inspect.signature(func)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_kwargs:
        result = func(**kwargs)
    else:
        supported = {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
        result = func(**supported)
    if inspect.isawaitable(result):
        return await result
    return result


class ImageSummarizer:
    """Resolves the image summary provider and manages optional concurrent tasks."""

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_SUMMARY_TIMEOUT_SECONDS,
        max_tracked_tasks: int = 256,
    ) -> None:
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._max_tracked_tasks = max(1, max_tracked_tasks)
        self._tasks: dict[str, asyncio.Task[str | None]] = {}

    async def resolve_summary_provider(
        self,
        context: Any,
        config: RouterConfig,
        routed_target_id: str,
    ) -> tuple[str, Any]:
        """Resolve the summary provider, falling back to the routed answering provider."""
        configured_id = config.summary_provider_id
        if configured_id:
            provider = await resolve_provider_by_id(context, configured_id)
            if provider is not None:
                return configured_id, provider
            logger.warning(
                f"{PLUGIN_NAME}: summary_provider_id={configured_id!r} not found; "
                f"fallback to target provider {routed_target_id!r}."
            )

        fallback_id = routed_target_id or config.target_provider_id
        if not fallback_id:
            return "", None
        provider = await resolve_provider_by_id(context, fallback_id)
        if provider is None:
            logger.warning(
                f"{PLUGIN_NAME}: fallback summary provider {fallback_id!r} not found; "
                "skip image summary."
            )
            return "", None
        return fallback_id, provider

    async def generate_summary(
        self,
        context: Any,
        snapshot: SummarySnapshot,
        config: RouterConfig,
        routed_target_id: str,
    ) -> str | None:
        """Call the resolved summary model to produce a hidden image description."""
        provider_id, provider = await self.resolve_summary_provider(
            context, config, routed_target_id
        )
        if not provider_id or provider is None:
            return None

        user_prompt = build_summary_user_prompt(snapshot, config.summary_max_chars)
        try:
            response = await asyncio.wait_for(
                self._invoke_summary_model(
                    context=context,
                    provider_id=provider_id,
                    provider=provider,
                    user_prompt=user_prompt,
                    snapshot=snapshot,
                    system_prompt=config.summary_system_prompt,
                ),
                timeout=self._timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            logger.warning(
                f"{PLUGIN_NAME}: image summary timed out after "
                f"{self._timeout_seconds:.1f}s (provider={provider_id!r})."
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"{PLUGIN_NAME}: image summary failed (provider={provider_id!r}): {exc}",
                exc_info=True,
            )
            return None

        summary = clean_summary_text(
            getattr(response, "completion_text", ""),
            config.summary_max_chars,
        )
        if not summary:
            logger.debug(
                f"{PLUGIN_NAME}: summary provider {provider_id!r} returned empty text."
            )
            return None
        return summary

    def start_background_summary(
        self,
        task_key: str,
        *,
        context: Any,
        snapshot: SummarySnapshot,
        config: RouterConfig,
        routed_target_id: str,
    ) -> asyncio.Task[str | None]:
        """Kick off image summarization concurrently while the answering model runs."""
        self._prune_orphaned_done_tasks()
        existing = self._tasks.pop(task_key, None)
        if existing is not None and not existing.done():
            existing.cancel()

        task: asyncio.Task[str | None] = asyncio.create_task(
            self.generate_summary(context, snapshot, config, routed_target_id),
            name=f"{PLUGIN_NAME}:summary:{task_key}",
        )
        self._tasks[task_key] = task
        task.add_done_callback(self._consume_task_exception)
        return task

    async def await_or_generate(
        self,
        task_key: str | None,
        *,
        context: Any,
        snapshot: SummarySnapshot,
        config: RouterConfig,
        routed_target_id: str,
    ) -> str | None:
        """Await an already-started background summary task, or run synchronously."""
        task = self._tasks.pop(task_key, None) if task_key else None
        if task is not None:
            try:
                return await task
            except asyncio.CancelledError:
                return None
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    f"{PLUGIN_NAME}: background summary task failed: {exc}",
                    exc_info=True,
                )
                return None

        return await self.generate_summary(context, snapshot, config, routed_target_id)

    def cancel_task(self, task_key: str | None) -> None:
        """Cancel and discard a single pending summary task."""
        if not task_key:
            return
        task = self._tasks.pop(task_key, None)
        if task is not None and not task.done():
            task.cancel()

    async def cancel_all(self) -> None:
        """Cancel all active background tasks during plugin teardown."""
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def active_task_count(self) -> int:
        return len(self._tasks)

    def _prune_orphaned_done_tasks(self) -> None:
        if len(self._tasks) < self._max_tracked_tasks:
            return
        for key in list(self._tasks):
            if len(self._tasks) < self._max_tracked_tasks:
                break
            if self._tasks[key].done():
                self._tasks.pop(key, None)

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[str | None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.debug(f"{PLUGIN_NAME}: background summary task error: {exc}")

    async def _invoke_summary_model(
        self,
        *,
        context: Any,
        provider_id: str,
        provider: Any,
        user_prompt: str,
        snapshot: SummarySnapshot,
        system_prompt: str,
    ) -> Any:
        # Prefer AstrBot 4.5.7+ Context.llm_generate SDK when available; never
        # forward the answering request's `model` override to the summary provider.
        llm_generate = getattr(context, "llm_generate", None)
        if callable(llm_generate):
            return await llm_generate(
                chat_provider_id=provider_id,
                prompt=user_prompt,
                image_urls=list(snapshot.image_urls),
                system_prompt=system_prompt,
            )

        return await call_compat(
            provider.text_chat,
            {
                "prompt": user_prompt,
                "session_id": snapshot.session_id,
                "image_urls": list(snapshot.image_urls),
                "system_prompt": system_prompt,
            },
        )


def _message_role(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role") or "")
    return str(getattr(msg, "role", "") or "")


def _message_content(msg: Any) -> Any:
    if isinstance(msg, dict):
        return msg.get("content")
    return getattr(msg, "content", None)


def _extract_plain_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if str(item.get("type", "")).lower() == "text":
                    text = str(item.get("text") or "").strip()
                    if text:
                        parts.append(text)
            else:
                part_type = str(getattr(item, "type", "")).lower()
                if part_type == "text":
                    text = str(getattr(item, "text", "") or "").strip()
                    if text:
                        parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _content_has_image_part(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for item in content:
        if isinstance(item, dict):
            if str(item.get("type", "")).lower() in {"image_url", "image"}:
                return True
        else:
            if str(getattr(item, "type", "")).lower() in {"image_url", "image"}:
                return True
    return False


def _find_current_turn_user_message(messages: list[Any]) -> Any:
    fallback_user: Any = None
    for msg in reversed(messages):
        if _message_role(msg) != "user":
            continue
        if fallback_user is None:
            fallback_user = msg
        if _content_has_image_part(_message_content(msg)):
            return msg
    return fallback_user


def _message_contains_marker(msg: Any, marker: str) -> bool:
    content = _message_content(msg)
    if isinstance(content, str):
        return marker in content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if marker in str(item.get("text") or ""):
                    return True
            else:
                if marker in str(getattr(item, "text", "") or ""):
                    return True
    return False


def _append_block_to_dict_message(msg: dict[str, Any], block: str) -> bool:
    content = msg.get("content")
    if isinstance(content, list):
        content.append({"type": "text", "text": block})
        return True
    if isinstance(content, str):
        msg["content"] = f"{content}\n\n{block}" if content.strip() else block
        return True
    if content is None:
        msg["content"] = block
        return True
    return False


def _append_block_to_object_message(msg: Any, block: str) -> bool:
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        if content and isinstance(content[0], dict):
            content.append({"type": "text", "text": block})
        elif _AstrBotTextPart is not None:
            content.append(_AstrBotTextPart(text=block))
        else:
            content.append({"type": "text", "text": block})
        return True
    if isinstance(content, str):
        updated = f"{content}\n\n{block}" if content.strip() else block
        msg.content = updated
        return True
    if content is None:
        msg.content = block
        return True
    return False
