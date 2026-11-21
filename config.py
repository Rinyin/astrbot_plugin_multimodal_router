"""Configuration parsing and defaults for the multimodal router plugin."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger

PLUGIN_NAME = "astrbot_plugin_multimodal_router"

DEFAULT_CONFIRM_PROMPT = (
    "你正在处于多模态路由确认窗口中。回复前请先在内部判断当前请求、"
    "最近上下文和可见图片摘要是否仍需要视觉/多模态模型直接看原图；"
    "如果当前模型不足以处理，请明确提醒需要切换到多模态模型。不要向用户暴露插件实现细节。"
)

DEFAULT_SUMMARY_SYSTEM_PROMPT = (
    "你负责为对话历史生成图片上下文摘要。请结合已有对话摘要、用户本轮文本"
    "和图片内容，输出一段给后续纯文本语言模型阅读的中文描述。要求：只客观描述图片中"
    "与当前对话可能相关的事实、文字、对象、场景、数据、关系和关键细节；不要回答"
    "用户的问题；不要编造看不见的细节；不要输出标题、前缀或解释。"
)

DEFAULT_SUMMARY_TIMEOUT_SECONDS = 30.0
DEFAULT_CONFIRM_TURNS = 3
DEFAULT_SUMMARY_MAX_CHARS = 800
DEFAULT_SUMMARY_CONTEXT_TURNS = 4
MIN_SUMMARY_MAX_CHARS = 50
MAX_SUMMARY_MAX_CHARS = 4000
MAX_SUMMARY_CONTEXT_TURNS = 20
MAX_CONFIRM_TURNS = 50


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def _parse_int(
    value: Any,
    default: int,
    *,
    min_value: int,
    max_value: int,
    field_name: str,
) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning(
            f"{PLUGIN_NAME}: invalid {field_name}={value!r}; fallback to {default}."
        )
        return default
    if parsed < min_value:
        logger.warning(
            f"{PLUGIN_NAME}: {field_name}={parsed} is below {min_value}; "
            f"clamped to {min_value}."
        )
        return min_value
    if parsed > max_value:
        logger.warning(
            f"{PLUGIN_NAME}: {field_name}={parsed} exceeds {max_value}; "
            f"clamped to {max_value}."
        )
        return max_value
    return parsed


def _parse_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _parse_id_set(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(str(item).strip() for item in value if str(item).strip())
    return frozenset()


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Validated runtime configuration snapshot for MultimodalRouterPlugin."""

    enabled: bool = True
    target_provider_id: str = ""
    summary_provider_id: str = ""
    enable_image_summary: bool = True
    parallel_image_summary: bool = True
    summary_system_prompt: str = DEFAULT_SUMMARY_SYSTEM_PROMPT
    summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS
    summary_context_turns: int = DEFAULT_SUMMARY_CONTEXT_TURNS
    route_quoted_images: bool = True
    confirm_turns: int = DEFAULT_CONFIRM_TURNS
    vision_provider_ids: frozenset[str] = frozenset()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> RouterConfig:
        data = raw if isinstance(raw, Mapping) else {}
        return cls(
            enabled=_parse_bool(data.get("enabled", True), True),
            target_provider_id=_parse_str(data.get("target_provider_id", "")),
            summary_provider_id=_parse_str(data.get("summary_provider_id", "")),
            enable_image_summary=_parse_bool(
                data.get("enable_image_summary", True), True
            ),
            parallel_image_summary=_parse_bool(
                data.get("parallel_image_summary", True), True
            ),
            summary_system_prompt=_parse_str(
                data.get("summary_system_prompt"), DEFAULT_SUMMARY_SYSTEM_PROMPT
            ),
            summary_max_chars=_parse_int(
                data.get("summary_max_chars", DEFAULT_SUMMARY_MAX_CHARS),
                DEFAULT_SUMMARY_MAX_CHARS,
                min_value=MIN_SUMMARY_MAX_CHARS,
                max_value=MAX_SUMMARY_MAX_CHARS,
                field_name="summary_max_chars",
            ),
            summary_context_turns=_parse_int(
                data.get("summary_context_turns", DEFAULT_SUMMARY_CONTEXT_TURNS),
                DEFAULT_SUMMARY_CONTEXT_TURNS,
                min_value=0,
                max_value=MAX_SUMMARY_CONTEXT_TURNS,
                field_name="summary_context_turns",
            ),
            route_quoted_images=_parse_bool(
                data.get("route_quoted_images", True), True
            ),
            confirm_turns=_parse_int(
                data.get("confirm_turns", DEFAULT_CONFIRM_TURNS),
                DEFAULT_CONFIRM_TURNS,
                min_value=0,
                max_value=MAX_CONFIRM_TURNS,
                field_name="confirm_turns",
            ),
            vision_provider_ids=_parse_id_set(data.get("vision_provider_ids", ())),
        )

    @property
    def effective_summary_provider_id(self) -> str:
        """Configured summary provider ID, falling back to the routed answering provider."""
        return self.summary_provider_id or self.target_provider_id

    @property
    def uses_separate_summary_provider(self) -> bool:
        """Whether a dedicated summary provider distinct from target_provider_id is set."""
        return bool(
            self.summary_provider_id
            and self.summary_provider_id != self.target_provider_id
        )

    @property
    def should_run_summary_in_parallel(self) -> bool:
        """Only run summary concurrently when enabled and using a distinct provider."""
        return (
            self.enable_image_summary
            and self.parallel_image_summary
            and self.uses_separate_summary_provider
        )
