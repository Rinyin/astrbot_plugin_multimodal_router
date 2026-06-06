from __future__ import annotations

from typing import Any

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register


PLUGIN_NAME = "astrbot_plugin_multimodal_router"

CONFIRM_PROMPT = """
你正在处于多模态路由确认窗口中。回复前请先在内部判断当前请求、
最近上下文和可见图片是否仍需要视觉/多模态模型能力；如果当前模型
不足以处理，请明确提醒需要切换到多模态模型。不要向用户暴露插件实现细节。
""".strip()


@register(
    "multimodal_router",
    "Rinyi",
    "当图片请求遇到非多模态 LLM 时，临时路由到指定多模态 provider。",
    "1.0.0",
)
class MultimodalRouterPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._confirm_remaining: dict[str, int] = {}

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("mmproviders")
    async def list_providers(self, event: AstrMessageEvent):
        """列出当前加载的 LLM providers，便于配置 target_provider_id。"""
        try:
            providers = list(self.context.get_all_providers() or [])
            using = self.context.get_using_provider()
            using_id = self._provider_id(using)

            if not providers:
                yield event.plain_result("当前没有加载 LLM provider。")
                return

            lines = ["当前加载的 LLM providers:"]
            for provider in providers:
                provider_id = self._provider_id(provider)
                name = self._provider_name(provider)
                marker = " *当前使用*" if provider_id and provider_id == using_id else ""
                lines.append(f"- id={provider_id or '<unknown>'}, name={name}{marker}")

            target_id = self._target_provider_id()
            if target_id:
                lines.append(f"\n当前插件 target_provider_id: {target_id}")
            else:
                lines.append("\n当前插件 target_provider_id 未配置。")

            yield event.plain_result("\n".join(lines))
        except Exception as exc:
            logger.error(f"{PLUGIN_NAME}: failed to list providers: {exc}", exc_info=True)
            yield event.plain_result("列出 provider 失败，请查看 AstrBot 日志。")

    @filter.on_llm_request(priority=100)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """Decorate the final request after provider selection has happened."""
        try:
            self._apply_pending_confirmation(event, req)

            if not self._message_has_image(event):
                return
            self._merge_image_urls(event, req)
        except Exception as exc:
            logger.error(f"{PLUGIN_NAME}: on_llm_request failed: {exc}", exc_info=True)

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def prepare_provider_selection(self, event: AstrMessageEvent):
        """Select the multimodal provider before AstrBot builds the LLM request."""
        try:
            if not self._message_has_image(event):
                return

            current_provider = self._current_provider(event)
            if current_provider is None:
                logger.warning(f"{PLUGIN_NAME}: no active LLM provider; skip routing.")
                return

            target_id = self._target_provider_id()
            current_id = self._provider_id(current_provider)
            if target_id and current_id == target_id:
                return

            if self._provider_supports_vision(current_provider):
                return

            if not target_id:
                logger.warning(f"{PLUGIN_NAME}: target_provider_id is empty; skip routing.")
                return

            target_provider = self.context.get_provider_by_id(target_id)
            if target_provider is None:
                logger.error(
                    f"{PLUGIN_NAME}: target provider {target_id!r} was not found. "
                    "Use /mmproviders to inspect loaded provider ids."
                )
                return

            event.set_extra("selected_provider", target_id)
            self._confirm_remaining[self._session_key(event)] = self._confirm_turns()
            logger.info(
                f"{PLUGIN_NAME}: selected provider {target_id!r} for one image request."
            )
        except Exception as exc:
            logger.error(
                f"{PLUGIN_NAME}: prepare_provider_selection failed: {exc}", exc_info=True
            )

    def _target_provider_id(self) -> str:
        return str(self.config.get("target_provider_id", "") or "").strip()

    def _confirm_turns(self) -> int:
        raw_value = self.config.get("confirm_turns", 3)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            logger.warning(
                f"{PLUGIN_NAME}: invalid confirm_turns={raw_value!r}; use default 3."
            )
            return 3
        if value < 1:
            logger.warning(f"{PLUGIN_NAME}: confirm_turns must be >= 1; use default 3.")
            return 3
        return value

    def _vision_provider_ids(self) -> set[str]:
        raw_value = self.config.get("vision_provider_ids", [])
        if isinstance(raw_value, str):
            return {item.strip() for item in raw_value.split(",") if item.strip()}
        if isinstance(raw_value, list):
            return {str(item).strip() for item in raw_value if str(item).strip()}
        return set()

    def _message_has_image(self, event: AstrMessageEvent) -> bool:
        message_obj = getattr(event, "message_obj", None)
        segments = getattr(message_obj, "message", None) or []
        return any(self._is_image_segment(segment) for segment in segments)

    def _is_image_segment(self, segment: Any) -> bool:
        return isinstance(segment, Comp.Image) or segment.__class__.__name__ == "Image"

    def _merge_image_urls(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        if not hasattr(req, "image_urls"):
            return

        image_urls = list(getattr(req, "image_urls", None) or [])
        seen = {str(item) for item in image_urls}

        message_obj = getattr(event, "message_obj", None)
        segments = getattr(message_obj, "message", None) or []
        for segment in segments:
            if not self._is_image_segment(segment):
                continue
            for attr in ("url", "file", "path"):
                value = getattr(segment, attr, None)
                if value and str(value) not in seen:
                    image_urls.append(value)
                    seen.add(str(value))
                    break

        setattr(req, "image_urls", image_urls)

    def _provider_supports_vision(self, provider: Any) -> bool:
        provider_id = self._provider_id(provider)
        if provider_id and provider_id in self._vision_provider_ids():
            return True

        modalities = self._provider_modalities(provider)
        if isinstance(modalities, list):
            return "image" in modalities

        for value in self._provider_capability_values(provider):
            if isinstance(value, bool) and value:
                return True
            if isinstance(value, str) and self._capability_text_mentions_vision(value):
                return True
            if isinstance(value, (list, tuple, set)) and any(
                self._capability_text_mentions_vision(str(item)) for item in value
            ):
                return True
            if isinstance(value, dict) and self._capability_dict_supports_vision(value):
                return True

        return False

    def _provider_modalities(self, provider: Any) -> Any:
        provider_config = getattr(provider, "provider_config", None)
        if isinstance(provider_config, dict):
            return provider_config.get("modalities", None)
        return None

    def _provider_capability_values(self, provider: Any) -> list[Any]:
        values: list[Any] = []
        for attr in (
            "supports_multimodal",
            "support_multimodal",
            "is_multimodal",
            "supports_vision",
            "support_vision",
            "supports_image",
            "support_image",
            "modalities",
            "capabilities",
            "model_capabilities",
            "metadata",
            "meta",
        ):
            if hasattr(provider, attr):
                value = getattr(provider, attr)
                values.append(value() if callable(value) else value)

        for container_attr in ("config", "model_config"):
            container = getattr(provider, container_attr, None)
            if isinstance(container, dict):
                for key in (
                    "supports_multimodal",
                    "supports_vision",
                    "supports_image",
                    "modalities",
                    "capabilities",
                ):
                    if key in container:
                        values.append(container[key])
        return values

    def _capability_dict_supports_vision(self, value: dict[Any, Any]) -> bool:
        for key, item in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in ("vision", "image", "multimodal")):
                if item is True:
                    return True
                if isinstance(item, str) and item.lower() not in ("false", "0", "no"):
                    return True
                if isinstance(item, (list, tuple, set, dict)) and bool(item):
                    return True
            if self._capability_text_mentions_vision(str(item)):
                return True
        return False

    def _capability_text_mentions_vision(self, value: str) -> bool:
        text = value.lower()
        return any(
            token in text
            for token in ("vision", "image", "multimodal", "multi-modal", "vl")
        )

    def _apply_pending_confirmation(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        key = self._session_key(event)
        remaining = self._confirm_remaining.get(key, 0)
        if remaining <= 0:
            self._confirm_remaining.pop(key, None)
            return

        self._append_system_prompt(req, CONFIRM_PROMPT)
        remaining -= 1
        if remaining > 0:
            self._confirm_remaining[key] = remaining
        else:
            self._confirm_remaining.pop(key, None)

    def _append_system_prompt(self, req: ProviderRequest, prompt: str) -> None:
        if not hasattr(req, "system_prompt"):
            logger.warning(f"{PLUGIN_NAME}: ProviderRequest has no system_prompt field.")
            return
        current = getattr(req, "system_prompt", "") or ""
        if prompt in current:
            return
        separator = "\n\n" if current else ""
        setattr(req, "system_prompt", f"{current}{separator}{prompt}")

    def _session_key(self, event: AstrMessageEvent) -> str:
        return str(
            getattr(event, "unified_msg_origin", None)
            or getattr(getattr(event, "message_obj", None), "session_id", None)
            or getattr(event, "session_id", None)
            or "default"
        )

    def _provider_id(self, provider: Any) -> str:
        if provider is None:
            return ""
        for attr in ("id", "provider_id", "provider_name", "name"):
            value = getattr(provider, attr, None)
            if value:
                return str(value)

        metadata = getattr(provider, "metadata", None) or getattr(provider, "meta", None)
        if callable(metadata):
            try:
                metadata = metadata()
            except Exception as exc:
                logger.debug(f"{PLUGIN_NAME}: provider meta() failed: {exc}")
        if isinstance(metadata, dict):
            for key in ("id", "provider_id", "name"):
                value = metadata.get(key)
                if value:
                    return str(value)
        for attr in ("id", "provider_id", "name"):
            value = getattr(metadata, attr, None)
            if value:
                return str(value)

        return ""

    def _current_provider(self, event: AstrMessageEvent) -> Any:
        selected_provider = None
        if hasattr(event, "get_extra"):
            selected_provider = event.get_extra("selected_provider")
        if selected_provider:
            provider = self.context.get_provider_by_id(str(selected_provider))
            if provider is not None:
                return provider

        try:
            return self.context.get_using_provider(
                umo=getattr(event, "unified_msg_origin", None)
            )
        except TypeError:
            return self.context.get_using_provider()

    def _provider_name(self, provider: Any) -> str:
        if provider is None:
            return "<none>"
        for attr in ("display_name", "name", "provider_name", "model"):
            value = getattr(provider, attr, None)
            if value:
                return str(value)
        return provider.__class__.__name__

    async def terminate(self):
        self._confirm_remaining.clear()
