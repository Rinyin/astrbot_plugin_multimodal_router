"""Shared AstrBot stubs and test factories for multimodal router tests."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "astrbot_plugin_multimodal_router"

if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))


def _ensure_astrbot_stubs() -> None:
    if "astrbot.api" in sys.modules:
        return

    astrbot_pkg = types.ModuleType("astrbot")
    astrbot_pkg.__path__ = []  # type: ignore[attr-defined]
    api_pkg = types.ModuleType("astrbot.api")
    api_pkg.__path__ = []  # type: ignore[attr-defined]
    event_pkg = types.ModuleType("astrbot.api.event")
    event_pkg.__path__ = []  # type: ignore[attr-defined]
    filter_mod = types.ModuleType("astrbot.api.event.filter")
    comp_mod = types.ModuleType("astrbot.api.message_components")
    provider_mod = types.ModuleType("astrbot.api.provider")
    star_mod = types.ModuleType("astrbot.api.star")
    core_pkg = types.ModuleType("astrbot.core")
    core_pkg.__path__ = []  # type: ignore[attr-defined]
    agent_pkg = types.ModuleType("astrbot.core.agent")
    agent_pkg.__path__ = []  # type: ignore[attr-defined]
    message_mod = types.ModuleType("astrbot.core.agent.message")

    class AstrBotConfig(dict):
        pass

    class PermissionType:
        ADMIN = "admin"
        MEMBER = "member"

    class EventMessageType:
        ALL = "all"
        GROUP_MESSAGE = "group"
        PRIVATE_MESSAGE = "private"

    def _decorator_factory(*_args: Any, **_kwargs: Any):
        def _decorator(fn: Any) -> Any:
            return fn

        return _decorator

    filter_mod.PermissionType = PermissionType  # type: ignore[attr-defined]
    filter_mod.EventMessageType = EventMessageType  # type: ignore[attr-defined]
    filter_mod.permission_type = _decorator_factory  # type: ignore[attr-defined]
    filter_mod.command = _decorator_factory  # type: ignore[attr-defined]
    filter_mod.event_message_type = _decorator_factory  # type: ignore[attr-defined]
    filter_mod.on_waiting_llm_request = _decorator_factory  # type: ignore[attr-defined]
    filter_mod.on_llm_request = _decorator_factory  # type: ignore[attr-defined]
    filter_mod.on_llm_response = _decorator_factory  # type: ignore[attr-defined]
    filter_mod.on_agent_begin = _decorator_factory  # type: ignore[attr-defined]
    filter_mod.on_agent_done = _decorator_factory  # type: ignore[attr-defined]

    class Plain:
        def __init__(self, text: str = "") -> None:
            self.text = text

    class Image:
        def __init__(
            self,
            url: str | None = None,
            file: str | None = None,
            path: str | None = None,
        ) -> None:
            self.url = url
            self.file = file
            self.path = path

    class Reply:
        def __init__(self, id: str = "", chain: list[Any] | None = None) -> None:
            self.id = id
            self.chain = chain or []

    comp_mod.Plain = Plain  # type: ignore[attr-defined]
    comp_mod.Image = Image  # type: ignore[attr-defined]
    comp_mod.Reply = Reply  # type: ignore[attr-defined]

    @dataclass
    class ProviderRequest:
        prompt: str = ""
        session_id: str = ""
        image_urls: list[str] = field(default_factory=list)
        contexts: list[Any] = field(default_factory=list)
        system_prompt: str = ""
        model: str | None = None

    @dataclass
    class LLMResponse:
        role: str = "assistant"
        completion_text: str = ""

    provider_mod.ProviderRequest = ProviderRequest  # type: ignore[attr-defined]
    provider_mod.LLMResponse = LLMResponse  # type: ignore[attr-defined]

    class AstrMessageEvent:
        def __init__(
            self,
            segments: list[Any] | None = None,
            *,
            umo: str = "platform:group:session-1",
            is_at_or_wake_command: bool = True,
        ) -> None:
            self.unified_msg_origin = umo
            self.session_id = umo
            self.is_at_or_wake_command = is_at_or_wake_command
            self.message_obj = types.SimpleNamespace(
                message=list(segments or []),
                session_id=umo,
            )
            self._extras: dict[str, Any] = {}

        def get_extra(self, key: str, default: Any = None) -> Any:
            return self._extras.get(key, default)

        def set_extra(self, key: str, value: Any) -> None:
            self._extras[key] = value

        def plain_result(self, text: str) -> str:
            return text

    event_pkg.AstrMessageEvent = AstrMessageEvent  # type: ignore[attr-defined]
    event_pkg.filter = filter_mod  # type: ignore[attr-defined]

    class Context:
        pass

    class Star:
        def __init__(self, context: Context) -> None:
            self.context = context

    star_mod.Context = Context  # type: ignore[attr-defined]
    star_mod.Star = Star  # type: ignore[attr-defined]

    @dataclass
    class TextPart:
        text: str
        type: str = "text"

    @dataclass
    class ImageURLPart:
        image_url: dict[str, Any]
        type: str = "image_url"

    @dataclass
    class Message:
        role: str
        content: str | list[Any] | None = None
        tool_calls: list[Any] | None = None
        tool_call_id: str | None = None

    message_mod.TextPart = TextPart  # type: ignore[attr-defined]
    message_mod.ImageURLPart = ImageURLPart  # type: ignore[attr-defined]
    message_mod.Message = Message  # type: ignore[attr-defined]

    api_pkg.AstrBotConfig = AstrBotConfig  # type: ignore[attr-defined]
    api_pkg.logger = logging.getLogger("astrbot.test")  # type: ignore[attr-defined]
    api_pkg.event = event_pkg  # type: ignore[attr-defined]
    api_pkg.message_components = comp_mod  # type: ignore[attr-defined]
    api_pkg.provider = provider_mod  # type: ignore[attr-defined]
    api_pkg.star = star_mod  # type: ignore[attr-defined]

    astrbot_pkg.api = api_pkg  # type: ignore[attr-defined]
    astrbot_pkg.core = core_pkg  # type: ignore[attr-defined]
    core_pkg.agent = agent_pkg  # type: ignore[attr-defined]
    agent_pkg.message = message_mod  # type: ignore[attr-defined]

    for name, module in (
        ("astrbot", astrbot_pkg),
        ("astrbot.api", api_pkg),
        ("astrbot.api.event", event_pkg),
        ("astrbot.api.event.filter", filter_mod),
        ("astrbot.api.message_components", comp_mod),
        ("astrbot.api.provider", provider_mod),
        ("astrbot.api.star", star_mod),
        ("astrbot.core", core_pkg),
        ("astrbot.core.agent", agent_pkg),
        ("astrbot.core.agent.message", message_mod),
    ):
        sys.modules[name] = module


def _ensure_plugin_package() -> None:
    if PACKAGE_NAME in sys.modules:
        return
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(PLUGIN_ROOT)]  # type: ignore[attr-defined]
    pkg.__package__ = PACKAGE_NAME
    sys.modules[PACKAGE_NAME] = pkg


_ensure_astrbot_stubs()
_ensure_plugin_package()


class FakeProvider:
    """Test double matching AstrBot's Provider interface."""

    def __init__(
        self,
        provider_id: str,
        *,
        model: str = "test-model",
        modalities: list[str] | None = None,
        extra_config: dict[str, Any] | None = None,
        response_text: str = "summary text",
        text_chat_side_effect: Any = None,
        **extra_attrs: Any,
    ) -> None:
        self.provider_config: dict[str, Any] = {
            "id": provider_id,
            **(extra_config or {}),
        }
        if modalities is not None:
            self.provider_config["modalities"] = list(modalities)
        self._model = model
        self._response_text = response_text
        self._text_chat_side_effect = text_chat_side_effect
        self.calls: list[dict[str, Any]] = []
        for key, value in extra_attrs.items():
            setattr(self, key, value)

    def get_model(self) -> str:
        return self._model

    def meta(self) -> Any:
        return types.SimpleNamespace(
            id=self.provider_config["id"],
            model=self._model,
            type="openai_chat_completion",
        )

    async def text_chat(
        self,
        prompt: str | None = None,
        session_id: str | None = None,
        image_urls: list[str] | None = None,
        contexts: list[Any] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> Any:
        from astrbot.api.provider import LLMResponse

        call = {
            "prompt": prompt,
            "session_id": session_id,
            "image_urls": image_urls,
            "contexts": contexts,
            "system_prompt": system_prompt,
            "model": model,
        }
        self.calls.append(call)
        if self._text_chat_side_effect is not None:
            result = self._text_chat_side_effect(call)
            if importlib.util.find_spec("asyncio") and hasattr(result, "__await__"):
                return await result
            return result
        return LLMResponse(role="assistant", completion_text=self._response_text)


class FakeContext:
    """Test double matching AstrBot's Context provider & SDK methods."""

    def __init__(
        self,
        providers: list[Any],
        *,
        using_provider: Any = None,
        use_llm_generate: bool = True,
        use_async_using_provider: bool = True,
        native_caption_provider_id: str = "",
    ) -> None:
        self._providers = list(providers)
        self._by_id = {
            p.provider_config["id"]: p
            for p in providers
            if isinstance(getattr(p, "provider_config", None), dict)
            and p.provider_config.get("id")
        }
        self._using = using_provider
        self._native_caption_provider_id = native_caption_provider_id
        self.llm_generate_calls: list[dict[str, Any]] = []

        if use_llm_generate:
            self.llm_generate = self._llm_generate  # type: ignore[method-assign]
        if use_async_using_provider:
            self.get_using_provider_async = self._get_using_provider_async  # type: ignore[method-assign]

    def get_all_providers(self) -> list[Any]:
        return list(self._providers)

    def get_provider_by_id(self, provider_id: str) -> Any:
        return self._by_id.get(provider_id)

    def get_using_provider(self, umo: str | None = None) -> Any:
        del umo
        return self._using

    async def _get_using_provider_async(self, umo: str | None = None) -> Any:
        del umo
        return self._using

    def get_config(self, umo: str | None = None) -> dict[str, Any]:
        del umo
        return {
            "provider_settings": {
                "default_image_caption_provider_id": self._native_caption_provider_id,
            }
        }

    async def _llm_generate(
        self,
        *,
        chat_provider_id: str,
        prompt: str | None = None,
        image_urls: list[str] | None = None,
        system_prompt: str | None = None,
        contexts: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.llm_generate_calls.append(
            {
                "chat_provider_id": chat_provider_id,
                "prompt": prompt,
                "image_urls": image_urls,
                "system_prompt": system_prompt,
                "contexts": contexts,
                "kwargs": kwargs,
            }
        )
        provider = self.get_provider_by_id(chat_provider_id)
        if provider is None:
            raise KeyError(f"Provider {chat_provider_id} not found")
        return await provider.text_chat(
            prompt=prompt,
            image_urls=image_urls,
            contexts=contexts,
            system_prompt=system_prompt,
        )
