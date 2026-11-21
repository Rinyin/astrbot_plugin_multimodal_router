"""Provider inspection, vision capability detection, and diagnostics."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger

from .config import PLUGIN_NAME, RouterConfig

_VISION_TOKEN_RE = re.compile(
    r"(?:^|[^a-z0-9])(vision|image|multimodal|multi-modal|vl|vlm)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)

_EXPLICIT_BOOL_ATTRS = (
    "supports_multimodal",
    "support_multimodal",
    "is_multimodal",
    "supports_vision",
    "support_vision",
    "supports_image",
    "support_image",
)

_CAPABILITY_ATTRS = (
    "modalities",
    "capabilities",
    "model_capabilities",
)

_CONTAINER_ATTRS = ("config", "model_config")

_CAPABILITY_DICT_KEYS = (
    *_EXPLICIT_BOOL_ATTRS,
    *_CAPABILITY_ATTRS,
)


@dataclass(frozen=True, slots=True)
class VisionCapability:
    """Resolved vision capability and how it was determined."""

    supported: bool
    source: str
    modalities: tuple[str, ...] | None = None


def get_provider_id(provider: Any) -> str:
    """Resolve the canonical provider ID matching AstrBot's provider registry."""
    if provider is None:
        return ""

    provider_config = getattr(provider, "provider_config", None)
    if isinstance(provider_config, dict):
        cfg_id = provider_config.get("id")
        if cfg_id:
            return str(cfg_id).strip()

    for attr in ("id", "provider_id"):
        value = getattr(provider, attr, None)
        if value:
            return str(value).strip()

    meta_obj = _read_provider_meta(provider)
    if isinstance(meta_obj, dict):
        for key in ("id", "provider_id", "name"):
            value = meta_obj.get(key)
            if value:
                return str(value).strip()
    elif meta_obj is not None:
        for attr in ("id", "provider_id", "name"):
            value = getattr(meta_obj, attr, None)
            if value:
                return str(value).strip()

    for attr in ("provider_name", "name"):
        value = getattr(provider, attr, None)
        if value:
            return str(value).strip()

    return ""


def get_provider_model(provider: Any) -> str:
    """Return a human-readable model or display name for a provider."""
    if provider is None:
        return "<none>"

    get_model = getattr(provider, "get_model", None)
    if callable(get_model):
        try:
            model = get_model()
            if model:
                return str(model).strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"{PLUGIN_NAME}: provider.get_model() failed: {exc}")

    provider_config = getattr(provider, "provider_config", None)
    if isinstance(provider_config, dict):
        for key in ("model_config", "model"):
            value = provider_config.get(key)
            if isinstance(value, dict) and value.get("model"):
                return str(value["model"]).strip()
            if isinstance(value, str) and value.strip():
                return value.strip()

    for attr in ("model_name", "model", "display_name", "name", "provider_name"):
        value = getattr(provider, attr, None)
        if value:
            return str(value).strip()

    return provider.__class__.__name__


def get_provider_modalities(provider: Any) -> list[str] | None:
    """Return provider_config['modalities'] when explicitly configured as a list."""
    provider_config = getattr(provider, "provider_config", None)
    if isinstance(provider_config, dict):
        modalities = provider_config.get("modalities")
        if isinstance(modalities, list):
            return [
                str(item).strip().lower() for item in modalities if str(item).strip()
            ]
    return None


def inspect_vision_capability(
    provider: Any,
    vision_whitelist: frozenset[str] = frozenset(),
) -> VisionCapability:
    """Determine whether a provider supports image inputs.

    Resolution order:
    1. Explicit plugin whitelist (`vision_provider_ids`).
    2. AstrBot canonical `provider_config["modalities"]` list (empty list `[]`
       follows AstrBot core's `supports_modality()` backward-compatibility rule).
    3. Explicit provider capability attributes / config fields.
    4. Conservative word-boundary token matching on capability descriptors.
    """
    if provider is None:
        return VisionCapability(supported=False, source="none")

    provider_id = get_provider_id(provider)
    modalities = get_provider_modalities(provider)
    modalities_tuple = tuple(modalities) if modalities is not None else None

    if provider_id and provider_id in vision_whitelist:
        return VisionCapability(
            supported=True,
            source="whitelist",
            modalities=modalities_tuple,
        )

    if modalities is not None:
        if len(modalities) == 0:
            return VisionCapability(
                supported=True,
                source="modalities_legacy_empty",
                modalities=(),
            )
        return VisionCapability(
            supported="image" in modalities,
            source="modalities",
            modalities=modalities_tuple,
        )

    for value in _collect_capability_values(provider):
        if isinstance(value, bool):
            if value:
                return VisionCapability(supported=True, source="attribute")
            continue
        if isinstance(value, str):
            if _text_mentions_vision(value):
                return VisionCapability(supported=True, source="heuristic")
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            if any(_text_mentions_vision(str(item)) for item in value):
                return VisionCapability(supported=True, source="heuristic")
            continue
        if isinstance(value, dict) and _capability_dict_supports_vision(value):
            return VisionCapability(supported=True, source="attribute")

    return VisionCapability(supported=False, source="none")


def provider_supports_vision(
    provider: Any,
    vision_whitelist: frozenset[str] = frozenset(),
) -> bool:
    """Convenience boolean wrapper around inspect_vision_capability."""
    return inspect_vision_capability(provider, vision_whitelist).supported


async def resolve_provider_by_id(context: Any, provider_id: str) -> Any:
    """Look up a provider by ID, handling both sync and async context helpers."""
    normalized = str(provider_id or "").strip()
    if not normalized:
        return None
    getter = getattr(context, "get_provider_by_id", None)
    if not callable(getter):
        return None
    result = getter(normalized)
    if inspect.isawaitable(result):
        return await result
    return result


async def resolve_current_provider(context: Any, event: Any) -> Any:
    """Resolve the active chat provider for an event, respecting selected_provider."""
    if hasattr(event, "get_extra"):
        selected = event.get_extra("selected_provider")
        if selected:
            provider = await resolve_provider_by_id(context, str(selected))
            if provider is not None:
                return provider

    umo = getattr(event, "unified_msg_origin", None)
    async_getter = getattr(context, "get_using_provider_async", None)
    if callable(async_getter):
        try:
            return await async_getter(umo=umo)
        except TypeError:
            return await async_getter()
        except ValueError as exc:
            logger.warning(f"{PLUGIN_NAME}: get_using_provider_async failed: {exc}")
            return None

    sync_getter = getattr(context, "get_using_provider", None)
    if callable(sync_getter):
        try:
            result = sync_getter(umo=umo)
        except TypeError:
            result = sync_getter()
        except ValueError as exc:
            logger.warning(f"{PLUGIN_NAME}: get_using_provider failed: {exc}")
            return None
        if inspect.isawaitable(result):
            return await result
        return result

    return None


def detect_native_image_caption_provider_id(context: Any, event: Any) -> str:
    """Return AstrBot's native default_image_caption_provider_id if configured."""
    get_config = getattr(context, "get_config", None)
    if not callable(get_config):
        return ""
    umo = getattr(event, "unified_msg_origin", None)
    try:
        cfg = get_config(umo=umo)
    except TypeError:
        try:
            cfg = get_config()
        except Exception:  # noqa: BLE001
            return ""
    except Exception:  # noqa: BLE001
        return ""

    if not isinstance(cfg, dict):
        return ""
    provider_settings = cfg.get("provider_settings")
    if not isinstance(provider_settings, dict):
        return ""
    return str(provider_settings.get("default_image_caption_provider_id") or "").strip()


async def format_providers_report(
    context: Any,
    event: Any,
    config: RouterConfig,
) -> str:
    """Build a human-readable diagnostic report for the /mmproviders admin command."""
    providers = list(context.get_all_providers() or [])
    using_provider = await resolve_current_provider(context, event)
    using_id = get_provider_id(using_provider)

    if not providers:
        return "当前没有加载 LLM provider。"

    effective_summary_id = config.effective_summary_provider_id
    lines = ["当前加载的 LLM providers:"]
    providers_by_id: dict[str, Any] = {}

    for provider in providers:
        pid = get_provider_id(provider)
        if pid:
            providers_by_id[pid] = provider
        model = get_provider_model(provider)
        cap = inspect_vision_capability(provider, config.vision_provider_ids)
        vision_label = _format_vision_label(cap)

        tags: list[str] = []
        if pid and pid == using_id:
            tags.append("当前会话")
        if pid and pid == config.target_provider_id:
            tags.append("临时回答")
        if pid and config.enable_image_summary and pid == effective_summary_id:
            if config.uses_separate_summary_provider:
                tags.append("图片总结")
            else:
                tags.append("图片总结(回退)")

        tag_suffix = f" [{' / '.join(tags)}]" if tags else ""
        lines.append(
            f"- id={pid or '<unknown>'}, model={model}, vision={vision_label}{tag_suffix}"
        )

    lines.append("")
    lines.append("当前插件配置:")
    lines.append(f"- 启用状态: {'已启用' if config.enabled else '已禁用'}")
    lines.append(
        f"- 临时回答模型 (target_provider_id): {config.target_provider_id or '<未配置>'}"
    )
    if not config.enable_image_summary:
        lines.append("- 隐藏图片摘要: 已关闭")
    elif config.uses_separate_summary_provider:
        mode = "并行" if config.should_run_summary_in_parallel else "串行"
        lines.append(
            f"- 图片总结模型 (summary_provider_id): {config.summary_provider_id} ({mode})"
        )
    else:
        fallback_target = config.target_provider_id or "<未配置>"
        lines.append(
            f"- 图片总结模型 (summary_provider_id): <留空，回退使用 {fallback_target}>"
        )
    lines.append(
        f"- 引用图片路由: {'开启' if config.route_quoted_images else '关闭'} | "
        f"确认窗口轮数: {config.confirm_turns}"
    )

    warnings = _collect_config_warnings(context, event, config, providers_by_id)
    if warnings:
        lines.append("")
        lines.append("配置提示:")
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)


def _format_vision_label(cap: VisionCapability) -> str:
    if cap.supported:
        source_map = {
            "whitelist": "是(白名单)",
            "modalities": "是(modalities)",
            "modalities_legacy_empty": "是(modalities未限制)",
            "attribute": "是(能力属性)",
            "heuristic": "是(启发式)",
        }
        return source_map.get(cap.source, "是")
    if cap.source == "modalities":
        return "否(modalities不含image)"
    return "否"


def _collect_config_warnings(
    context: Any,
    event: Any,
    config: RouterConfig,
    providers_by_id: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if not config.target_provider_id:
        warnings.append("尚未配置 target_provider_id，插件不会执行临时路由。")
    else:
        target = providers_by_id.get(config.target_provider_id)
        if target is None:
            warnings.append(
                f"target_provider_id={config.target_provider_id!r} 未在已加载列表中找到。"
            )
        elif not provider_supports_vision(target, config.vision_provider_ids):
            warnings.append(
                f"target_provider_id={config.target_provider_id!r} 未声明 image 模态，"
                "请确认该 provider 在 AstrBot 中已勾选图片输入能力。"
            )

    if config.enable_image_summary and config.summary_provider_id:
        summary_prov = providers_by_id.get(config.summary_provider_id)
        if summary_prov is None:
            warnings.append(
                f"summary_provider_id={config.summary_provider_id!r} 未找到，运行时将回退到 target_provider_id。"
            )
        elif not provider_supports_vision(summary_prov, config.vision_provider_ids):
            warnings.append(
                f"summary_provider_id={config.summary_provider_id!r} 未声明 image 模态，"
                "请确认其支持看图总结。"
            )

    native_caption_id = detect_native_image_caption_provider_id(context, event)
    if native_caption_id:
        warnings.append(
            f"检测到 AstrBot 原生图片转述模型已启用 ({native_caption_id})；"
            "若目标回答模型未勾选 image 模态，原图可能被核心先转述并清空。"
        )
    return warnings


def _read_provider_meta(provider: Any) -> Any:
    for attr in ("meta", "metadata"):
        if not hasattr(provider, attr):
            continue
        raw = getattr(provider, attr)
        if callable(raw):
            try:
                return raw()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"{PLUGIN_NAME}: provider.{attr}() failed: {exc}")
                continue
        if raw is not None:
            return raw
    return None


def _collect_capability_values(provider: Any) -> list[Any]:
    values: list[Any] = []
    for attr in (*_EXPLICIT_BOOL_ATTRS, *_CAPABILITY_ATTRS):
        if not hasattr(provider, attr):
            continue
        raw = getattr(provider, attr)
        if callable(raw):
            try:
                raw = raw()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"{PLUGIN_NAME}: provider.{attr}() failed: {exc}")
                continue
        values.append(raw)

    for container_attr in _CONTAINER_ATTRS:
        container = getattr(provider, container_attr, None)
        if isinstance(container, dict):
            for key in _CAPABILITY_DICT_KEYS:
                if key in container:
                    values.append(container[key])

    meta_obj = _read_provider_meta(provider)
    if isinstance(meta_obj, dict):
        for key in _CAPABILITY_DICT_KEYS:
            if key in meta_obj:
                values.append(meta_obj[key])
    return values


def _capability_dict_supports_vision(value: dict[Any, Any]) -> bool:
    for key, item in value.items():
        key_text = str(key).lower()
        if any(token in key_text for token in ("vision", "image", "multimodal")):
            if item is True:
                return True
            if isinstance(item, str) and item.strip().lower() not in {
                "",
                "false",
                "0",
                "no",
                "none",
            }:
                return True
            if isinstance(item, (list, tuple, set, dict)) and bool(item):
                return True
        if isinstance(item, str) and _text_mentions_vision(item):
            return True
    return False


def _text_mentions_vision(value: str) -> bool:
    return bool(_VISION_TOKEN_RE.search(value))
