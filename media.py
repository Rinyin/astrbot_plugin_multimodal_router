"""Image segment detection and request image URL helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import astrbot.api.message_components as Comp


def is_image_segment(segment: Any) -> bool:
    """Return True if a message segment represents an image component."""
    image_cls = getattr(Comp, "Image", None)
    if isinstance(image_cls, type) and isinstance(segment, image_cls):
        return True
    return type(segment).__name__ == "Image"


def is_reply_segment(segment: Any) -> bool:
    """Return True if a message segment represents a quoted reply component."""
    reply_cls = getattr(Comp, "Reply", None)
    if isinstance(reply_cls, type) and isinstance(segment, reply_cls):
        return True
    return type(segment).__name__ == "Reply"


def iter_image_segments(event: Any, *, include_quoted: bool = True) -> list[Any]:
    """Collect image segments from the message chain and optionally quoted replies."""
    message_obj = getattr(event, "message_obj", None)
    segments = getattr(message_obj, "message", None) or []
    found: list[Any] = []

    for segment in segments:
        if is_image_segment(segment):
            found.append(segment)
        elif include_quoted and is_reply_segment(segment):
            chain = getattr(segment, "chain", None)
            if isinstance(chain, Iterable) and not isinstance(chain, (str, bytes)):
                for quoted in chain:
                    if is_image_segment(quoted):
                        found.append(quoted)

    return found


def message_has_image(event: Any, *, include_quoted: bool = True) -> bool:
    """Return True when the incoming event carries at least one image."""
    return bool(iter_image_segments(event, include_quoted=include_quoted))


def extract_image_ref(segment: Any) -> str | None:
    """Extract the most usable image reference from an Image message segment."""
    for attr in ("path", "url", "file"):
        value = getattr(segment, attr, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def dedupe_image_urls(urls: Iterable[Any] | None) -> list[str]:
    """Normalize and deduplicate image URLs/paths while preserving order."""
    result: list[str] = []
    seen: set[str] = set()
    if not urls:
        return result
    for item in urls:
        if item is None:
            continue
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def backfill_image_urls_if_empty(
    event: Any,
    req: Any,
    *,
    include_quoted: bool = True,
) -> list[str]:
    """Backfill req.image_urls from event segments only when AstrBot left it empty.

    In AstrBot 4.x, prepare_request_images runs before on_llm_request and replaces
    raw platform URLs with downloaded/compressed local file paths. Appending raw
    segment URLs on top of already-prepared req.image_urls would duplicate images.
    """
    if not hasattr(req, "image_urls"):
        return []

    existing = dedupe_image_urls(getattr(req, "image_urls", None))
    if existing:
        req.image_urls = existing
        return existing

    refs: list[str] = []
    seen: set[str] = set()
    for segment in iter_image_segments(event, include_quoted=include_quoted):
        ref = extract_image_ref(segment)
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)

    if refs:
        req.image_urls = refs
    return refs
