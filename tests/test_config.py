"""Tests for configuration parsing, schema alignment, and metadata integrity."""

from __future__ import annotations

import json
from pathlib import Path

from astrbot_plugin_multimodal_router.config import (
    DEFAULT_CONFIRM_TURNS,
    DEFAULT_SUMMARY_CONTEXT_TURNS,
    DEFAULT_SUMMARY_MAX_CHARS,
    DEFAULT_SUMMARY_SYSTEM_PROMPT,
    RouterConfig,
)

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def test_schema_defaults_align_with_router_config() -> None:
    schema = json.loads((PLUGIN_ROOT / "_conf_schema.json").read_text("utf-8"))
    raw_defaults = {key: item["default"] for key, item in schema.items()}

    parsed = RouterConfig.from_mapping(raw_defaults)

    assert parsed.enabled is True
    assert parsed.target_provider_id == ""
    assert parsed.summary_provider_id == ""
    assert parsed.enable_image_summary is True
    assert parsed.parallel_image_summary is True
    assert parsed.summary_system_prompt == DEFAULT_SUMMARY_SYSTEM_PROMPT
    assert parsed.summary_max_chars == DEFAULT_SUMMARY_MAX_CHARS
    assert parsed.summary_context_turns == DEFAULT_SUMMARY_CONTEXT_TURNS
    assert parsed.route_quoted_images is True
    assert parsed.confirm_turns == DEFAULT_CONFIRM_TURNS
    assert parsed.vision_provider_ids == frozenset()

    assert schema["target_provider_id"]["_special"] == "select_provider"
    assert schema["summary_provider_id"]["_special"] == "select_provider"


def test_dual_model_properties_and_fallback() -> None:
    same_cfg = RouterConfig.from_mapping(
        {
            "target_provider_id": "opus5",
            "summary_provider_id": "",
        }
    )
    assert same_cfg.effective_summary_provider_id == "opus5"
    assert same_cfg.uses_separate_summary_provider is False
    assert same_cfg.should_run_summary_in_parallel is False

    explicit_same_cfg = RouterConfig.from_mapping(
        {
            "target_provider_id": "opus5",
            "summary_provider_id": "opus5",
        }
    )
    assert explicit_same_cfg.uses_separate_summary_provider is False
    assert explicit_same_cfg.should_run_summary_in_parallel is False

    split_cfg = RouterConfig.from_mapping(
        {
            "target_provider_id": "opus5",
            "summary_provider_id": "kimi",
        }
    )
    assert split_cfg.effective_summary_provider_id == "kimi"
    assert split_cfg.uses_separate_summary_provider is True
    assert split_cfg.should_run_summary_in_parallel is True

    serial_split_cfg = RouterConfig.from_mapping(
        {
            "target_provider_id": "opus5",
            "summary_provider_id": "kimi",
            "parallel_image_summary": False,
        }
    )
    assert serial_split_cfg.uses_separate_summary_provider is True
    assert serial_split_cfg.should_run_summary_in_parallel is False


def test_config_coercion_and_clamping() -> None:
    cfg = RouterConfig.from_mapping(
        {
            "enabled": "false",
            "confirm_turns": 0,
            "summary_max_chars": 10,
            "summary_context_turns": -5,
            "vision_provider_ids": "gpt4o, claude-3-5 , ",
        }
    )
    assert cfg.enabled is False
    assert cfg.confirm_turns == 0
    assert cfg.summary_max_chars == 50
    assert cfg.summary_context_turns == 0
    assert cfg.vision_provider_ids == frozenset({"gpt4o", "claude-3-5"})

    bad_cfg = RouterConfig.from_mapping(
        {
            "confirm_turns": "not-an-int",
            "summary_max_chars": 999999,
            "summary_context_turns": 999,
        }
    )
    assert bad_cfg.confirm_turns == DEFAULT_CONFIRM_TURNS
    assert bad_cfg.summary_max_chars == 4000
    assert bad_cfg.summary_context_turns == 20


def test_metadata_yaml_has_required_astrbot_fields() -> None:
    text = (PLUGIN_ROOT / "metadata.yaml").read_text("utf-8")
    assert "name: astrbot_plugin_multimodal_router" in text
    assert "display_name: 多模态临时路由" in text
    assert "version: v2.0.0" in text
    assert "author: Rinyin" in text
