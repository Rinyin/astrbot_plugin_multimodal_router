# 多模态临时路由

AstrBot Multimodal Router

当用户消息包含图片，且当前 LLM provider 不具备多模态能力时，本插件尝试仅将本次 LLM 请求路由到配置的多模态 provider。

## 配置

- `target_provider_id`: 目标多模态 provider ID。WebUI 会显示“选择提供商”按钮，也可在 AstrBot 中使用 `/mmproviders` 查看。
- `confirm_turns`: 成功路由后继续注入确认提示的轮数，默认 `3`。
- `vision_provider_ids`: 已知具备图片/视觉能力的 provider ID 白名单。

## 行为

- 插件会在消息预处理阶段为本次请求设置目标多模态 provider。
- 插件会在 `on_llm_request` 阶段补充图片 URL 并注入确认提示。
- 多模态模型完成回答后，插件会额外生成一段隐藏图片描述，并写入会话上下文供后续对话使用。
- 插件不会修改 AstrBot 全局当前 provider。

## 管理员命令

```text
/mmproviders
```

列出当前加载的 LLM providers，并标记当前使用项。

## License

AGPL-3.0
