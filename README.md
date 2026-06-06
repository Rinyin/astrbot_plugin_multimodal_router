# AstrBot Multimodal Router

当用户消息包含图片，且当前 LLM provider 不具备多模态能力时，本插件尝试仅将本次 LLM 请求路由到配置的多模态 provider。

## 配置

- `target_provider_id`: 目标多模态 provider ID。WebUI 会显示“选择提供商”按钮，也可在 AstrBot 中使用 `/mmproviders` 查看。
- `confirm_turns`: 成功路由后继续注入确认提示的轮数，默认 `3`。
- `vision_provider_ids`: 已知具备图片/视觉能力的 provider ID 白名单。

## 行为

- 插件只在 `on_llm_request` 阶段尝试修改本次请求对象。
- 插件不会修改 AstrBot 全局当前 provider。
- 如果当前 AstrBot 版本的 `ProviderRequest` 没有请求级 provider 覆盖字段，插件会记录错误并跳过路由。

## 管理员命令

```text
/mmproviders
```

列出当前加载的 LLM providers，并标记当前使用项。

## License

AGPL-3.0
