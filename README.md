# 多模态临时路由 (`astrbot_plugin_multimodal_router`)

当用户消息（或引用消息）包含图片，且当前会话所使用的 LLM Provider 不具备视觉能力时，本插件仅将本次请求临时路由到指定的多模态模型直接看图回答，而不修改 AstrBot 的全局或会话默认模型。

同时支持**回答模型与图片总结模型分离**：你可以让高能力多模态模型（如 Opus 5）负责直接看图回答，同时指定低成本轻量视觉模型（如 Kimi / Flash）生成隐藏图片摘要并写入当前回合历史，供后续切回纯文本模型（如 DeepSeek）继续追问。

## 工作流程

1. **路由判定**：在 LLM 请求准备前（`on_waiting_llm_request`）检测本轮消息及引用消息是否含图；若当前会话模型不支持图片输入，则仅为本次请求设置 `selected_provider = target_provider_id`。
2. **直接看图回答**：AstrBot 使用 `target_provider_id` 携带原图完成本次回答。
3. **独立轻量总结**：
   - 若开启隐藏图片摘要，使用 `summary_provider_id`（留空则回退到 `target_provider_id`）生成客观简洁的图片上下文摘要；
   - 当总结模型与回答模型为不同 Provider 且开启并行开关时，二者并发执行以降低耗时（主回答失败或取消时自动取消总结任务）。
4. **安全写入历史**：在回合结束前（`on_agent_done`）将摘要以 `<image_context>` 文本段附加到本轮用户消息中，由 AstrBot 原生持久化，兼容 OpenAI / DeepSeek / Anthropic / Gemini 等所有主流 Provider。
5. **后续纯文本追问**：后续对话自动回到原来的纯文本模型，并可在最多 `confirm_turns` 轮内注入视觉确认提示。

## 配置说明

在 AstrBot WebUI 插件配置页中可直接配置（兼容 v1.x 旧配置键）：

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | `bool` | `true` | 是否启用多模态临时路由 |
| `target_provider_id` | `string` | `""` | **临时回答多模态模型**（如 Opus 5），直接看图回答用户问题 |
| `summary_provider_id` | `string` | `""` | **图片总结模型**（如 Kimi / Flash），用于生成隐藏图片摘要；留空则回退到 `target_provider_id` |
| `enable_image_summary` | `bool` | `true` | 是否生成并写入隐藏图片上下文摘要 |
| `parallel_image_summary` | `bool` | `true` | 总结模型与回答模型不同时是否并行生成摘要（两者相同时自动串行） |
| `summary_system_prompt` | `text` | 内置提示词 | 图片总结模型的系统提示词 |
| `summary_max_chars` | `int` | `800` | 单次写入历史的图片摘要最大字符数 |
| `summary_context_turns` | `int` | `4` | 传给总结模型的最近纯文本历史消息条数（自动剔除工具调用与历史图片） |
| `route_quoted_images` | `bool` | `true` | 用户引用/回复含图消息时是否同样触发临时路由 |
| `confirm_turns` | `int` | `3` | 路由完成后，为后续多少轮普通文本请求注入多模态确认提醒；`0` 表示关闭 |
| `vision_provider_ids` | `list` | `[]` | 已知具备视觉能力的 Provider ID 白名单（用于兜底判定当前模型是否已支持视觉） |

## 典型双模型省钱配置示例

假设日常纯文本聊天使用 DeepSeek：

- `target_provider_id`: 选择强视觉多模态模型（如 `opus5`）
- `summary_provider_id`: 选择轻量廉价视觉模型（如 `kimi` / `gemini-flash`）
- `enable_image_summary`: `true`
- `parallel_image_summary`: `true`

效果：
- 纯文本聊天：只调用 DeepSeek；
- 发图提问：Opus 5 直接看原图给出高质量回答，同时 Kimi 并行生成图片摘要写入历史；
- 下一轮纯文本追问图片内容：切回 DeepSeek，借助历史里的 Kimi 摘要继续准确回复，不再消耗 Opus 5。

## 与 AstrBot 原生图片转述的关系

建议关闭 AstrBot 原生的“默认图片转述模型”，或确保 `target_provider_id` 在 AstrBot 模型配置中已勾选 `image` 模态。

- **原生图片转述**：先把图片转成文字描述并清空原图，再交给纯文本模型回答。
- **本插件**：带图时临时切到多模态模型直接看原图回答；同时可用另一款低成本轻量视觉模型生成隐藏图片摘要写入历史，兼顾看图回答质量与后续追问成本。

## 管理员命令

```text
/mmproviders
```

列出当前已加载的 LLM Provider、模型名、视觉能力识别结果与来源，标记当前会话 / 临时回答 / 图片总结模型，并自动检查常见配置问题（例如未找到 Provider、未勾选 `image` 模态或原生图片转述冲突提示）。

## 开发与测试

推荐使用 Python 3.12+，无需启动完整 AstrBot 实例即可运行单元测试：

```bash
pytest
```

## License

本项目使用 [AGPL-3.0](LICENSE) 协议。
