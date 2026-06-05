# Skills 概述

## 什么是 Skills

Skills 是一种面向 Agent 的**模块化能力封装机制**：把某类可复用任务所需的触发条件、操作流程、领域知识、示例、模板和可执行脚本，组织成一个可被模型按需调用的目录。

在 Claude 生态中，Skills 由 Anthropic 推广并实现；从规范角度看，它也逐渐被抽象为平台无关的 Agent Skills 形式。它的核心价值不是“给模型注入新参数”，而是让 Agent 在合适的任务场景下，**动态加载专门的上下文和工具资源**，从而稳定完成重复性、流程化或领域化任务。

> **核心公式：Skill = 触发描述 + 操作指令 + 可选资源**
>
> `SKILL.md` 负责告诉 Agent “什么时候使用这个 Skill”以及“使用时应该怎么做”；脚本、参考文档、模板和素材等文件只在需要时再被读取或执行。

---

## 为什么需要 Skills

传统 Prompt 往往适合一次性指令，但当任务需要长期复用、多人共享或稳定执行时，会遇到几个问题：

| 问题 | Skills 的解决方式 |
|---|---|
| Prompt 太长，容易污染上下文 | 只在任务匹配时加载相关 Skill |
| 工作流难以复用 | 将流程沉淀为独立目录和 `SKILL.md` |
| 复杂资料一次性塞进上下文成本高 | 通过 references、assets 等文件按需读取 |
| 需要确定性处理 | 通过 scripts 放置可执行脚本，让 Agent 调用工具完成 |
| 团队规范难统一 | 将品牌规范、代码规范、输出模板等封装为共享 Skill |

---

## 标准目录结构

一个 Skill 本质上是一个目录，最小结构只需要 `SKILL.md`：

```text
my-skill/
├── SKILL.md          # 必需：元数据 + 核心指令
├── scripts/          # 可选：可执行脚本，如 Python、Node.js、Bash
├── references/       # 可选：详细规范、API 文档、领域知识
├── assets/           # 可选：模板、图片、数据文件、样式资源
└── examples/         # 可选：输入输出示例或参考样例
```

需要注意的是，`scripts/`、`references/`、`assets/`、`examples/` 都是常见约定，不是每个 Skill 都必须具备的固定目录。实际结构应服务于任务本身：简单 Skill 可以只有一个 `SKILL.md`，复杂 Skill 才需要拆出脚本、模板和参考资料。

---

## `SKILL.md` 的组成

`SKILL.md` 是 Skill 的入口文件，通常由两部分组成：

1. **YAML Frontmatter**

   用于声明元数据，帮助 Agent 判断这个 Skill 是否适合当前任务。

   ```yaml
   ---
   name: code-review
   description: Review code changes, identify bugs and risks, and suggest focused fixes.
   ---
   ```

2. **Markdown 指令正文**

   用于描述 Agent 使用该 Skill 时应遵循的流程、规则和输出要求。

   常见内容包括：

   - 适用场景和不适用场景
   - 分步骤 workflow
   - 输出格式要求
   - 需要优先读取的参考文件
   - 可调用脚本及其参数说明
   - 典型输入输出示例
   - 异常情况和边界条件处理

---

## 工作机制：渐进式上下文披露

Skills 的关键机制是 **Progressive Disclosure（渐进式上下文披露）**。

| 阶段 | 加载内容 | 作用 |
|---|---|---|
| 启动/索引阶段 | Skill 名称和 description | 让 Agent 知道有哪些能力可用 |
| 触发阶段 | 完整 `SKILL.md` | 获取具体任务流程和约束 |
| 执行阶段 | scripts、references、assets 等附加文件 | 只在需要时读取资料或运行脚本 |

这种方式让 Skills 既能保存大量专业资料，又不会在每次对话中都占用上下文窗口。

---

## Skills 与相关概念的区别

| 概念 | 主要作用 | 是否动态加载 | 典型用途 |
|---|---|---|---|
| Prompt | 一次性指令或当前对话要求 | 否 | 临时任务说明 |
| Custom Instructions | 全局偏好和长期行为约束 | 通常常驻 | 语气、偏好、通用规则 |
| Project Knowledge | 项目背景知识 | 通常常驻或项目内可用 | 项目资料、背景文档 |
| MCP | 连接外部系统和实时数据源 | 按工具调用 | 数据库、API、第三方服务 |
| Skills | 特定任务的流程、知识和资源包 | 是 | 代码审查、文档生成、品牌规范、数据处理 |
| Plugins | 更大的分发单元 | 视内容而定 | 打包 Skills、MCP、命令、子 Agent 等能力 |

简而言之：**MCP 负责连接外部世界，Skills 负责告诉 Agent 如何完成某类任务。**

---

## 适合封装为 Skill 的场景

- 任务会重复出现，并且有稳定流程
- 需要遵循固定规范、模板或输出格式
- 需要引用较多领域资料，但不希望每次都塞进 Prompt
- 需要结合脚本完成确定性处理
- 希望在不同项目、不同 Agent 或团队成员之间复用经验

不适合封装为 Skill 的场景：

- 一次性问题
- 没有稳定流程的开放式讨论
- 只需要一句简单 Prompt 就能完成的任务
- 包含敏感凭据、私钥或不可共享数据的操作

---

## 设计一个好 Skill 的原则

1. **聚焦单一工作流**

   一个 Skill 只解决一类明确任务。与其写一个“大而全”的万能 Skill，不如拆成多个边界清晰的小 Skill。

2. **写清楚触发条件**

   `description` 不只是介绍文字，它会影响 Agent 是否调用该 Skill。应明确写出“做什么”和“什么时候用”。

3. **保持入口文件简洁**

   `SKILL.md` 放核心流程和导航信息，详细 API 文档、长示例、大模板应拆到 `references/` 或 `assets/`。

4. **把确定性逻辑交给脚本**

   需要解析文件、生成包、校验格式、转换数据时，优先用 `scripts/` 封装确定性步骤，减少模型自由发挥。

5. **提供成功样例**

   示例可以让 Agent 更准确地理解输出形态，尤其适合报告、邮件、代码审查、结构化数据生成等任务。

6. **注意安全边界**

   不要在 Skill 中硬编码 API Key、密码或私钥；启用第三方 Skill 前应审查其 `SKILL.md` 和脚本内容。

---

## 一个最小示例

```text
release-notes/
└── SKILL.md
```

```markdown
---
name: release-notes
description: Generate concise release notes from merged pull requests, commit messages, or changelog drafts.
---

# Release Notes

Use this skill when the user asks to write or polish release notes.

## Workflow

1. Identify user-facing changes.
2. Group changes into Added, Changed, Fixed, and Deprecated when applicable.
3. Remove internal implementation details unless they affect users.
4. Keep each bullet concise and action-oriented.

## Output

Return Markdown with:

- A short summary paragraph
- Grouped bullet points
- Known breaking changes, if any
```

---

## 小结

Skills 可以理解为 Agent 的“可复用任务包”。它通过 `SKILL.md` 定义触发条件和执行流程，通过可选资源目录承载脚本、文档、模板和示例，并通过渐进式上下文披露降低上下文成本。

它不是普通 Prompt 的简单替代品，也不是 MCP 这样的外部连接协议；更准确地说，Skills 是把可重复的专家经验、操作流程和任务资源结构化，让 Agent 在需要时稳定调用。

## 参考资料

- [Anthropic Engineering](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [Claude Docs - Agent Skills](https://docs.claude.com/en/docs/claude-code/skills)
