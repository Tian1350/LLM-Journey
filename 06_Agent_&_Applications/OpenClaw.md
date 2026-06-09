# OpenClaw 概述

OpenClaw 不是一个大模型，也不是单纯的聊天壳。更准确地说，它是一个自托管的 **Agent Gateway**：把用户常用的聊天入口、Web 控制台、CLI、移动端节点、Agent runtime、工具、插件和 Skills 接在一起，让一个 Agent 可以长期在线地处理个人事务。

如果用一句话概括：

> OpenClaw 负责让 Agent 接触真实环境；LLM 负责理解、推理和生成。

所以它更接近 Agent 系统里的 **Harness / Gateway / Control Plane**，而不是“模型本身”。

---

## 它解决什么问题

普通 ChatGPT 或 Claude 对话通常是这样的：

```text
用户输入 -> 模型回复
```

OpenClaw 想做的是另一件事：

```text
用户从任意渠道发消息
        ↓
OpenClaw Gateway 接收、路由、鉴权、维护会话
        ↓
Agent runtime 调用模型、读取上下文、使用工具
        ↓
执行任务并把结果回到原渠道
```

也就是说，OpenClaw 的重点不是“再造一个聊天窗口”，而是让 Agent 可以从多个入口被唤起，并且能使用本地或远程工具持续做事。

---

## 核心定位

| 角度 | OpenClaw 是什么 |
|---|---|
| 对用户 | 一个可以从聊天软件、网页、CLI 或移动端触达的个人 Agent |
| 对模型 | 一个给 LLM 提供上下文、工具、会话和执行环境的 Harness |
| 对系统 | 一个长期运行的 Gateway 进程，负责连接渠道、节点和 Agent |
| 对开发者 | 一个可扩展的 Agent 平台，可以接插件、工具、Skills 和不同模型提供商 |

这里最容易误解的是“本地运行”。

OpenClaw 可以运行在自己的电脑或服务器上，但这不代表所有推理都在本地完成。它可以接云端模型，也可以接本地模型；真正本地的是 Gateway、配置、会话路由、部分工作区和执行环境。

---

## 主要组成

### Gateway

Gateway 是 OpenClaw 的中枢。它长期运行，负责：

- 连接 WhatsApp、Telegram、Slack、Discord、iMessage、Signal 等消息渠道
- 提供 Web 控制台、CLI、macOS/iOS/Android 节点的接入
- 管理会话、路由、鉴权和连接状态
- 把用户请求交给 Agent runtime
- 把 Agent 输出送回对应渠道

官方文档里强调：Gateway 是 sessions、routing 和 channel connections 的中心。

### Agent runtime

Agent runtime 负责实际执行 Agent 循环：

```text
接收任务 -> 组织上下文 -> 调用模型 -> 选择工具 -> 执行动作 -> 汇总结果
```

这部分决定了 Agent 是否能真正做事，而不是只生成一段回复。

### Tools

Tools 是 Agent 可以调用的动作能力，例如：

- 执行命令
- 读写文件
- 搜索网页
- 操作浏览器
- 发送消息
- 调用外部 API
- 管理会话或子 Agent

工具让 Agent 能够影响真实环境，因此也是最需要权限控制的部分。

### Skills

Skills 不是工具，而是“怎么做事”的说明书。

例如，一个 Skill 可以告诉 Agent 如何写发布说明、如何审查代码、如何按某种格式整理会议纪要。工具负责行动，Skills 负责流程和规范。

### Plugins

Plugins 用来扩展 OpenClaw 的运行能力。它可以添加：

- 新工具
- 新消息渠道
- 新模型提供商
- 新语音或媒体能力
- Hooks
- 打包好的 Skills

如果说 Skills 更像可复用的工作流说明，Plugins 更像真正给系统加功能的安装包。

---

## 一次请求大概怎么走

以“从 Telegram 发消息让 Agent 整理一个项目状态”为例：

1. 用户在 Telegram 发消息。
2. Telegram channel 把消息交给 Gateway。
3. Gateway 检查来源、权限和会话。
4. Gateway 把请求路由给对应 Agent。
5. Agent runtime 组装上下文，选择模型。
6. 模型判断需要读取文件、搜索资料或调用其它工具。
7. 工具执行完成后，结果回到 Agent。
8. Agent 生成总结。
9. Gateway 把回复发回 Telegram。

这条链路说明了 OpenClaw 的价值：它不只是“聊天”，而是把聊天入口变成一个可执行任务的入口。

---

## 和常见概念的区别

| 概念 | 作用 | 和 OpenClaw 的关系 |
|---|---|---|
| LLM | 负责语言理解、推理和生成 | OpenClaw 调用它，但不等于它 |
| Agent | 能规划、调用工具并执行任务的系统 | OpenClaw 承载和运行 Agent |
| Harness | 给 LLM 提供工具、上下文和执行环境 | OpenClaw 可以看作一种 Harness |
| MCP | 连接外部工具和数据源的协议 | OpenClaw 可以接入或使用类似工具层能力 |
| Skills | 可复用的工作流说明 | OpenClaw 用它约束 Agent 怎么做事 |
| Plugins | 扩展系统能力的包 | OpenClaw 用它增加渠道、工具、模型等能力 |

---

## 适合的场景

OpenClaw 更适合这些需求：

- 想把 Agent 接到常用聊天软件里
- 想让个人 Agent 长期在线
- 想在自己的机器或服务器上控制数据和运行环境
- 想把代码、文件、日程、消息、浏览器等能力交给 Agent 编排
- 想通过插件和 Skills 扩展 Agent 的工作方式

不太适合这些场景：

- 只想偶尔问答，不需要工具执行
- 不想维护本地服务或配置
- 不愿意处理权限、密钥、渠道接入和安全风险
- 对 Agent 自动操作真实系统没有明确边界

---

## 需要注意的安全边界

OpenClaw 的能力越强，风险也越实际。因为它可能接触消息、文件、浏览器、命令行、API Key 和第三方服务。

使用时至少要注意：

- 不要把 Gateway 暴露到公网
- 给消息来源设置 allowlist
- 为不同 Agent 限制工具权限
- 对写文件、执行命令、发消息等动作保留确认机制
- 定期检查插件、Skills 和配置文件
- API Key 不要写进公开仓库

这类系统的风险不在于“模型会说错话”这么简单，而在于模型可能通过工具把错误变成真实动作。

---

## 小结

OpenClaw 的核心不是“一个更聪明的聊天机器人”，而是一个自托管的 Agent Gateway。它把消息渠道、会话、工具、模型、节点、插件和 Skills 串起来，让个人 Agent 能够从多个入口被调用，并在受控环境里执行任务。

理解它时可以抓住三句话：

```text
模型负责思考。
工具负责行动。
OpenClaw 负责把入口、上下文、工具和 Agent 执行过程连起来。
```

---

## 参考资料

- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [OpenClaw 官方文档](https://docs.openclaw.ai/)
- [Gateway architecture](https://docs.openclaw.ai/concepts/architecture)
- [Capabilities overview](https://docs.openclaw.ai/tools)
