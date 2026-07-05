# Harness（Agent 执行框架）

## 什么是 Harness

在 Agent 语境下，Harness 是**包裹在 LLM 外层、驱动其完成任务的执行框架**。LLM 本身只是一个"输入文本 → 输出文本"的函数，它不能真正读文件、执行命令、调用 API。Harness 负责把 LLM 的文本输出翻译成对真实世界的操作，并把操作结果反馈回 LLM。

$$\text{Agent} = \text{Harness} + \text{LLM}$$

一个直观的类比：LLM 是"大脑"，负责思考和决策；Harness 是"身体和神经系统"，负责感知环境、执行动作、把反馈送回大脑。没有 Harness，再强的模型也只能"纸上谈兵"。

> 注意：这里的 Harness 与软件测试中的 "test harness（测试框架）" 是完全不同的概念，不要混淆。

---

## Harness 的核心职责

### 1. Agent 循环（Agentic Loop）

这是 Harness 最核心的部分——驱动"思考 → 行动 → 观察"的循环：

```
用户输入
   │
   ▼
┌─────────────────────────────────┐
│  1. 组装上下文（历史 + 工具定义）  │
│  2. 调用 LLM，得到输出            │
│  3. 解析输出：是文本回答还是工具调用？│
│  4. 若是工具调用 → 执行 → 得到结果  │
│  5. 把结果拼回上下文，回到步骤 1    │
└─────────────────────────────────┘
   │ （LLM 判断任务完成）
   ▼
返回最终结果
```

### 2. 工具执行（Tool Execution）

- 向 LLM 声明可用工具（名称、参数 schema、用途描述）
- 解析 LLM 请求调用哪个工具、传什么参数
- 实际执行工具（可能通过 MCP 连接外部 Server，或直接调用本地函数）
- 把执行结果格式化后返回给 LLM
- 处理工具执行的错误、超时、权限校验

### 3. 上下文管理（Context Management）

- 维护对话历史和工具调用记录
- 当上下文接近模型窗口上限时，进行压缩/摘要（compaction）
- 决定哪些信息保留、哪些截断
- 管理系统提示、记忆文件等长期上下文

### 4. 权限与安全

- 拦截危险操作（删除文件、执行破坏性命令等）并请求用户确认
- 沙箱隔离、权限白名单/黑名单
- 防止 prompt injection 等攻击

---

## Harness 与相关概念的关系

```
用户
  │
  ▼
Harness ─── Agent 循环 / 上下文管理 / 权限控制
  │
  ├──→ LLM（通过 API 调用，负责推理决策）
  │
  └──→ 工具层
         ├── 本地工具（文件读写、命令执行）
         └── MCP Server（标准化协议接入的外部能力）
```

- **LLM**：推理引擎，只负责根据上下文输出文本/工具调用意图
- **Harness**：调度中枢，把 LLM 的意图变成实际动作
- **MCP**：Harness 连接外部工具的一种标准化协议（详见 [MCP](./MCP.md)）
- **Agent**：Harness + LLM 组成的完整系统

以 Claude Code 为例：它本身就是一个 Harness——提供终端交互界面、驱动 Agent 循环、管理工具调用（Read/Edit/Bash 等）、处理上下文压缩和权限确认，而背后的推理由 Claude 模型通过 API 完成。

---

## 为什么 Harness 是 Agent 能力的关键

同一个 LLM，配上不同质量的 Harness，实际表现可能天差地别。Harness 的设计决定了：

- **工具的丰富度和可靠性**：能做多少种操作，执行有多稳
- **上下文管理的质量**：长任务中会不会"失忆"或被无关信息淹没
- **错误恢复能力**：工具失败、模型输出格式错误时能否优雅处理
- **安全边界**：能否防止模型执行危险操作

换句话说：模型能力决定了 Agent 的**上限**，Harness 工程决定了这个上限能被兑现多少。

---

## 参考资料

- [Anthropic - Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Anthropic - Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2023)](https://arxiv.org/abs/2210.03629)
