# MCP（Model Context Protocol）

## 是什么

MCP 是 Anthropic 于 2024 年底推出的开放协议，定义了 LLM 应用与外部工具/数据源之间的标准通信方式。可以把它类比为"AI 应用的 USB-C 接口"——不管底层接什么设备，协议层面是统一的。

核心价值：让工具的提供者和 AI 应用的开发者各自独立开发，通过协议对接，而不是每对组合都写一套定制集成。

---

## 架构

MCP 采用 Client-Server 架构，三个角色：

| 角色 | 职责 | 举例 |
|------|------|------|
| **Host（宿主）** | 面向用户的应用，内部管理一个或多个 Client 实例 | Claude Desktop、Cursor、Claude Code |
| **Client（客户端）** | 协议层，负责与某一个 Server 建立连接、收发 JSON-RPC 消息 | Host 内部组件，用户一般不直接接触 |
| **Server（服务端）** | 对外暴露能力的轻量服务，每个 Server 可以提供三类能力 | 文件系统 Server、GitHub Server、数据库 Server |

Server 暴露的三类能力：

- **Tools（工具）**：模型可以主动调用的函数，如"读取文件"、"执行 SQL"、"发送消息"
- **Resources（资源）**：模型可以读取的结构化数据，类似 REST 里的 GET 端点
- **Prompts（提示模板）**：预定义的交互模板，用户或模型可以触发，类似快捷指令

通信方式支持两种传输：

- **stdio**：Server 作为子进程启动，通过标准输入输出通信（本地场景）
- **Streamable HTTP**：通过 HTTP + SSE 通信（远程/云端场景）

---

## MCP 与 Agent 的关系

MCP 本身不是 Agent 框架，它解决的是"Agent 怎么调用外部工具"这个问题。

以 Claude Code 为例说明各层的关系：

```
用户
  │
  ▼
Claude Code（Host + Agent 循环）
  │── 内置 Client ──► 本地文件系统 Server（读写文件、执行命令）
  │── 内置 Client ──► GitHub MCP Server（操作 PR、Issue）
  │── ...
  │
  ▼
Claude 模型（通过 API 调用，负责推理和决策）
```

- **Claude Code** 既是 Host（提供终端交互界面），又承担 Agent 循环（接收模型输出 → 执行工具 → 把结果喂回模型）。
- **Claude 模型** 是推理引擎，通过 API 被调用，它本身不是 MCP 架构中的任何角色。
- **MCP Server** 是能力的提供方，被 Client 连接和调用。

简单说：Agent 决定"做什么"，MCP 解决"怎么连到工具去做"。

---

## 参考资料

- [MCP Official Documentation](https://modelcontextprotocol.io/introduction)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Anthropic - Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
