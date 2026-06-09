# Agent 概述

## 什么是 Agent

Agent 是一个以 LLM 为核心，具备**感知理解、自主规划、存储记忆、使用工具、自主执行**能力的系统。

它将 LLM 的语言理解与生成能力，同规划、记忆和工具使用等关键模块相结合，实现了超越传统大模型的自主性和复杂任务处理能力。

> **核心公式：Agent = Harness + LLM**
>
> Harness 是一种执行框架，它让 LLM 能够接触真实环境并调用工具完成任务。

**示例：**

| 类别      | 产品                                   |
|---------|--------------------------------------|
| Agent   | Codex、Claude Code、Gemini Code Assist |
| Harness | OpenClaw                             |

---

## Agent 的核心组件

| 组件             | 职责                   |
|----------------|----------------------|
| LLM（大脑）        | 语言理解、推理、生成           |
| Planning（规划）   | 任务分解、制定执行策略          |
| Memory（记忆）     | 短期上下文 + 长期知识存储       |
| Tools（工具）      | 调用外部 API、执行代码、访问数据库等 |
| Perception（感知） | 接收用户输入与环境反馈          |

---

## Agent 的类型划分

### 按自主程度分类

| 类型           | 特点                | 示例                    |
|--------------|-------------------|-----------------------|
| Copilot 型    | 人机协作，辅助完成任务，需人类确认 | GitHub Copilot、Cursor |
| Autonomous 型 | 高度自主，端到端完成复杂任务    | Codex、Devin           |

### 按架构模式分类

| 类型                 | 特点                        |
|--------------------|---------------------------|
| Single Agent       | 单一 LLM 驱动，顺序执行任务          |
| Multi-Agent        | 多个 Agent 协作，各司其职          |
| Hierarchical Agent | 主 Agent 分配任务给子 Agent，层级管理 |

### 按应用领域分类

| 类型             | 代表产品                     |
|----------------|--------------------------|
| Coding Agent   | Claude Code、Codex、Cursor |
| Research Agent | Deep Research、Perplexity |
| General Agent  | AutoGPT、BabyAGI          |
| Domain Agent   | 金融分析、医疗诊断等垂直领域 Agent     |

---

## Agent 设计范式

Anthropic 在 "Building Effective Agents" 中提出了一个重要区分：大部分所谓的"Agent"实际上是**工作流（Workflow）**——预定义的编排模式；只有真正让 LLM 自主决定控制流的系统才算 Agent。下面按从简单到复杂的顺序梳理各种范式。

### 基础构建块

| 范式 | 做法 | 适用场景 |
|------|------|---------|
| **Prompt Chaining** | 将任务拆成固定的多步，每步输出作为下一步输入，步骤间可插入校验门控 | 任务可分解为明确的串行子步骤（如：生成大纲 → 写正文 → 审校） |
| **Routing** | 一个分类步骤先判断输入类型，再分发到不同的专用处理流程 | 输入多样、不同类别需要不同处理逻辑（如客服分流） |
| **Parallelization** | 同一输入同时送多个 LLM 分支，结果聚合（投票/合并/选最优） | 需要多视角分析、或需要提升可靠性 |

### 核心设计模式

**Tool Use（工具调用）**

LLM 决定调用哪个工具、传什么参数，harness 执行后将结果返回。这是 Agent 与外部世界交互的基础能力。工具可以是 API、代码执行器、数据库查询、文件操作等。

**ReAct（Reasoning + Acting）**

交替进行推理和行动：
```
Thought: 我需要查一下这个函数的定义
Action: grep("function_name", codebase)
Observation: 找到了，在 src/utils.ts 第 42 行
Thought: 看起来参数类型不对，我需要修改...
Action: edit(...)
```

核心特点：每步行动前先用自然语言推理，推理结果指导行动选择；行动的观察结果反馈回推理。这是最基础也是最通用的 Agent 循环模式。

**Planning & Execute**

先让 LLM 生成完整计划（任务列表），再逐步执行，执行过程中可以根据反馈修订计划：

```
Plan: [步骤1, 步骤2, 步骤3, ...]
Execute 步骤1 → 观察结果 → 必要时修订后续计划
Execute 步骤2 → ...
```

与 ReAct 的区别：ReAct 是一步一想，Planning & Execute 是先全局规划再执行。后者更适合复杂长任务，但计划可能因环境变化而失效，需要 replan 机制。

**Reflection（自我反思）**

让 LLM 评估自己的输出质量，发现问题后修正：

```
生成初始输出 → 用另一个 prompt 要求评审/找错 → 根据反馈改进 → 循环直到满意
```

可以是自评（同一个模型换 prompt），也可以是互评（不同模型/角色交叉审查）。本质上是把人类 review 流程自动化。

**Memory（记忆增强）**

- **短期记忆**：当前对话上下文（受窗口限制）
- **长期记忆**：跨对话持久化的信息，通常通过外部存储（文件、数据库、向量库）实现
- 记忆的读写本身也是工具调用——Agent 决定何时写入记忆、何时检索记忆

**Multi-Agent Collaboration**

多个 Agent 协作完成任务，常见模式：

| 模式 | 做法 | 示例 |
|------|------|------|
| 分工型 | 不同 Agent 负责不同子任务，由一个 orchestrator 协调 | 一个写代码、一个写测试、一个做 review |
| 对抗型 | Agent 之间相互质疑和挑战，提升输出质量 | 辩论式推理 |
| 层级型 | 主 Agent 分解任务并分配给子 Agent，汇总结果 | Claude Code 的 sub-agent 模式 |

### 实践中的选择

**从最简单的方案开始**。能用 prompt chaining 解决的不要上 ReAct，能单 Agent 搞定的不要上 Multi-Agent。复杂性是有代价的——更多的 LLM 调用意味着更高延迟、更多出错机会、更难调试。

---

## 参考资料

- [Anthropic - Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Anthropic - Equipping Agents for the Real World with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [LangChain - What is an AI Agent?](https://www.langchain.com/what-is-an-ai-agent)
- [Lilian Weng - LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2023)](https://arxiv.org/abs/2210.03629)
- [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366)
