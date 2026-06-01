# Agent 概述

## 什么是 Agent

Agent 是一个以 LLM 为核心，具备**感知理解、自主规划、存储记忆、使用工具、自主执行**能力的系统。

它将 LLM 的语言理解与生成能力，同规划、记忆和工具使用等关键模块相结合，实现了超越传统大模型的自主性和复杂任务处理能力。

> **核心公式：Agent = Harness + LLM**
>
> Harness 是一种执行框架，它让 LLM 能够接触真实环境并调用工具完成任务。

**示例：**

| 类别 | 产品 |
|------|------|
| Agent | Codex、Claude Code、Gemini Code Assist |
| Harness | OpenClaw |

---

## Agent 的核心组件

| 组件 | 职责 |
|------|------|
| LLM（大脑） | 语言理解、推理、生成 |
| Planning（规划） | 任务分解、制定执行策略 |
| Memory（记忆） | 短期上下文 + 长期知识存储 |
| Tools（工具） | 调用外部 API、执行代码、访问数据库等 |
| Perception（感知） | 接收用户输入与环境反馈 |

---

## Agent 的类型划分

### 按自主程度分类

| 类型 | 特点 | 示例 |
|------|------|------|
| Copilot 型 | 人机协作，辅助完成任务，需人类确认 | GitHub Copilot、Cursor |
| Autonomous 型 | 高度自主，端到端完成复杂任务 | Codex、Devin |

### 按架构模式分类

| 类型 | 特点 |
|------|------|
| Single Agent | 单一 LLM 驱动，顺序执行任务 |
| Multi-Agent | 多个 Agent 协作，各司其职 |
| Hierarchical Agent | 主 Agent 分配任务给子 Agent，层级管理 |

### 按应用领域分类

| 类型 | 代表产品 |
|------|----------|
| Coding Agent | Claude Code、Codex、Cursor |
| Research Agent | Deep Research、Perplexity |
| General Agent | AutoGPT、BabyAGI |
| Domain Agent | 金融分析、医疗诊断等垂直领域 Agent |
