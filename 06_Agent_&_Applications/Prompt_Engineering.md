# Prompt Engineering

## 是什么

Prompt Engineering 是通过设计输入文本来引导 LLM 产生期望输出的技术。说白了就是"怎么问问题才能得到好答案"——但在工程实践中，这比听起来复杂得多。

它不是玄学。LLM 的行为高度依赖输入的措辞、结构和上下文，好的 prompt 和差的 prompt 在同一个模型上的表现差距可以非常大。

---

## 为什么重要

- **零成本调优**：不需要微调、不需要改模型参数，改 prompt 就能改变行为
- **消除歧义**：自然语言天然有多义性，精确的 prompt 把模型的输出空间约束到我们想要的范围
- **激发能力**：模型具备的能力（推理、格式化、角色扮演等）不会自动展现，需要正确的 prompt 来触发

---

## Prompt 的基本构成

一个完整的 prompt 通常包含以下要素（不是每次都需要全部）：

| 要素 | 作用 | 示例 |
|------|------|------|
| **System/Role** | 设定模型的身份和行为边界 | "你是一个资深后端工程师" |
| **Context** | 提供背景信息 | 项目说明、相关代码、领域知识 |
| **Task/Instruction** | 明确要模型做什么 | "重构这个函数，使其支持批量处理" |
| **Examples（Few-shot）** | 通过示例展示期望的输入输出格式 | 给 2-3 个输入→输出的示例对 |
| **Constraints** | 限制输出的格式、长度、风格等 | "用 JSON 格式返回"、"不超过 100 字" |
| **Output format** | 明确期望的输出结构 | "按以下格式回答：\n原因：...\n结论：..." |

---

## 常用技巧

### Few-shot（少样本示例）

在 prompt 中给出几个输入→输出的示例，模型会模仿示例的模式。

适用：输出格式复杂、任务定义不好用文字描述清楚时。示例数量一般 2-5 个够用，太多会浪费上下文窗口。

### Chain-of-Thought（CoT，思维链）

让模型"一步步思考"而不是直接给答案。

```
# 直接回答（效果差）
Q: 一个商店进了 15 箱苹果，每箱 24 个，卖出了 173 个，还剩多少？
A: 187

# CoT（效果好）
Q: ...
A: 让我一步步计算。
   总数：15 × 24 = 360
   卖出：173
   剩余：360 - 173 = 187
```

原理：中间步骤生成的 token 充当了"工作记忆"，让模型在复杂推理中不丢失中间状态。对数学、逻辑、多步推理任务提升显著。

变体：
- **Zero-shot CoT**：只需加"Let's think step by step"就能触发
- **Few-shot CoT**：在示例中展示推理过程
- **Self-consistency**：多次生成 CoT 取多数投票，提升可靠性

### 结构化输出引导

明确指定输出格式可以大幅提升可解析性：

```
请分析以下代码的问题，按这个格式回答：

问题类型：[bug/性能/可读性]
位置：[文件:行号]
描述：[一句话说明]
修复建议：[具体做法]
```

### 角色设定（System Prompt）

给模型一个身份可以约束它的知识范围和表达风格：
- "你是一个 PostgreSQL DBA，用户问的都是数据库相关问题"
- "你是一个代码审查者，只指出 bug，不做风格建议"

### Negative Prompting（负面约束）

告诉模型不要做什么，有时比告诉它做什么更有效：
- "不要解释代码的基础语法"
- "不要给出模糊的建议，给出具体的代码"
- "如果不确定，直接说不知道，不要编造"

---

## 进阶模式

| 模式 | 做法 | 场景 |
|------|------|------|
| **Self-refine** | 生成初始答案 → 要求模型自我批评 → 修正 | 提升输出质量 |
| **Decomposition** | 把复杂问题拆成子问题，逐个解决后合并 | 复杂任务 |
| **Meta-prompting** | 让模型自己设计 prompt | 探索最优提示方式 |
| **Retrieval-augmented** | 检索相关内容后拼入 prompt | 需要外部知识（即 RAG） |

---

## 局限性

Prompt Engineering 不是银弹，清楚它的边界才能用对它：

| 局限 | 具体表现 |
|------|---------|
| **上下文窗口有限** | Few-shot 示例和背景信息都占 token，复杂任务的 prompt 容易撑满窗口。但这个问题随着长上下文模型的发展在逐渐缓解 |
| **脆弱性** | 微小的措辞变化（换个词、调整顺序）可能导致输出质量大幅波动，难以保证稳定性 |
| **跨模型不可迁移** | 在 GPT-4 上调好的 prompt 换到 Claude 可能效果完全不同，每个模型有自己的"脾气" |
| **无法根除幻觉** | prompt 可以降低幻觉频率（如"不确定就说不知道"），但无法从根本上消除——模型仍可能自信地编造事实 |
| **回报递减** | 当模型能力本身是瓶颈时（如复杂数学推理），再怎么优化 prompt 也无法突破模型的能力上限，此时应考虑微调或换更强的模型 |

关于"门槛消失"：随着模型越来越强（指令遵循能力提升、上下文窗口变长），简单任务确实不再需要精心设计 prompt。但对于复杂的系统级 prompt（Agent 系统提示、生产环境的 system prompt 设计），prompt engineering 仍然是核心工程能力。门槛不是在消失，而是在**上移**——从"怎么让模型听懂我"变成了"怎么设计一个鲁棒的多步指令系统"。

---

## 参考资料

- [Anthropic - Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [OpenAI - Prompt Engineering Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
- [Chain-of-Thought Prompting Elicits Reasoning (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)
