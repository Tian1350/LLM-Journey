# 有监督微调（SFT）

## 什么是 SFT

SFT（Supervised Fine-Tuning）是在预训练模型基础上，用带标注的"输入-输出"数据继续训练，让模型学会遵循指令、适配特定任务或领域。

在大模型的训练流程中，SFT 处于承上启下的位置：

```
预训练（学语言和世界知识）→ SFT（学会遵循指令）→ RLHF/DPO（对齐人类偏好）
```

需要澄清一个常见误解：现代 LLM 的 SFT 不是"针对每个下游任务单独微调"（那是 BERT 时代的做法）。现在的主流是**指令微调（Instruction Tuning）**——用覆盖大量不同任务的指令数据一次性微调，让模型获得通用的指令遵循能力，而不是为每个任务训一个专用模型。

---

## SFT 的训练方式

本质上还是语言模型的下一个 token 预测，但只在**回答部分**计算损失：

```
输入：[Instruction] 把这句话翻译成英文：你好
输出：Hello

训练时：Instruction 部分的 token 不计算 loss（或 mask 掉），
       只对 "Hello" 部分计算交叉熵损失
```

这样模型学的是"给定指令该如何回答"，而不是去拟合指令本身。

---

## SFT 的分类

### 按数据任务覆盖

| 类型 | 做法 | 特点 |
|------|------|------|
| 单任务微调 | 只用某个特定任务的数据 | 任务表现好，但可能损失通用能力（灾难性遗忘） |
| 指令微调 | 用多任务、多样化的指令数据 | 通用指令遵循能力强，现代 LLM 主流 |
| 领域微调 | 用特定领域数据（医疗、法律、代码） | 领域知识强化，需平衡通用能力 |

### 按参数更新范围

| 类型 | 更新参数 | 显存/成本 | 适用 |
|------|---------|----------|------|
| 全参数微调（Full FT） | 所有参数 | 高 | 数据充足、追求最佳效果 |
| 参数高效微调（PEFT） | 少量新增/选定参数 | 低 | 资源受限、快速适配 |

---

## 参数高效微调（PEFT）

全参数微调一个 70B 模型需要存储全部参数的梯度和优化器状态，显存需求巨大。PEFT 只训练一小部分参数，冻结大部分预训练权重。

| 方法 | 核心思路 |
|------|---------|
| **LoRA** | 在权重旁路插入低秩矩阵 $\Delta W = BA$，只训 $A, B$，原权重冻结 |
| **QLoRA** | 在 LoRA 基础上把基座模型量化到 4-bit，进一步省显存 |
| **Adapter** | 在 Transformer 层间插入小型瓶颈网络，只训 adapter |
| **Prefix/Prompt Tuning** | 在输入前添加可学习的虚拟 token，只训这些 token |

LoRA 是目前最主流的 PEFT 方法（详见 [LoRA](./LoRA.md)）。

---

## SFT 的常见问题

1. **灾难性遗忘（Catastrophic Forgetting）**
   - 在窄领域数据上过度微调，模型会丢失预训练习得的通用能力
   - 缓解：混入通用数据、用 PEFT 限制参数变动、控制训练步数

2. **数据质量 > 数据数量**
   - LIMA 论文证明：1000 条高质量指令数据的微调效果可以超过几万条低质量数据
   - SFT 阶段更应该追求数据的多样性和质量，而非堆量

3. **过拟合到表面格式**
   - 如果数据格式单一，模型可能学到的是"格式套路"而非真正的能力
   - 缓解：数据来源和表达方式多样化

---

## 参考资料

- [Finetuned Language Models Are Zero-Shot Learners - FLAN (Wei et al., 2021)](https://arxiv.org/abs/2109.01652)
- [Training language models to follow instructions with human feedback - InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [LIMA: Less Is More for Alignment (Zhou et al., 2023)](https://arxiv.org/abs/2305.11206)
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
