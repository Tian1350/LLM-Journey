# 强化学习与对齐

## 什么是对齐

对齐（Alignment）指让模型的输出符合人类的意图和价值观——不只是"答得对"，还要"答得有用、无害、诚实"（helpful, harmless, honest）。

为什么预训练 + SFT 不够：
- 预训练只让模型学会"预测下一个 token"，不知道什么是"好回答"
- SFT 能教模型模仿示范回答，但人类示范的覆盖面有限，且难以表达"A 比 B 好多少"这种偏好程度
- 强化学习可以用一个**奖励信号**来表达"什么样的输出更好"，让模型在更广的输出空间里优化

---

## RLHF 的标准三阶段

经典的 RLHF（InstructGPT）流程：

```
阶段 1：SFT
  用人工标注的高质量问答数据微调预训练模型，得到初始策略模型

阶段 2：训练奖励模型（Reward Model, RM）
  收集人类对同一问题多个回答的排序偏好 → 训练 RM 学会给回答打分

阶段 3：RL 优化
  用 RM 作为奖励信号，通过 PPO 优化策略模型，使其生成高奖励的回答
```

**奖励模型怎么训练？**

人类标注者对同一 prompt 的多个回答排序（如 A > B > C），RM 学习预测这个偏好。损失函数（Bradley-Terry 模型）：

$$\mathcal{L} = -\log \sigma(r(x, y_w) - r(x, y_l))$$

其中 $y_w$ 是更好的回答，$y_l$ 是更差的回答，$\sigma$ 是 sigmoid。目标是让好回答的分数高于差回答。

**为什么 RL 阶段需要 KL 惩罚？**

PPO 优化时会在奖励中加一项与初始 SFT 模型的 KL 散度惩罚：

$$\text{reward} = r(x, y) - \beta \cdot \text{KL}(\pi_{\text{RL}} \| \pi_{\text{SFT}})$$

原因：如果只追求 RM 高分，模型会找到 RM 的漏洞输出乱码（reward hacking）。KL 惩罚约束模型不要偏离 SFT 模型太远，保持语言能力。

---

## 反馈来源：RLHF vs RLAIF vs RLVR

三者的区别在于**奖励信号从哪来**：

| 方法 | 反馈来源 | 特点 | 适用场景 |
|------|---------|------|---------|
| **RLHF** | 人类标注偏好 | 质量高但标注昂贵、慢、难规模化 | 通用对齐（有用性、安全性） |
| **RLAIF** | AI 模型生成偏好 | 用一个 LLM 按预设原则（Constitution）评判，便宜可规模化 | 替代/补充人类标注，Constitutional AI |
| **RLVR** | 可验证的规则/程序 | 答案对错可自动判定（如数学答案、代码能否通过测试） | 数学、代码、逻辑推理 |

补充说明：
- **RLAIF**（AI Feedback）是 Anthropic Constitutional AI 的核心——用一套"宪法"原则让 AI 自我批评和改进，减少对人类标注的依赖
- **RLVR**（Verifiable Rewards）是近年推理模型（如 DeepSeek-R1、OpenAI o1）的关键。奖励不来自模型打分，而来自客观验证：数学题答案是否正确、代码是否通过单元测试。这避免了 reward hacking，因为奖励是确定性的、无法被"骗"的

---

## 强化学习算法

### TRPO（Trust Region Policy Optimization）

通过限制每次策略更新的步长（信任域），保证更新不会太激进导致性能崩溃。用 KL 散度约束新旧策略的差异。理论严谨但计算复杂（涉及二阶优化），实践中很少直接用。

### PPO（Proximal Policy Optimization）

TRPO 的简化版，用裁剪（clipping）替代复杂的信任域约束，是 RLHF 的经典选择。

核心是裁剪的目标函数：

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t A_t, \ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

其中 $r_t = \pi_{\text{new}}/\pi_{\text{old}}$ 是概率比，$A_t$ 是优势函数。clip 限制概率比在 $[1-\epsilon, 1+\epsilon]$ 内，防止单次更新偏离太远。

PPO 在 RLHF 中需要同时维护 4 个模型：策略模型、奖励模型、价值模型（critic）、参考模型（KL 约束），显存开销大、调参复杂。

### DPO（Direct Preference Optimization）

DPO 的核心洞察：**跳过显式的奖励模型和 RL 过程，直接用偏好数据优化策略模型**。

通过数学推导，DPO 证明了 RLHF 的优化目标可以重写为一个直接在偏好数据上的分类损失：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

优点：
- 不需要训练独立的奖励模型
- 不需要 RL 采样循环，训练像普通监督学习一样稳定
- 显存和工程复杂度大幅降低

代价：
- 偏好数据是离线固定的，不像 PPO 能在线探索新样本
- 对偏好数据的质量和分布更敏感

### 算法对比

| 算法 | 是否需要 RM | 训练稳定性 | 工程复杂度 | 在线/离线 |
|------|-----------|-----------|-----------|----------|
| TRPO | 是 | 高 | 很高 | 在线 |
| PPO | 是 | 中 | 高 | 在线 |
| DPO | 否 | 高 | 低 | 离线 |
| GRPO | 否（用组内相对奖励） | 中 | 中 | 在线 |

补充：**GRPO**（DeepSeek 提出）是 PPO 的变体，去掉了价值模型，用同一 prompt 多个采样的组内平均奖励作为 baseline，显著降低显存开销，是 DeepSeek-R1 等推理模型的训练算法。

---

## 参考资料

- [Training language models to follow instructions with human feedback - InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization - DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [Proximal Policy Optimization Algorithms - PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)](https://arxiv.org/abs/2212.08073)
- [DeepSeekMath: GRPO (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
