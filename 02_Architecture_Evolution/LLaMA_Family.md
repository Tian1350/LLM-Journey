# LLaMA 家族

LLaMA（Large Language Model Meta AI）是 Meta 于 2023 年发布的开源大模型系列。它的意义不在于提出全新架构，而是把一系列已被验证有效的改进组合在一起，证明了"开源 + 小模型 + 高质量数据"这条路线的可行性。

下面梳理 LLaMA 相对于初代 GPT 架构的关键改进点。

---

## 架构改进一览

| 组件 | GPT-1 / GPT-2 | LLaMA | 改进收益 |
|------|---------------|-------|---------|
| 位置编码 | 可学习绝对位置编码 | RoPE | 相对位置感知，支持长度外推 |
| 激活函数 | GELU | SwiGLU | 更强的表达力，实验涨点稳定 |
| 归一化 | Post-LayerNorm | Pre-RMSNorm | 训练更稳定，计算更快（省去均值计算） |
| 注意力 | MHA | GQA（LLaMA 2 起） | 推理时 KV Cache 显存大幅降低 |
| Bias | 有 bias | 去掉所有 bias | 减少参数，实测无性能损失 |

---

## 训练策略：Chinchilla 定律的实践

Chinchilla（DeepMind, 2022）的核心结论：在固定计算预算下，模型参数量和训练 token 数应按近似 1:20 的比例分配，而非一味堆大模型。

LLaMA 的做法正是这一思路的体现——用 1.4T token 训练 7B/13B 模型，远超同规模模型的典型训练量。结果是 LLaMA-13B 在多项 benchmark 上匹配甚至超过 GPT-3（175B）。

不过需要注意：后续的 LLaMA 2/3 已经不再严格遵循 Chinchilla 比例，而是选择"过训练"（overtrain）——在推理成本固定的前提下，多花训练算力来压榨小模型的性能上限。这说明 Chinchilla 定律给出的是训练效率最优解，不是模型性能最优解。

---

## LLaMA 系列演进

| 版本 | 时间 | 规模 | 关键变化 |
|------|------|------|---------|
| LLaMA 1 | 2023.02 | 7B/13B/33B/65B | 开源权重，奠定架构基线 |
| LLaMA 2 | 2023.07 | 7B/13B/70B | GQA、4K→4K context、RLHF Chat 版本 |
| LLaMA 3 | 2024.04 | 8B/70B | 15T token 训练、128K context、更大词表（128K） |
| LLaMA 3.1 | 2024.07 | 8B/70B/405B | 405B 首次开源对标 GPT-4 级别 |

---

## 小结

LLaMA 的贡献更多在工程整合和开源推动，而非单点创新。它确立的 RoPE + SwiGLU + Pre-RMSNorm + GQA 组合已成为后续开源模型（Qwen、DeepSeek、Mistral 等）的事实标准。

---

## 参考资料

- [LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971)
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288)
- [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)