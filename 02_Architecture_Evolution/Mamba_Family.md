# Mamba 家族

## 背景：为什么需要 Transformer 的替代方案

Transformer 的注意力机制复杂度是 $O(n^2)$（$n$ 为序列长度），推理时 KV Cache 随序列线性增长。当上下文窗口推到 100K+ 时，这些问题变得愈发突出。SSM（State Space Model）系列试图在保持强建模能力的同时，将复杂度降到 $O(n)$。

---

## 前身：S4（Structured State Space for Sequences）

S4 的基础是连续时间状态空间方程：

$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t) + Dx(t)$$

离散化后得到递推形式（类似 RNN）：

$$h_k = \bar{A}h_{k-1} + \bar{B}x_k, \quad y_k = Ch_k$$

S4 的关键洞察：这个递推展开后等价于一个全局卷积（$y = x * \bar{K}$，其中 $\bar{K}$ 是由 $\bar{A}, \bar{B}, C$ 决定的卷积核）。因此：

- **训练时**：用卷积模式，$O(n \log n)$（FFT 加速），可以并行
- **推理时**：用递推模式，$O(1)$ 每步（只维护固定大小的隐状态），无需存储完整历史

S4 的问题：$A, B, C$ 矩阵是固定参数，对所有输入 token 一视同仁——这意味着模型无法根据内容动态决定"关注什么、忽略什么"。

---

## Mamba：选择性 SSM

Mamba（Gu & Dao, 2023）的核心改进：**让 $B, C, \Delta$（离散化步长）依赖于输入**。

$$B_k = \text{Linear}(x_k), \quad C_k = \text{Linear}(x_k), \quad \Delta_k = \text{softplus}(\text{Linear}(x_k))$$

这使得状态转移变成输入相关的（input-dependent），模型可以：
- 遇到重要信息时"放大"写入（$\Delta$ 大 → $\bar{B}$ 大）
- 遇到无关信息时"跳过"（$\Delta$ 小 → 隐状态几乎不更新）

这就是"选择性"（Selectivity）的含义——类似于注意力让模型选择关注哪些 token，但通过隐状态递推实现，而非显式的全局交互。

### 训练并行化

引入输入依赖后，参数不再是固定的，无法直接用全局卷积。Mamba 用**并行扫描算法（parallel scan / prefix sum）** 实现训练并行：

- 递推 $h_k = \bar{A}_k h_{k-1} + \bar{B}_k x_k$ 是一个前缀运算，可以用 $O(n)$ 工作量、$O(\log n)$ 深度在 GPU 上并行完成
- Dao 为此写了专门的 CUDA kernel（硬件感知实现），避免了中间状态的显存开销

所以 Mamba 不是"无需卷积"，而是用**并行扫描替代了卷积**作为训练时的并行手段。

### Mamba Block 结构

```
输入 → Linear → 分支 ──→ Conv1d → SiLU → SSM ──→ 乘 ──→ Linear → 输出
                  └──→ SiLU ─────────────────────┘
```

没有注意力、没有 MLP，整个 block 比 Transformer 层更轻量。

---

## Mamba-2

Mamba-2（Dao & Gu, 2024）的核心贡献：建立了 SSM 与注意力的**结构化对偶关系**（Structured State Space Duality, SSD）。

具体地，证明了选择性 SSM 的计算等价于一种特殊的半可分（semiseparable）矩阵乘法，而标准注意力是其中的一个特例。这个理论统一带来了实际好处：

- 可以复用矩阵乘法硬件原语（tensor core），训练速度比 Mamba-1 快 2-8x
- 支持更大的状态维度：Mamba-1 受限于静态随机存取存储器（SRAM）大小，Mamba-2 通过分块计算突破了这个限制
- 架构上引入了多头模式（类似 MHA/GQA 的 head 结构）

---

## 基于 Mamba 的大模型

| 类型 | 代表模型 | 思路 |
|------|---------|------|
| 纯 Mamba | Mamba-3B、FalconMamba-7B | 全部用 Mamba block 替代 Transformer layer |
| Mamba + Transformer 混合 | Jamba（AI21, 2024）、Zamba | 交替堆叠 Mamba 层和注意力层，兼顾效率和召回能力 |
| Mamba + MoE | Jamba | Mamba 层中嵌入 MoE，进一步扩大容量 |

混合架构的动机：纯 Mamba 在需要精确回溯长距离信息的任务上（如多跳问答、精确复制）弱于注意力，穿插少量注意力层可以补足这一短板。

---

## Mamba 的局限

1. **精确信息检索能力弱**
   - 隐状态是固定大小的压缩表征，本质上是有损的。当任务需要从很长的上下文中精确提取某个特定细节时（如"第 3 段第 2 句说了什么"），Mamba 不如注意力的显式 token 级访问
   - 这不完全等同于"推理能力弱"——Mamba 在很多推理任务上表现不差，它弱的是**需要精确回溯的任务**

2. **Scaling 验证不充分**
   - 目前最大的纯 Mamba 模型在 7B 级别，没有经过 70B+ 规模的充分验证
   - 不是"参数规模难以扩大"（技术上没有障碍），而是目前缺乏足够的大规模实验来证明其 scaling 曲线与 Transformer 一样好

3. **生态和工程成熟度差距**
   - Transformer 有多年的推理优化积累（vLLM、TensorRT-LLM、连续批处理、投机解码等），Mamba 的 serving 基础设施还在早期

---

## 参考资料

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060)
- [Efficiently Modeling Long Sequences with Structured State Spaces - S4 (Gu et al., 2022)](https://arxiv.org/abs/2111.00396)
- [Jamba: A Hybrid Transformer-Mamba Language Model (Lieber et al., 2024)](https://arxiv.org/abs/2403.19887)