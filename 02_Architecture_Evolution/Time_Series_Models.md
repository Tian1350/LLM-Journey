# 时序预测模型分类

## Transformer 与 GRU 的核心区别

这两者不是"简单 vs 复杂"的关系，而是根本的计算范式不同：

| 维度 | GRU（RNN 家族） | Transformer |
|------|----------------|-------------|
| 信息传递方式 | 递归：逐步传递隐状态 $h_t = f(h_{t-1}, x_t)$ | 并行：所有位置同时通过注意力交互 |
| 序列依赖建模 | 通过隐状态链式传递，长距离信息逐步衰减 | 任意两个位置直接计算注意力，距离无关 |
| 计算并行度 | 训练时无法并行（$h_t$ 依赖 $h_{t-1}$） | 训练时完全并行（所有位置同时计算） |
| 复杂度 | 序列长度 $O(n)$，每步 $O(d^2)$ | $O(n^2 \cdot d)$（标准注意力） |
| 门控机制 | 更新门 + 重置门，控制信息的遗忘与写入 | 注意力权重决定信息的聚合比例 |

### 关于"注意力是更复杂的门控"这个类比

这个说法有一定直觉价值但不够准确。GRU 的门控和注意力解决的是不同层次的问题：

- **GRU 门控**：决定"当前时刻的隐状态中，保留多少旧信息、写入多少新信息"——是对**时间步之间信息流**的控制
- **注意力**：决定"当前位置应该从序列中所有其他位置各取多少信息"——是对**全局信息聚合方式**的计算

更准确的理解：注意力是一种**动态的、全局的加权求和**，而门控是一种**局部的、二选一式的信息过滤**。它们不是同一机制的不同复杂度版本。

### 为什么大模型选择了 Transformer

1. **并行训练**：GRU 的递归结构让 GPU 利用率极低，而 Transformer 可以充分利用现代硬件的并行能力，训练吞吐量高出一个数量级
2. **Scaling 表现**：在 10B+ 参数量级，Transformer 的 loss 随参数/数据量下降更平稳（符合 Scaling Law），RNN 系列在大规模下优势不明显
3. **长距离依赖**：GRU 虽然比 vanilla RNN 好，但隐状态仍是固定维度的瓶颈，信息必须压缩进有限的向量；Transformer 通过 KV Cache 保留所有历史 token 的完整表征

---

## 时序预测领域的模型分类

时序预测的模型格局远不止"Transformer-based vs Linear-based"两类。近几年的研究反复证明，架构选择高度依赖任务特性（预测长度、变量数、分布稳定性等），不存在统一的最优架构。

### 1. 传统统计方法

| 方法 | 特点 |
|------|------|
| ARIMA / SARIMA | 线性假设，捕捉趋势和季节性，小数据表现稳健 |
| ETS（指数平滑） | 状态空间模型，适合短期预测 |
| Prophet | Facebook 出品，分解趋势+季节+节假日，工程友好 |

这些方法在单变量、短预测窗口的场景下依然很能打，不应被忽视。

### 2. RNN/CNN-based

| 代表模型 | 核心思路 |
|---------|---------|
| DeepAR | 自回归 RNN + 概率预测（输出分布参数） |
| LSTNet | CNN 提局部模式 + RNN/Skip-RNN 建长依赖 |
| TCN（Temporal Conv） | 因果膨胀卷积，并行度优于 RNN，感受野可控 |
| N-BEATS | 纯全连接，无注意力无递归，堆叠残差块做分解 |

### 3. Transformer-based

| 代表模型 | 关键改进 |
|---------|---------|
| Informer (2021) | ProbSparse Attention 降低复杂度 |
| Autoformer (2021) | 序列分解 + Auto-Correlation 替代点积注意力 |
| FEDformer (2022) | 频域注意力（FFT/小波变换） |
| PatchTST (2023) | 把时间序列切 patch（类似 ViT），通道独立建模 |
| iTransformer (2024) | 倒转维度——注意力作用在变量维度而非时间维度 |

Transformer 在时序领域的争议：2023 年 "Are Transformers Effective for Time Series Forecasting?" (DLinear) 一文指出，简单线性模型在长期预测上可以匹配甚至超越复杂的 Transformer 变体，引发了对注意力必要性的反思。

### 4. Linear/MLP-based

| 代表模型 | 核心思路 |
|---------|---------|
| DLinear (2023) | 分解趋势+季节，各用一个线性层映射 |
| NLinear (2023) | 去除最后一个值的偏移后做线性映射 |
| TSMixer (2023) | 纯 MLP，交替在时间维度和变量维度做混合 |
| TiDE (2023) | Encoder-Decoder 结构但全用 MLP，无注意力 |

这类模型的核心论点：长期预测中，复杂的注意力模式可能是过拟合噪声，简单的线性映射反而更鲁棒。

### 5. State Space Model (SSM)-based

| 代表模型 | 核心思路 |
|---------|---------|
| S4 (2022) | 结构化状态空间模型，对角化加速，建模超长依赖 |
| Mamba (2023) | 选择性 SSM + 硬件感知设计，线性复杂度的序列建模 |
| S-Mamba / TimeMachine | Mamba 在时序预测上的适配 |

SSM 是目前最活跃的新方向之一：复杂度 $O(n)$，兼具 RNN 的推理效率和 Transformer 的训练并行性。

### 6. Foundation Model（预训练大模型）

| 代表模型 | 思路 |
|---------|------|
| TimesFM (Google, 2024) | 大规模时序数据预训练，零样本预测 |
| Lag-Llama (2024) | 基于 LLaMA 架构的时序预训练模型 |
| Chronos (Amazon, 2024) | 将时序量化为 token，用语言模型架构训练 |
| Timer (2024) | 统一的时序预训练 Transformer |

这是最新趋势：借鉴 NLP 的预训练范式，在海量时序数据上训练通用模型，下游零样本或少样本适配。

---

## 参考资料

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Are Transformers Effective for Time Series Forecasting? (Zeng et al., 2023)](https://arxiv.org/abs/2205.13504)
- [A Time Series is Worth 64 Words: PatchTST (Nie et al., 2023)](https://arxiv.org/abs/2211.14730)
- [iTransformer (Liu et al., 2024)](https://arxiv.org/abs/2310.06625)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [Chronos: Learning the Language of Time Series (Ansari et al., 2024)](https://arxiv.org/abs/2403.07815)
- [TimesFM (Das et al., 2024)](https://arxiv.org/abs/2310.10688)
