# Transformer 架构

Transformer 由 Vaswani et al. 在 2017 年的 "Attention Is All You Need" 中提出。整体结构：输入 Embedding + 位置编码 → Encoder 堆叠 → Decoder 堆叠 → 输出头。

---

## 核心机制：Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

计算过程：Q 与 K 做点积得到注意力分数 → 缩放 → softmax 归一化为权重 → 对 V 加权求和。

### 为什么除以 $\sqrt{d_k}$

当 $d_k$ 较大时，点积的数量级随维度增长（假设 Q、K 各分量独立且均值为 0、方差为 1，则 $Q \cdot K$ 的方差为 $d_k$）。除以 $\sqrt{d_k}$ 将方差拉回到 1 附近，防止 softmax 输入过大导致梯度集中在极少数位置（接近 one-hot），训练早期几乎无梯度流动。

### 自注意力（Self-Attention）

Q、K、V 都由同一个输入 $X$ 经不同的线性投影得到：

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

每个 token 与序列中所有 token 计算相关性，建模全局依赖。

### 掩码自注意力（Masked Self-Attention）

在注意力分数矩阵上施加掩码，将特定位置设为 $-\infty$（softmax 后变为 0）：

- **因果掩码**：上三角设为 $-\infty$，确保每个位置只能看到自身及左侧，用于自回归生成
- **填充掩码（Padding Mask）**：屏蔽 padding token，避免其参与注意力计算

### 交叉注意力（Cross-Attention）

Q 来自 Decoder 的输入，K 和 V 来自 Encoder 的输出。这是 Decoder "读取" Encoder 编码信息的通道。

### 多头注意力（Multi-Head Attention）

将 $d_{model}$ 拆成 $h$ 个头，每个头独立做注意力（维度 $d_k = d_{model}/h$），最后拼接再投影。多个头可以关注不同子空间的模式（如句法关系、语义相似性等）。

---

## 位置编码

注意力计算本身是置换不变的（permutation invariant）——打乱 token 顺序，注意力输出只是跟着打乱，不会"发现"顺序变了。位置编码注入序列顺序信息。

原始论文使用固定的正弦/余弦编码：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

后续演进：可学习位置编码（BERT）、旋转位置编码 RoPE（LLaMA）、ALiBi 等。

---

## Embedding 层

将离散的 token ID 映射为 $d_{model}$ 维的稠密向量。本质是一个查找表（lookup table），形状为 $[V \times d_{model}]$，其中 $V$ 是词表大小。

注意：这里不是"线性投影"——线性投影是对连续向量做矩阵乘法；Embedding 是对离散 ID 做索引查表，虽然数学上等价于 one-hot 向量乘权重矩阵，但实现上直接按索引取行。

---

## Encoder

每个 Encoder 层的结构：

```
输入 → LayerNorm → Multi-Head Self-Attention → 残差连接
    → LayerNorm → FFN → 残差连接 → 输出
```

- FFN 通常是两层线性变换 + 激活函数：$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$
- 残差连接让梯度能直接回传，缓解深层网络的训练困难
- 原始论文用 Post-Norm（先 attention 再 norm），后来主流改为 Pre-Norm（先 norm 再 attention），训练更稳定

Encoder 堆叠 $N$ 层（原始论文 $N=6$），最终输出是每个位置的上下文感知表征。

---

## Decoder

每个 Decoder 层比 Encoder 层多一个交叉注意力块：

```
输入 → LayerNorm → Masked Self-Attention → 残差连接
    → LayerNorm → Cross-Attention(Q=self, KV=Encoder输出) → 残差连接
    → LayerNorm → FFN → 残差连接 → 输出
```

Masked Self-Attention 保证自回归性质（生成第 $t$ 个 token 时只能看到前 $t-1$ 个），Cross-Attention 让 Decoder 在每一步都能查询 Encoder 对输入的编码。

---

## 输出头

Decoder 最后一层输出经过线性投影 $[d_{model} \rightarrow V]$ 得到 logits，再接 softmax 得到词表上的概率分布。

在很多实现中，输出头的权重与 Embedding 层共享（weight tying），既减少参数量，又让语义空间保持一致。

---

## 参考资料

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer - Harvard NLP](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
