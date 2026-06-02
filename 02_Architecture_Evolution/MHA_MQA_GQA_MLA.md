# MHA、MQA、GQA、MLA 注意力机制对比

## 核心问题

自回归大模型在推理时，每生成一个新 token，都需要让当前 token 的 Query 去访问历史 token 的 Key 和 Value。为了避免每一步都重新计算历史 token 的 K/V，推理系统会保存 **KV Cache**。

不同注意力机制的主要差异可以概括为一句话：

> **MHA、MQA、GQA、MLA 都在处理同一个矛盾：多头注意力的表达能力 vs. 推理阶段 KV Cache 的显存占用和内存带宽。**

其中：

- **MHA**：每个 Query Head 都有独立的 K/V Head，表达能力强，但 KV Cache 最大。
- **MQA**：所有 Query Head 共享一组 K/V Head，KV Cache 最小，但可能损失质量。
- **GQA**：多个 Query Head 分成若干组，每组共享一组 K/V Head，是 MHA 和 MQA 的折中。
- **MLA**：不直接缓存完整 K/V，而是缓存压缩后的 latent 表示，再在计算时恢复或等价变换，进一步压缩 KV Cache。

---

## 统一符号

设：

- `h`：Query Head 数量
- `g`：Key/Value Head 数量
- `d_head`：每个 head 的维度
- `d_model = h * d_head`
- `T`：当前序列长度

对于标准注意力：

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{head}}}\right)V
```

在自回归解码中，当前 token 只新增一个 Query，但需要访问历史所有 token 的 K/V：

```text
当前 Q:     [1, h, d_head]
历史 K/V:  [T, g, d_head]
```

所以，`g` 越大，每层每个 token 需要缓存的 K/V 就越多。

---

## 总览对比

| 机制 | Query Head 数 | K/V Head 数 | KV Cache 规模 | 表达能力 | 推理效率 |
|---|---:|---:|---|---|---|
| MHA | `h` | `h` | 最大，约 `2 * h * d_head` | 强 | 慢 |
| MQA | `h` | `1` | 最小，约 `2 * d_head` | 较弱 | 快 |
| GQA | `h` | `g`，且 `1 < g < h` | 中等，约 `2 * g * d_head` | 接近 MHA | 接近 MQA |
| MLA | `h` | 隐式压缩表示 | 通常约 `d_c + d_r` | 强 | 高 |

> 表中 KV Cache 规模是“单层、单 token、忽略 batch 和 dtype”的近似量级。`d_c` 表示 MLA 的压缩 latent 维度，`d_r` 表示解耦 RoPE 部分需要缓存的维度。

---

## MHA：Multi-Head Attention

### 机制

MHA 是 Transformer 原论文中的标准多头注意力。它为每个 head 分别学习独立的投影矩阵：

```math
Q_i = XW_i^Q,\quad K_i = XW_i^K,\quad V_i = XW_i^V
```

每个 head 单独计算注意力：

```math
\text{head}_i = \text{Attention}(Q_i, K_i, V_i)
```

最后将多个 head 拼接后再经过输出投影：

```math
\text{MHA}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
```

### 为什么 MHA 有效

MHA 的关键不是“把 token 物理切成多个部分”，而是**用多组不同的投影矩阵，把同一份隐藏状态映射到多个表示子空间**。

不同 head 可以学习到不同的注意力模式，例如：

- 局部邻近关系
- 长距离依赖
- 语法关系
- 指代关系
- 不同类型的信息聚合方式

但也要注意：并不是每个 head 都一定学到完全不同的功能。实际模型中可能存在冗余 head，这也是后来 MQA、GQA、Head Pruning 等方法能够成立的原因之一。

### KV Cache

MHA 中每个 Query Head 都有自己的 K/V：

```text
h 个 Q head
h 个 K head
h 个 V head
```

因此，每层每个 token 需要缓存：

```text
K: h * d_head
V: h * d_head
总计: 2 * h * d_head = 2 * d_model
```

这带来的问题是：长上下文和大 batch 推理时，KV Cache 会占用大量显存，并且每步解码都要从显存读取大量历史 K/V，容易受到内存带宽限制。

---

## MQA：Multi-Query Attention

### 机制

MQA 保留多个 Query Head，但让所有 Query Head 共享同一组 K/V Head：

```text
h 个 Q head
1 个 K head
1 个 V head
```

也就是说，每个 head 仍然有不同的 Query 投影，但它们查询的是同一份 Key/Value 表示。

### 解决的问题

MQA 主要优化的是**自回归解码阶段的 KV Cache 和内存带宽**。

对于 MHA：

```text
KV Cache ≈ 2 * h * d_head
```

对于 MQA：

```text
KV Cache ≈ 2 * d_head
```

如果 `h = 32`，MQA 的 K/V 缓存量理论上约为 MHA 的 `1/32`。实际加速幅度还取决于硬件、batch size、上下文长度、算子实现和其它模块开销。

### 容易误解的点

MQA 不是“所有头的 Q 共享一个 K 和 V 矩阵”这么简单。更准确地说：

- Query 仍然有多个 head。
- Key 和 Value 只有一个 head，供所有 Query Head 共享。
- 共享的是 K/V 的投影结果和缓存结构，不是把所有 Query 合并成一个 Query。

### 代价

MQA 的代价是 K/V 表示能力变弱。因为所有 Query Head 只能面对同一份 K/V 表示，模型可用的注意力上下文子空间减少，质量可能下降，尤其是在大模型和复杂任务上更明显。

---

## GQA：Grouped-Query Attention

### 机制

GQA 是 MHA 和 MQA 之间的折中。它将 `h` 个 Query Head 分成 `g` 组，每组共享一组 K/V Head：

```text
h 个 Q head
g 个 K head
g 个 V head
其中 1 < g < h
```

例如：

```text
h = 32, g = 8
```

则每 4 个 Query Head 共享 1 组 K/V。

### 与 MHA、MQA 的关系

GQA 可以看作一个连续谱：

```text
g = h  -> MHA
g = 1  -> MQA
1 < g < h -> GQA
```

所以，MHA 和 MQA 都可以看作 GQA 的特殊情况。

### 为什么 GQA 常用

GQA 在现代大模型中非常常见，因为它通常能取得比较好的折中：

- KV Cache 明显小于 MHA
- 推理内存带宽压力明显降低
- 质量通常比 MQA 更接近 MHA
- 实现上比 MLA 更简单

以 `h = 32, g = 8` 为例：

```text
MHA KV Cache ≈ 2 * 32 * d_head
GQA KV Cache ≈ 2 * 8 * d_head
MQA KV Cache ≈ 2 * 1 * d_head
```

GQA 的缓存量约为 MHA 的 `1/4`，但保留了比 MQA 更多的 K/V 表示子空间。

---

## MLA：Multi-Head Latent Attention

### 背景

MLA 由 DeepSeek-V2 引入，目标是进一步降低 KV Cache，同时尽量保留多头注意力的表达能力。

MQA 和 GQA 的思路是减少 K/V Head 数量；MLA 的思路不同：**不直接缓存完整 K/V，而是缓存一个更低维的 latent 表示**。

### 核心机制

MLA 会先把输入隐藏状态压缩成一个低维 latent：

```math
c_t^{KV} = W^{DKV}h_t
```

然后通过升维矩阵恢复出各个 head 所需的 Key 和 Value 内容：

```math
k_t^C = W^{UK}c_t^{KV},\quad v_t^C = W^{UV}c_t^{KV}
```

推理时，MLA 不需要缓存完整的 `k_t^C` 和 `v_t^C`，只需要缓存压缩后的：

```text
c_t^{KV}
```

这就是 MLA 压缩 KV Cache 的主要来源。

### RoPE 为什么要解耦

如果模型使用 RoPE，问题会复杂一些。

RoPE 是位置相关的旋转变换，它作用在 Q/K 上，而且不同位置的旋转不同。这样一来，K 的位置编码部分不容易直接被低秩压缩和矩阵吸收。

DeepSeek-V2 的做法是将 Key 拆成两部分：

```text
K = [内容部分 K^C ; 位置部分 K^R]
```

其中：

- `K^C`：内容部分，可以通过 latent 压缩。
- `K^R`：RoPE 位置部分，单独计算并缓存。

因此，MLA 推理时缓存的不是完整 K/V，而是：

```text
KV Cache ≈ c_t^{KV} + k_t^R
```

这也是常见表述中 `d_c + d_r` 的来源。

### 矩阵吸收的直觉

对于内容部分的注意力打分：

```math
q_i^T k_j^C = q_i^T W_i^{UK} c_j^{KV}
```

利用矩阵结合律，可以把 `W_i^{UK}` 吸收到 Query 侧：

```math
(W_i^{UK})^T q_i
```

这样推理时可以直接用变换后的 Query 去和缓存的 `c_j^{KV}` 做打分，而不必显式恢复完整 Key。

Value 侧也可以和输出投影做类似的等价变换，从而减少显式展开完整 Value 带来的开销。

### 容易误解的点

MLA 不是简单地“把 K 和 V 都分解成升维矩阵和降维矩阵的乘积”。

更准确地说：

- MLA 先把 K/V 相关信息压缩到共享 latent `c^{KV}`。
- 内容 Key 和 Value 可以从 latent 恢复或通过等价矩阵变换参与计算。
- RoPE 相关的 Key 部分需要解耦出来单独处理。
- 推理阶段的核心收益来自缓存 `c^{KV}` 和少量位置向量，而不是缓存完整 K/V。

---

## KV Cache 与复杂度澄清

### KV Cache 降低了什么

KV Cache 的作用是避免每生成一个新 token 时重复计算历史 token 的 K/V 投影。

没有 KV Cache 时：

```text
每一步都要重新计算历史所有 token 的 K/V
```

有 KV Cache 时：

```text
历史 K/V 只算一次，之后直接读取缓存
```

### KV Cache 没有消除什么

KV Cache 并不会让自回归注意力从根本上摆脱随上下文长度增长的计算。

生成第 `t` 个 token 时，当前 Query 仍然要和前面 `t` 个 Key 做注意力打分。因此：

- 单步注意力读取和打分随 `t` 增长
- 生成完整长度 `T` 的序列，总注意力计算仍然近似是 `O(T^2)`
- MQA、GQA、MLA 主要减少的是 KV Cache 大小和读取带宽，而不是把注意力本身变成常数复杂度

---

## 如何记忆

| 机制 | 一句话记忆 |
|---|---|
| MHA | 每个 Q Head 都有自己的 K/V，表达强，缓存大 |
| MQA | 多个 Q Head 共享一组 K/V，缓存小，可能掉质量 |
| GQA | 一组 Q Head 共享一组 K/V，兼顾质量和效率 |
| MLA | 缓存压缩 latent，而不是完整 K/V，进一步压缩缓存 |

---

## 小结

MHA 提供了多头表示能力，但 KV Cache 随 head 数线性增长。MQA 通过共享单组 K/V 极大减少缓存和内存带宽，但可能牺牲表达能力。GQA 通过设置中间数量的 K/V Head，在质量和效率之间取得折中。MLA 则进一步改变缓存对象，把完整 K/V 替换为压缩 latent 和解耦位置向量，是面向长上下文和高吞吐推理的更激进优化。

从工程视角看，理解这些机制时最重要的不是只记名字，而是抓住两个变量：

```text
Query Head 数量是否保留？
Key/Value Cache 到底缓存了什么？
```

只要这两个问题清楚，MHA、MQA、GQA、MLA 的关系就很容易串起来。
