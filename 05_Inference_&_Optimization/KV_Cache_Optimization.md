# KV Cache Optimization

## 先说结论

KV Cache 是大模型自回归推理里最基础、也最容易被低估的优化：模型生成第 `t` 个 token 时，不再重复计算前 `t-1` 个 token 在每一层 Attention 里的 Key 和 Value，而是把它们缓存下来，下一步直接复用。

它解决的是**重复计算历史 K/V**的问题，不是把 Attention 本身变成常数时间。生成新 token 时，当前 Query 仍然要和历史所有 Key 做注意力计算，所以序列越长，单步解码仍然越慢；同时，缓存本身会随 `batch_size`、层数和上下文长度线性增长，最后变成显存和带宽瓶颈。

一句话概括：

> KV Cache 用显存换计算。短上下文时它几乎是免费加速；长上下文和高并发时，它会从“优化手段”变成“主要瓶颈”。

---

## KV Cache 到底缓存了什么

以 Decoder-only Transformer 为例，每一层 Attention 都会从隐藏状态中投影出：

```text
Q = XWq
K = XWk
V = XWv
```

在自回归生成中，新 token 只能看见自己和它之前的 token。假设 prompt 已经有 `T` 个 token，现在要生成下一个 token：

- 当前 token 的 `Q` 只在这一步使用一次，没有必要缓存；
- 历史 token 的 `K/V` 会被之后每一步反复访问，值得缓存；
- 缓存是逐层保存的，因为每一层的 `K/V` 都不同。

因此 KV Cache 的典型形状可以理解为：

```text
[num_layers, batch_size, seq_len, 2, num_kv_heads, head_dim]
```

其中 `2` 分别代表 Key 和 Value。不同框架的真实布局可能不同，比如为了 CUDA kernel、分页管理或连续读写会调整维度顺序，但容量估算基本遵循这个结构。

---

## Prefill 和 Decode：两个阶段要分开看

LLM 推理通常可以分成两个阶段：

| 阶段 | 输入 | 主要工作 | KV Cache 的角色 |
|---|---|---|---|
| Prefill | 整段 prompt | 并行计算 prompt 的所有 token | 一次性建立初始缓存 |
| Decode | 每次 1 个新 token | 逐步生成后续 token | 读取历史缓存，并追加新 token 的 K/V |

这也是很多性能问题容易讲混的地方。

Prefill 阶段可以并行处理整段 prompt，计算量大，但 GPU 利用率通常比较高。Decode 阶段每次只生成一个 token，矩阵规模小，却要反复读取越来越长的历史 K/V；当 batch 或上下文长度变大时，瓶颈常常从算力转向显存带宽和缓存管理。

---

## 为什么不用缓存会很慢

如果没有 KV Cache，生成每个新 token 时，模型都要把“已有上下文 + 已生成内容”重新跑一遍，重新得到每一层的历史 `K/V`。这会造成大量重复计算。

有了 KV Cache 后：

```text
第 1 步：计算 prompt 的 K/V，写入缓存
第 2 步：只计算新 token 的 K/V，追加到缓存；Q 访问历史 K/V
第 3 步：继续只追加最新 token 的 K/V
...
```

需要注意一个常见误解：

> KV Cache 不会让长上下文 Attention 的开销消失。它省掉的是历史 token 的投影和前向重复计算，但当前 Query 仍然要读取并关注历史 K/V。

所以，在单个 decode step 中，Attention 对历史长度的依赖仍近似是 `O(T)`；只是相比“每一步重算全部历史”，已经少了很多无意义工作。

---

## 显存占用怎么估算

KV Cache 的显存可以粗略估算为：

```text
KV Cache bytes
= batch_size
  * seq_len
  * num_layers
  * 2
  * num_kv_heads
  * head_dim
  * bytes_per_element
```

其中：

- `2` 是因为同时保存 K 和 V；
- `num_kv_heads` 不一定等于 Query heads，GQA/MQA 会减少它；
- `bytes_per_element` 取决于精度，FP16/BF16 通常是 2 bytes，INT8 是 1 byte，INT4 理想情况下约 0.5 byte，但实际还要考虑 scale、zero point 和对齐开销。

举个数量级例子：如果一个模型有 `32` 层、`32` 个 KV heads、`head_dim = 128`，使用 FP16/BF16，那么单个 token 的 KV Cache 大约是：

```text
32 * 2 * 32 * 128 * 2 bytes = 524,288 bytes ≈ 512 KiB
```

这意味着一个 4096 token 的单请求上下文，光 KV Cache 就接近：

```text
512 KiB * 4096 ≈ 2 GiB
```

这还没算模型权重、激活临时空间、CUDA workspace、batch 并发和内存碎片。实际部署时，KV Cache 往往是决定“能开多大 batch、能接多长上下文”的关键因素。

---

## 主要优化方向

KV Cache 优化不是单一技术，而是一组围绕“少存、少读、少浪费”的方法。

### 1. 减少 KV heads：MQA 和 GQA

标准 Multi-Head Attention 中，每个 Query head 都有自己的 Key/Value head，因此缓存规模和 head 数成正比。

Multi-Query Attention（MQA）让所有 Query heads 共享同一组 K/V。这样 KV Cache 可以按 head 数大幅下降，但因为 K/V 表示能力变弱，模型质量可能受影响。

Grouped-Query Attention（GQA）是 MHA 和 MQA 之间的折中：把 Query heads 分组，每组共享一组 K/V。它通常能保留接近 MHA 的效果，同时显著降低缓存和带宽压力。

可以把三者关系理解成：

| 机制 | Query heads | KV heads | KV Cache | 典型取舍 |
|---|---:|---:|---|---|
| MHA | `h` | `h` | 最大 | 表达能力强，推理成本高 |
| MQA | `h` | `1` | 最小 | 推理快，但可能损失质量 |
| GQA | `h` | `g` | 居中 | 常用折中方案 |

这里的关键不是“少几个参数”这么简单，而是 decode 时每一步要从显存读取的历史 K/V 变少了。对长上下文和高并发服务来说，这通常比单纯减少计算量更重要。

### 2. KV Cache 量化

KV Cache 默认常用 FP16/BF16 保存。把 K/V 量化到 INT8、INT4 或 FP8，可以直接降低显存占用，也能减少读写带宽。

但 KV Cache 量化比权重量化更敏感，因为缓存内容是动态生成的，分布会随输入、层数和时间变化。常见做法包括：

- per-token 或 per-channel scaling，避免少数异常值拉大整体量化范围；
- K 和 V 分开量化，因为二者分布不完全相同；
- 保留部分层或特殊位置为高精度，降低质量损失；
- 在服务框架里把量化格式和 Attention kernel 一起设计，避免反量化开销抵消收益。

量化的核心判断标准不是“显存少了多少”，而是端到端是否真的提升吞吐或降低延迟。如果每一步都要昂贵地反量化，收益会被吃掉。

### 3. 分页管理：PagedAttention

传统做法通常给每个请求预留一段连续 KV Cache 空间。但真实服务里，请求长度差异很大，有的很快结束，有的不断生成；如果按最大长度预留，显存会被大量浪费。

PagedAttention 借鉴操作系统分页思想，把 KV Cache 切成固定大小的 block。逻辑上每个请求仍然拥有连续上下文，物理显存里却可以分散存放。这样做的好处是：

- 减少预分配和内存碎片造成的浪费；
- 请求结束后可以按 block 回收；
- 多个请求共享相同 prefix 时，可以复用已有 block；
- 更容易支持连续批处理和动态调度。

这类优化不改变模型数学形式，但会明显影响服务吞吐。vLLM 的核心贡献之一就是围绕 PagedAttention 做高效 KV Cache 管理。

### 4. Prefix Cache 共享

很多线上请求有共同前缀，比如系统提示词、工具说明、RAG 模板或多轮对话的固定开头。Prefix caching 会把这些公共前缀的 KV Cache 复用起来，避免每个请求重复 prefill。

它适合：

- 固定 system prompt；
- agent 工具说明很长；
- 批量请求共享同一段上下文；
- RAG 模板稳定、变化部分靠后。

限制也很明显：只要前缀 token 不完全一致，缓存就不能直接共享。实际工程里通常要配合 tokenizer、prompt 模板和缓存 key 设计，否则命中率会很低。

### 5. 滑动窗口、淘汰和压缩

当上下文越来越长时，完整保留所有历史 K/V 并不总是划算。一些模型或系统会选择只保留最近窗口，或者根据重要性淘汰部分 token 的缓存。

这类方法的风险比 MQA/GQA 和分页管理更高，因为它可能改变模型实际可见的上下文。常见思路包括：

- sliding window attention：只看最近 `W` 个 token；
- attention sink：保留少量早期关键 token，再滑动保留最近 token；
- token eviction：根据注意力分数或启发式指标丢弃低价值 token；
- KV compression：对历史 K/V 做低秩、聚类或量化压缩。

它们适合超长上下文或资源受限场景，但需要用真实任务评估，不能只看 perplexity 或单轮 benchmark。

### 6. Offloading：把缓存放到 CPU 或磁盘

当 GPU 显存不够时，可以把一部分 KV Cache 移到 CPU 内存，甚至更慢的存储层级。这样能扩大可支持的上下文或并发，但代价是 PCIe/NVLink 传输延迟。

Offloading 更适合吞吐优先、延迟不敏感，或者请求有明显冷热分层的场景。对低延迟在线生成来说，如果每步都要跨设备搬运 K/V，通常会很痛。

---

## 容易混淆的点

**KV Cache 不是训练时的标准优化。**  
训练阶段通常一次性处理完整序列，并且需要保留激活用于反向传播；它的并行方式和自回归 decode 不一样。KV Cache 主要服务于推理，尤其是逐 token 生成。

**KV Cache 不是只缓存最后一层。**  
每一层 Attention 都有自己的 K/V，因此每层都要缓存。层数越多，缓存越大。

**KV Cache 不缓存 Query。**  
Query 只用于当前 step，之后不会再被访问；真正会被未来 token 反复读取的是历史 Key 和 Value。

**FlashAttention 和 KV Cache 不是同一类优化。**  
FlashAttention 优化的是 Attention 计算时的 IO 和 kernel 实现；KV Cache 优化的是自回归生成中历史 K/V 的复用。两者可以同时存在。

**长上下文的瓶颈不只在容量，也在带宽。**  
即使显存装得下，每个 decode step 都要读取大量历史 K/V。上下文越长、batch 越大，显存带宽压力越明显。

---

## 实践判断

做 KV Cache 优化时，可以按这个顺序排查：

1. **先算容量**：用公式估算目标 batch、上下文和 dtype 下需要多少显存。
2. **看模型结构**：优先选择已经使用 GQA/MQA/MLA 的模型，别只看参数量。
3. **看服务框架**：长上下文和高并发场景优先考虑支持 PagedAttention、continuous batching、prefix caching 的框架。
4. **再考虑量化**：KV Cache 量化要结合 kernel 和任务质量评估，不能只看 bit 数。
5. **最后考虑淘汰/压缩/offload**：这些方法更容易影响质量或延迟，应放到具体约束下评估。

一个很实际的经验是：如果单请求短、并发低，KV Cache 优化通常不是第一瓶颈；如果上下文长、batch 大、请求长度差异明显，KV Cache 管理往往决定系统能不能跑得稳。

---

## 参考资料

- Noam Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), 2019.
- Joshua Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), 2023.
- Woosuk Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023.
- vLLM Documentation, [Paged Attention](https://docs.vllm.ai/en/latest/design/paged_attention/).
- Hugging Face Blog, [KV Caching Explained: Optimizing Transformer Inference Efficiency](https://huggingface.co/blog/not-lain/kv-caching).

