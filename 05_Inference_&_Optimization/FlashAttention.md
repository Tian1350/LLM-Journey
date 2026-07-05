# FlashAttention

## 要解决的问题

标准注意力的计算是：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

朴素实现的问题不在计算量（FLOPs），而在**显存访问**。它需要：
1. 计算 $S = QK^T$，得到 $N \times N$ 的注意力分数矩阵，写回显存（HBM）
2. 对 $S$ 做 softmax，再读出、写回
3. 计算 $S \cdot V$，再读写

这个 $N \times N$ 矩阵（$N$ 是序列长度）要在 HBM 和片上 SRAM 之间反复搬运。当 $N$ 很大时，注意力是**内存带宽受限（memory-bound）**的——GPU 的算力没用满，时间都花在等数据搬运上。

关键背景：GPU 的 SRAM 速度极快但容量极小（几十 MB），HBM 容量大但带宽慢得多（差一个数量级）。优化的核心是**减少 HBM 访问**。

---

## 核心思想

FlashAttention 的目标：在不把完整的 $N \times N$ 矩阵写入 HBM 的前提下完成注意力计算。两个关键技术：

### 1. 分块计算（Tiling）

把 $Q, K, V$ 切成小块，每次只把能放进 SRAM 的小块加载进来，在 SRAM 内完成该块的注意力计算，只把最终结果写回 HBM。

难点：softmax 需要对整行做归一化（要知道整行的最大值和指数和），但分块后每次只看到一部分。

解决方案——**Online Softmax（在线 softmax）**：
- 逐块处理时，维护当前的运行最大值 $m$ 和运行指数和 $\ell$
- 每来一个新块，更新 $m$ 和 $\ell$，并对已累积的输出做相应的缩放修正
- 处理完所有块后，结果和一次性计算完整 softmax 完全等价（数值精确，不是近似）

这样就避免了在 HBM 中物化（materialize）整个 $N \times N$ 矩阵。

### 2. 重计算（Recomputation）

反向传播需要用到前向的注意力矩阵 $S$。标准做法是前向时把 $S$ 存下来，但那又要占 $O(N^2)$ 显存。

FlashAttention 的做法：**前向不保存 $S$，反向时用已保存的 $Q, K, V$ 和统计量（$m, \ell$）重新算一遍**。

这是经典的"用计算换显存"——重算增加了一些 FLOPs，但省下了巨大的显存读写，整体反而更快（因为本来就是 memory-bound）。

---

## 效果

- 显存占用从 $O(N^2)$ 降到 $O(N)$，使长上下文训练成为可能
- 计算结果与标准注意力**精确相等**（不是近似，不损失精度）
- 实际训练/推理速度提升 2-4 倍

---

## FlashAttention-2

v2 在 v1 基础上的工程优化，进一步提升 GPU 利用率：

| 改进点 | 说明 |
|--------|------|
| 减少非矩阵乘运算 | v1 中有较多的 rescaling 等非 matmul 操作，而 GPU 的 Tensor Core 专为 matmul 优化。v2 重排计算顺序，把这些操作降到最少 |
| 更好的并行划分 | v1 主要沿 batch 和 head 维度并行，序列很长但 batch 小时并行度不足。v2 增加了沿**序列长度维度**的并行，提升 SM 占用率 |
| 优化 warp 间的工作分配 | 调整线程块内 warp 的任务划分，减少共享内存的读写和同步开销 |

结果：v2 比 v1 再快约 2 倍，接近 GPU 理论峰值算力的利用率。

---

## FlashAttention-3

针对 Hopper 架构（H100）的进一步优化：利用 H100 的异步特性（TMA、warp-group 级矩阵乘）和 FP8 低精度计算，进一步提升吞吐。

---

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (Shah et al., 2024)](https://arxiv.org/abs/2407.08608)
- [Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)


