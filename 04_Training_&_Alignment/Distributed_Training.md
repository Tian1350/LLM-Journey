# 分布式训练

## 为什么需要分布式训练

一个 70B 参数的模型，仅参数本身（fp16）就需要约 140GB 显存，加上优化器状态、梯度和激活值，单卡根本放不下。即使放得下，单卡训练速度也不现实——训练 LLaMA-65B 用了 2048 张 A100，如果单卡跑需要几十年。

分布式训练的两个核心目标：
- **放得下**：把模型/数据/状态拆分到多张卡上，解决显存瓶颈
- **跑得快**：多卡并行计算，缩短训练时间

---

## 三种核心并行策略

### 1. 数据并行（Data Parallelism）

**思路**：每张卡持有完整的模型副本，但处理不同的数据子集。前向/反向各自独立计算，梯度通过 AllReduce 聚合后统一更新参数。

```
GPU 0: 完整模型 + 数据批次 0 → 计算梯度 ─┐
GPU 1: 完整模型 + 数据批次 1 → 计算梯度 ─┼─ AllReduce 聚合梯度 → 各自更新参数
GPU 2: 完整模型 + 数据批次 2 → 计算梯度 ─┘
```

优点：实现简单，扩展性好（加卡就加速）。
瓶颈：每张卡都要存完整模型 + 优化器状态，显存冗余严重。

**ZeRO（Zero Redundancy Optimizer）**

DeepSpeed 提出的优化，核心思想是消除数据并行中的冗余存储：

| 级别           | 分片内容   | 显存节省    |
|--------------|--------|---------|
| ZeRO Stage 1 | 优化器状态  | 约4倍     |
| ZeRO Stage 2 | + 梯度   | 约8倍     |
| ZeRO Stage 3 | + 模型参数 | 约N 张卡线性 |

Stage 3 下每张卡只存 1/N 的参数/梯度/优化器状态，需要时通过通信获取。本质上是用通信开销换显存——通信量增加，但允许在有限显存下训练更大的模型。

### 2. 张量并行（Tensor Parallelism）

**思路**：把单层内部的矩阵运算切分到多张卡上，每张卡计算矩阵的一部分。

**Attention 层的切分**：

多头注意力天然适合切分——N 个头分配到 N 张卡上，各自独立计算，最后拼接。

$$\text{head}_i = \text{Attention}(XW^Q_i, XW^K_i, XW^V_i)$$

每张卡负责一部分头，无需通信直到最终拼接后过输出投影。

**FFN 层的切分**：

第一个线性层按列切（Column Parallel），第二个线性层按行切（Row Parallel）：

```
GPU 0: X × W1[:, :d/2] → 激活 → × W2[:d/2, :] ─┐
                                                    ├─ AllReduce → 输出
GPU 1: X × W1[:, d/2:] → 激活 → × W2[d/2:, :] ─┘
```

这样中间的激活函数可以各自独立计算（不需要通信），只在第二个线性层之后做一次 AllReduce。

**特点**：通信频繁（每层都要同步），适合同一节点内高带宽互联（NVLink）的多卡。

### 3. 流水线并行（Pipeline Parallelism）

**思路**：把模型按层切成多段，每段放在一张卡上，数据像流水线一样依次流过。

```
GPU 0: Layer 0-7    →  GPU 1: Layer 8-15   →  GPU 2: Layer 16-23  →  GPU 3: Layer 24-31
```

**问题**：朴素实现会导致严重的"气泡"（bubble）——当 GPU 0 在做前向时，GPU 1/2/3 都在等待。

**解决方案——微批次（Micro-batch）**：

把一个大 batch 拆成多个微批次，让流水线的各段交错执行：

```
时间 →
GPU 0: [F1][F2][F3][F4]          [B4][B3][B2][B1]
GPU 1:     [F1][F2][F3][F4]      [B4][B3][B2][B1]
GPU 2:         [F1][F2][F3][F4]  [B4][B3][B2][B1]
```

微批次越多，气泡比例越小。典型方案：GPipe、PipeDream、Interleaved Pipeline（Megatron-LM）。

**特点**：层间通信量小（只传激活值），适合跨节点连接。但需要仔细调度以减少气泡。

---

## 三维并行（3D Parallelism）

实际训练大模型时，三种并行策略通常组合使用：

```
节点间：流水线并行（通信量小，适合低带宽）
节点内 GPU 间：张量并行（通信频繁，需要 NVLink 高带宽）
跨组：数据并行 + ZeRO（扩展批量大小）
```

例如 Megatron-LM 训练 GPT-3 175B：8 路张量并行 × 64 路流水线并行 × 8 路数据并行 = 4096 张 GPU。

---

## 混合精度训练

与分布式并行正交但几乎总是配合使用：

- 前向和反向用 FP16/BF16 计算（速度快、省显存）
- 主参数保留 FP32 副本用于参数更新（避免精度损失）
- 损失缩放（Loss Scaling）防止 FP16 下梯度下溢

BF16 相比 FP16 的优势：指数位更多（与 FP32 一致），不容易溢出，大模型训练中更常用。

---

## 通信原语

分布式训练的性能瓶颈往往在通信。通信原语是指在计算机网络和分布式系统中用于实现进程间通信的基本操作。分布式训练的几个核心操作：

| 原语 | 作用 | 场景 |
|------|------|------|
| AllReduce | 所有卡的张量归约后广播回所有卡 | 数据并行的梯度聚合 |
| AllGather | 每张卡的分片拼成完整张量广播给所有卡 | ZeRO Stage 3 前向时收集参数 |
| ReduceScatter | 归约后将结果分片分发到各卡 | ZeRO Stage 2 梯度分片 |
| All-to-All | 每张卡向其他所有卡发送不同数据 | MoE 的 token 路由 |

---

## 参考资料

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)](https://arxiv.org/abs/1910.02054)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (Narayanan et al., 2021)](https://arxiv.org/abs/2104.04473)
- [GPipe: Efficient Training of Giant Neural Networks (Huang et al., 2019)](https://arxiv.org/abs/1811.06965)
