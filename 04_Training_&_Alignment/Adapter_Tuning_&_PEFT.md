# Adapter Tuning 与 PEFT

## 先把名字说清楚

**Adapter Tuning** 属于 PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）的一类思路：冻结大部分预训练模型参数，只训练少量新增参数或少量可训练模块。这样做的目的不是让模型“学得更多”，而是让微调更便宜、更容易存储和部署。

一句话概括：

```text
全量微调：改整个模型。
Adapter/PEFT：尽量不动原模型，只训练一小块可插拔参数。
```

---

## Adapter Tuning 是什么

最早一类 Adapter 方法会在 Transformer 层里插入一个很小的瓶颈模块。常见结构大致是：

```text
hidden
  ↓
down projection：d_model -> r
  ↓
non-linearity
  ↓
up projection：r -> d_model
  ↓
residual add
```

其中 `r` 是瓶颈维度，通常远小于 `d_model`。

微调时：

- 预训练模型主体参数冻结
- 只训练 Adapter 参数
- 每个任务可以保存一套自己的 Adapter
- 推理时加载对应任务的 Adapter

这类方法的好处很直接：如果底座模型很大，不同任务不需要各自保存一整份模型，只需要保存小的 Adapter 权重。

---

## 为什么要做 PEFT

全量微调当然最直接，但问题也明显：

- 显存占用高
- optimizer state 很大
- 每个任务都要保存一份完整模型
- 多任务部署不方便
- 大模型上训练成本很高

PEFT 的思路是承认一个事实：很多下游任务不需要重写整个模型，只需要在原模型能力上做一个小幅偏移。

这种“小幅偏移”可以有不同实现：

- 插入小模块：Adapter
- 学习连续前缀：Prefix Tuning
- 学习低秩增量：LoRA
- 量化底座再训练 LoRA：QLoRA
- 动态分配低秩预算：AdaLoRA

---

## Adapter Tuning 与 Prefix Tuning

这两个方法都冻结原模型，但改的位置不一样。

| 方法 | 训练什么 | 放在哪里 | 主要代价 |
|---|---|---|---|
| Adapter Tuning | 小型瓶颈模块 | Transformer 层内部 | 增加额外计算，可能带来推理延迟 |
| Prefix Tuning | 连续可训练 prefix 向量 | 注意力上下文中，常表现为每层可关注的虚拟前缀 | 占用有效上下文或 KV Cache 空间 |

Prefix Tuning 不是在输入文本前面简单拼几个离散 token。它学习的是连续向量，模型后续 token 可以像关注普通上下文一样关注这些 prefix。

Adapter Tuning 的问题也不是“训练慢”这么简单，而是推理路径里多了一段小网络。对长序列或大 batch 来说影响未必总是致命，但在线短文本场景下，额外层可能比较敏感。

---

## LoRA：低秩增量

LoRA 是现在最常用的 PEFT 方法之一，但它和传统 Adapter 层不是一回事。

传统 Adapter 往模型里插入新层；LoRA 不插入完整的新网络，而是在某些线性层上学习一个低秩更新：

```math
W' = W + \Delta W
```

其中原始权重 `W` 冻结，只训练：

```math
\Delta W = BA
```

如果原线性层是：

```text
W: d_out × d_in
```

则：

```text
A: r × d_in
B: d_out × r
```

LoRA 参数量为：

```text
r * (d_in + d_out)
```

如果是方阵 `d_model × d_model`，才可以近似写成：

```text
2 * r * d_model
```

其中， `L_LoRA` 表示目标线性层数量。

### 为什么常见初始化是 A 随机、B 为 0

常见实现里，`A` 随机初始化，`B` 初始化为 0。这样一开始：

```text
BA = 0
W' = W
```

模型初始行为和原底座模型一致，不会一上来被随机 LoRA 分支扰乱。

同时，`A` 不是 0，`B` 的梯度可以先动起来。若 `A` 和 `B` 都初始化为 0，梯度容易被卡住，两个矩阵都学不起来。

### LoRA 的推理优势

训练后，LoRA 的低秩增量可以合并回原权重：

```math
W' = W + BA
```

合并后推理时仍然只做一次普通线性层计算。因此 LoRA 通常不会像传统 Adapter 层那样引入额外网络路径。

---

## QLoRA：量化底座 + LoRA

QLoRA 可以理解为：

```text
冻结的 4-bit 量化底座模型 + 可训练的 LoRA 参数
```

需要注意：QLoRA 不是“先量化模型，然后正常全量微调”。它的关键是：

- 底座模型权重量化到 4-bit，并保持冻结
- 前向和反向时会经过反量化计算
- 梯度更新的是 LoRA 参数，不是底座权重
- 通过 NF4、double quantization、paged optimizer 等技术进一步省显存

QLoRA 的意义是让很大的模型也能在较小显存上做高质量微调。它解决的重点是训练显存，而不是单纯的推理压缩。

---

## AdaLoRA：动态分配 rank

普通 LoRA 通常给每个目标矩阵相同的 rank，比如全都设成 `r = 8`。

这个做法简单，但不一定合理。不同层、不同矩阵的重要性不同，统一分配 rank 可能会浪费预算。

AdaLoRA 的思路是：

```text
重要的矩阵多分一点 rank。
不重要的矩阵少分一点 rank。
```

它把增量更新写成类似 SVD 的形式，并根据重要性分数逐步裁剪不重要的奇异值，从而在固定参数预算下分配得更细。

可以把它理解成“LoRA 的 rank 不再一刀切”。

---

## 几种方法怎么选

| 方法 | 适合场景 | 优点 | 主要代价 |
|---|---|---|---|
| Full Fine-tuning | 数据充足、算力充足、希望最大化任务性能 | 自由度最高 | 显存、存储、训练成本最高 |
| Adapter Tuning | 多任务部署，每个任务保存小模块 | 模块化强，任务切换方便 | 插入额外层，可能增加推理延迟 |
| Prefix Tuning | 生成任务、小参数预算 | 参数量很小 | 占用上下文/KV Cache，调参敏感 |
| LoRA | 大模型微调的默认选择之一 | 训练便宜，可合并权重，推理友好 | rank 和注入位置需要调 |
| QLoRA | 单卡或小显存微调大模型 | 显存省，实践成熟 | 训练速度和量化细节更复杂 |
| AdaLoRA | 参数预算很紧，希望自动分配 rank | rank 分配更细 | 实现和训练流程更复杂 |

---

## 容易混淆的点

### 1. Adapter 和 LoRA 都叫 PEFT，但不是同一种结构

Adapter 通常是在网络中插入小模块；LoRA 是给已有线性层加低秩增量。它们都属于 PEFT，但实现位置不同。

### 2. Prefix Tuning 不是 Prompt Engineering

Prompt Engineering 调的是自然语言文本；Prefix Tuning 学的是连续向量。它更接近一种可训练参数，而不是手写提示词。

### 3. QLoRA 不是只为推理服务

量化常用于推理，但 QLoRA 的重点是降低微调时的显存，让 LoRA 可以在量化底座上训练。

### 4. PEFT 不等于效果一定差

PEFT 参数少，但不代表一定比全量微调差很多。很多任务上，LoRA、QLoRA 这类方法已经足够接近全量微调。真正差异取决于数据规模、任务难度、底座模型、rank、注入位置和训练配置。

### 5. PEFT 也不是免费午餐

它省参数、省显存，但仍然需要认真处理：

- 数据质量
- 学习率
- rank
- target modules
- 训练轮数
- 是否合并权重
- 推理时如何加载多个 adapter

参数少只是降低门槛，不是自动得到好结果。

---

## 参考资料

- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)
