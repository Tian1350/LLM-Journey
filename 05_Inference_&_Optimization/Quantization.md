# 量化（Quantization）

## 核心思想

量化是把模型的浮点数表示（FP32/FP16/BF16）转换为更低位宽的整数或浮点格式（INT8/INT4/FP8），以减少显存占用和加速推理。

量化对象主要有三类：
- **权重（Weights）**：模型参数，量化后固定不变，收益最直接
- **激活值（Activations）**：推理时每层的中间输出，量化难度更高（值域动态变化）
- **KV Cache**：存储注意力的 Key/Value，长上下文场景下占用显存大，量化可显著降低显存压力

量化不是免费的午餐——位宽越低，精度损失越大，需要算法补偿。

---

## 量化的基本原理

以最常见的均匀线性量化为例，把一个浮点数 $x$ 映射到整数：

$$x_q = \text{round}\left(\frac{x}{s}\right) + z, \quad x \approx s \cdot (x_q - z)$$

其中 $s$（scale）是缩放因子，$z$（zero point）是零点偏移（对称量化下 $z=0$）。

**粒度**决定了每套 $s, z$ 覆盖多少参数：
- **Per-tensor**：整个张量共用一组 $s, z$，最粗，精度最低
- **Per-channel**：每个输出通道一组 $s, z$，精度更好，权重量化常用
- **Per-group / Per-token**：更细粒度，精度最好，计算开销也更大

---

## 量化策略

### 后训练量化（PTQ，Post-Training Quantization）

训练完成后直接对权重/激活量化，不需要重新训练，只需少量校准数据（或完全无数据）。部署成本低，是目前大模型量化的主流方式。

**代表算法：**

**GPTQ（2022）**

基于二阶信息（Hessian 矩阵）逐层量化权重。核心思路：把量化权重的误差视为优化问题，量化一个权重后，用该权重的 Hessian 信息调整同层其他未量化权重来补偿误差。

- 支持 INT4/INT3，精度损失小
- 只量化权重，激活值保持 FP16（W4A16 模式）
- 需要少量校准数据，量化速度较快

**AWQ（Activation-aware Weight Quantization，2023）**

观察到权重中有少量"重要"通道（对应激活值较大的通道），直接量化这些通道会引入大误差。AWQ 的做法：在量化前对重要通道的权重做逐通道缩放（放大），等效地平滑激活，使所有通道的量化误差更均匀。

- 不需要反向传播，速度快
- W4A16，和 GPTQ 类似精度，但无需 Hessian 计算，更轻量

**SmoothQuant（2022）**

专门针对 W8A8（权重和激活都量化为 INT8）的场景。问题在于激活值不同通道的幅度差异极大（某些通道值很大，其他很小），直接量化误差大。

SmoothQuant 的做法：把激活值的"难以量化性"（逐通道的幅度差异）转移到权重上：

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$$

$X$ 除以 $s$ 后幅度更均匀（容易量化），$W$ 乘以 $s$ 后同样可以量化（权重比激活更好量化）。

W8A8 可以利用 INT8 矩阵乘法硬件（如 NVIDIA Tensor Core 的 int8 指令），推理速度更快。

---

### 量化感知训练（QAT，Quantization-Aware Training）

在训练过程中插入**伪量化节点**（Fake Quantization）：前向传播时模拟量化误差，反向传播时用 STE（Straight-Through Estimator）近似量化算子的梯度，让模型在训练阶段就适应量化带来的精度损失。

```
前向：x → 量化 → 反量化 → 继续计算（含量化误差）
反向：梯度直通（STE），当作恒等函数处理
```

优点：精度最好，尤其在低位宽（INT4 以下）时远优于 PTQ。
代价：需要一定规模的训练数据和完整的训练流程，计算成本比 PTQ 高一到两个数量级。

实践中大模型很少用完整 QAT，更多采用 PTQ + 算法补偿的组合。

---

## 为什么 PTQ 比 QAT 更流行

核心原因不只是"成本低"，而是在大模型时代 PTQ 的性价比发生了根本性转变：

**1. 大模型对量化天然更鲁棒**

模型越大，量化精度损失越小。这是经验上反复验证的规律：7B 模型 INT4 量化后性能下降明显，而 70B 模型同样量化到 INT4，性能损失往往可以忽略不计。原因是大模型参数冗余更高，单个参数的精度损失对整体表征影响更小。

这意味着在 Scaling Law 的趋势下，PTQ 的可用性会越来越高——换句话说，不是靠"增大参数弥补精度损失"，而是"更大的模型本身就更能承受精度损失"。

**2. PTQ 的算法进步已经相当成熟**

GPTQ、AWQ、SmoothQuant 等算法通过二阶信息补偿、激活感知缩放等手段，把 PTQ 的精度损失压到了很低。在 INT8 甚至 INT4 上，现代 PTQ 算法的精度和完整 QAT 已经非常接近，继续承担 QAT 的成本收益越来越小。

**3. QAT 的工程代价在大模型时代不现实**

QAT 需要完整的训练流程：准备数据集、配置分布式训练、跑完整个训练 pipeline……对于一个已有的 70B 开源模型，仅训练成本就可能达到数十万美元。而 PTQ 往往只需要一台机器、几百条校准数据、几小时就能完成。

**4. 应用场景决定了 PTQ 足够用**

量化的主要需求是在显存受限的硬件上部署，对绝对精度要求不高——用户更在乎"能不能在 24GB 显存的消费级 GPU 上跑起来"，而不是"和 FP16 精度差了 0.3 个点"。PTQ 在这个需求下完全够用。

QAT 仍然有价值的场景：极端低位宽（INT2/INT3）、边缘设备部署（对精度要求极高但硬件极受限）、或者模型本身较小（QAT 成本可接受）。

---

## 常见量化格式与场景

| 格式 | 显存 vs FP16 | 典型场景 | 精度损失 |
|------|------------|---------|---------|
| INT8 (W8A8) | ~0.5x | 服务端推理，吞吐优先 | 小 |
| INT8 (W8A16) | ~0.5x 权重 | 中等精度要求 | 极小 |
| INT4 (W4A16) | ~0.25x 权重 | 消费级 GPU，内存受限 | 中等 |
| INT4 (W4A8) | ~0.25x 权重 | 极致压缩 | 较大 |
| FP8 | ~0.5x | H100 训练/推理 | 小（配合 scaling） |
| GPTQ/AWQ INT4 | ~0.25x | 开源模型本地部署 | 中等，算法补偿后可接受 |

---

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration (Lin et al., 2023)](https://arxiv.org/abs/2306.00978)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs (Xiao et al., 2022)](https://arxiv.org/abs/2211.10438)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (Dettmers et al., 2022)](https://arxiv.org/abs/2208.07339)
