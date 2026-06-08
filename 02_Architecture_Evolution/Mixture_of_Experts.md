# Mixture of Experts (MoE)

## 基本概念

MoE 的核心思想：用多个"专家"子网络替代 Transformer 中的单一 FFN 层，通过一个路由（Router/Gate）决定每个 token 送往哪几个专家处理。

```
输入 token → Router → 选中 Top-K 个专家 → 各专家输出按路由权重加权求和 → 输出
```

关键区分：**总参数量**（所有专家加起来）远大于**激活参数量**（每个 token 实际用到的专家参数）。比如 Mixtral 8x7B 总参数约 47B，但每个 token 只激活 2 个专家，激活参数约 13B。

---

## 为什么用 MoE

1. **以更低的训练成本获得更强的模型容量**
   - 总参数大 → 模型容量大，能存储更多知识
   - 激活参数小 → 每个 token 的计算量（FLOPs）与小模型相当，训练速度快
   - 实际效果：MoE 模型用同样的训练预算，往往能达到比同 FLOPs 稠密模型更好的性能

2. **推理时计算效率高**
   - 每个 token 只经过 K 个专家（通常 K=1 或 2），FLOPs 远小于把所有专家都跑一遍

3. **模型容量可灵活扩展**
   - 加专家不增加单 token 的计算量，只增加总参数（和显存）

---

## MoE 的代价

1. **显存需求大**：所有专家的参数都需要加载到显存（即使大部分不被当前 token 使用），模型部署门槛高
2. **训练不稳定**：路由决策是离散的，容易出现专家坍缩（部分专家从不被选中）和负载不均
3. **微调困难**：专家之间的稀疏激活使得微调时容易过拟合到少数专家，泛化性下降
4. **通信开销**：分布式训练中，不同专家可能分布在不同设备上，token 路由导致大量跨设备通信（All-to-All）

---

## 路由机制

### Token-Choice 路由（主流）

每个 token 经过一个可学习的线性层（门控网络），产生对所有专家的评分，取 Top-K：

$$g(x) = \text{TopK}(\text{softmax}(W_g \cdot x))$$

选中的 K 个专家的输出按归一化后的路由权重加权求和。

### Expert-Choice 路由

反过来，每个专家选择自己要处理哪些 token（固定每个专家的容量）。好处是天然负载均衡，缺点是某些 token 可能被零个专家选中或被所有专家抢。

### 负载均衡问题

硬路由（Top-K）训练时容易出现马太效应：初始阶段被选中较多的专家获得更多梯度更新 → 变得更强 → 更容易被选中。最终可能只有少数专家在工作，其余"饿死"。

常见解决方案：

- **辅助负载均衡损失（Auxiliary Load Balancing Loss）**：惩罚路由分布的不均匀，鼓励 token 均匀分配到各专家。这是目前最通用的做法（Switch Transformer、Mixtral 都在用）
- **专家容量限制（Expert Capacity）**：每个专家设置最大处理 token 数，溢出的 token 走 residual 或随机分配
- **噪声注入**：在路由评分上加噪声（如 Gaussian noise），增加探索性

---

## 专家结构是否必须相同

实践中几乎所有主流模型都用同构专家（结构相同，参数不同）：

- 同构专家可以批量并行计算，硬件利用率高
- 异构结构会加剧负载不均——处理速度不同的专家很难做 batching

不过在多模态领域，异构专家是一个研究方向：不同结构的专家分别处理文本、图像、音频等不同模态的输入。

---

## 同构专家的并行计算

在 MoE 中，同构意味着所有专家网络结构相同、仅参数不同。如果用 for 循环逐个跑专家，会产生大量算子下发（Kernel Launch）开销，GPU 利用率极低。

核心优化思路：**把多个专家的独立计算合并为一次批量矩阵运算**。具体实现方式因框架和硬件生态不同：

| 实现方式 | 生态 | 原理 |
|---------|------|------|
| Grouped/Batched GEMM | CUDA/GPU（Megatron-LM、DeepSpeed） | 将各专家的矩阵乘法打包为一次 cuBLAS batched matmul 调用 |
| Block-sparse MatMul | GPU（Megablocks） | 把 token-expert 分配关系编码为 block-sparse 矩阵，一次稀疏乘法完成所有专家计算 |
| vmap | JAX/TPU（Google 生态） | 将专家维度映射为 batch 维度，编译器自动融合为批量运算 |
| Triton 自定义 Kernel | GPU | 手写 fused kernel，把路由+计算+反路由融合在一起 |

**实际计算流程：**

1. **路由与分组**：根据 Router 输出，将 batch 内所有 token 按目标专家分组排序
2. **批量前向**：所有专家的权重在初始化时就存储为堆叠张量 `[E, d_model, d_ff]`，对分组后的 token 执行一次批量矩阵乘法，同时完成所有专家的计算
3. **反路由与聚合**：将各专家的输出按原始 token 顺序重排，再按路由权重加权求和，得到最终输出

关键细节：权重堆叠是模型初始化时就完成的（参数本身就以 `[E, ...]` 形状存储），不是每次前向传播的动态操作。

---

## MoE 蒸馏

将大的 MoE 模型蒸馏成小的稠密模型，常见做法：

- 用 MoE 模型的输出 logits 作为 soft label，训练一个参数量更小的稠密学生模型
- 蒸馏路由决策，让学生模型学会模仿不同专家的行为，训练一个更小的稀疏MoE

动机：部署时 MoE 的显存需求过高，蒸馏后的稠密模型更容易 serve。DeepSeek 系列就提供了 MoE 和对应蒸馏稠密版本。

---

## 代表模型

| 模型 | 专家配置 | 关键特点 |
|------|---------|---------|
| Switch Transformer (2022) | Top-1 路由 | 首次证明 Top-1 也能训练稳定 |
| Mixtral 8x7B (2024) | 8 专家 Top-2 | 开源 MoE 标杆，性能对标 LLaMA 2 70B |
| DeepSeek-V2/V3 | 细粒度专家（更多更小的专家） | 结合 MLA 注意力，推理效率极高 |
| Qwen-MoE | 多种配置 | 阿里开源 MoE 系列 |

---

## 参考资料

- [Switch Transformers: Scaling to Trillion Parameter Models (Fedus et al., 2022)](https://arxiv.org/abs/2101.03961)
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088)
- [DeepSeekMoE: Towards Ultimate Expert Specialization (Dai et al., 2024)](https://arxiv.org/abs/2401.06066)
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538)
