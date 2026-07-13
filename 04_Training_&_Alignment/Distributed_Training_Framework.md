# 大模型训练框架

分布式训练的**策略**（数据/张量/流水线并行、ZeRO 等，详见 [Distributed_Training](./Distributed_Training.md)）需要具体的**框架**来落地。这些框架封装了通信、显存管理、并行切分等复杂细节，让开发者不必从零手写。下面梳理主流的几个。

---

## FSDP（Fully Sharded Data Parallel）

PyTorch 官方内置的分布式方案，思想上等价于 DeepSpeed 的 ZeRO Stage 3——把模型参数、梯度、优化器状态全部分片到各张卡上，需要时通过通信临时聚合。

- **定位**：PyTorch 原生，无需额外依赖，和 PyTorch 生态无缝集成
- **原理**：前向/反向计算某层时，通过 AllGather 临时收集该层的完整参数，用完即释放；梯度用 ReduceScatter 分片
- **适用**：以数据并行为主、希望留在纯 PyTorch 技术栈内的场景
- **特点**：API 相对底层，新版本（FSDP2）在易用性和性能上有明显改进

---

## DeepSpeed

微软开源的深度学习优化库，最著名的贡献是 **ZeRO 系列优化**。

- **核心能力**：ZeRO Stage 1/2/3 分级显存优化、ZeRO-Offload（把优化器状态/参数卸载到 CPU 内存甚至 NVMe）、ZeRO-Infinity（突破单机显存限制训练超大模型）
- **附加功能**：混合精度、梯度累积、流水线并行、稀疏注意力、推理优化等
- **适用**：显存是主要瓶颈、想用有限硬件训练更大模型的场景
- **特点**：功能全面，配置通过 JSON 文件驱动，与 HuggingFace 生态集成好

---

## Megatron-LM

NVIDIA 开源的大模型训练框架，主打**张量并行**和极致的训练效率。

- **核心能力**：高效的张量并行（Tensor Parallelism）实现，配合流水线并行、数据并行组成三维并行
- **优势**：针对 NVIDIA GPU 深度优化（Tensor Core、通信 kernel），在超大规模（千卡以上）训练中吞吐领先
- **适用**：从头预训练百亿/千亿级大模型的工业级场景
- **特点**：偏底层、配置复杂，更适合有专门基础设施团队的机构。GPT-3、许多国产大模型的训练都基于它或其变体

---

## Accelerate

HuggingFace 出品的**轻量级封装库**，本身不发明新的并行策略，而是提供统一接口来调度底层的 FSDP、DeepSpeed 等。

- **定位**：降低分布式训练的使用门槛，同一份训练代码可以在单卡、多卡、多机、TPU 上无缝切换
- **原理**：在 PyTorch 训练循环外做一层薄封装，通过配置决定底层用 FSDP 还是 DeepSpeed
- **适用**：中小规模微调、快速实验、不想深入分布式细节的场景
- **特点**：上手快、代码改动小，但对超大规模训练的精细控制不如直接用 Megatron-LM

---

## 如何选择

| 框架 | 出品方 | 核心定位 | 典型场景 |
|------|--------|---------|---------|
| FSDP | PyTorch | 原生参数分片（≈ZeRO-3） | 纯 PyTorch 栈、数据并行为主 |
| DeepSpeed | 微软 | ZeRO 显存优化 + Offload | 显存受限、训练更大模型 |
| Megatron-LM | NVIDIA | 张量并行 + 极致吞吐 | 千卡级从头预训练 |
| Accelerate | HuggingFace | 统一封装、易用 | 微调、快速实验 |

实践中常见组合：
- **微调/中小规模**：Accelerate + DeepSpeed（或 FSDP），省心
- **超大规模预训练**：Megatron-LM，或 Megatron-DeepSpeed（两者结合，兼顾张量并行和 ZeRO）

---

## 参考资料

- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel (Zhao et al., 2023)](https://arxiv.org/abs/2304.11277)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)](https://arxiv.org/abs/1910.02054)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053)
- [HuggingFace Accelerate Documentation](https://huggingface.co/docs/accelerate)
