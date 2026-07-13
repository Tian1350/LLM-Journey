# LLM-Journey

记录我从理论到工程，系统理解大语言模型的学习路径与实践笔记。

## 知识地图

| 章节 | 主题 | 内容 |
|---|---|---|
| [01_Foundation_&_Mechanisms](./01_Foundation_&_Mechanisms) | 基础与机制 | Transformer、注意力、位置编码、归一化、激活函数等核心机制 |
| [02_Architecture_Evolution](./02_Architecture_Evolution) | 架构演进 | Decoder-Only、BERT/LLaMA/Mamba 家族、MoE、扩散模型 |
| [03_Multimodal_&_Cross_Domain](./03_Multimodal_&_Cross_Domain) | 多模态与跨领域 | 多模态模型、视觉语言模型、时序预测、推荐大模型 |
| [04_Training_&_Alignment](./04_Training_&_Alignment) | 训练与对齐 | SFT、PEFT、强化学习对齐、分布式训练、混合精度 |
| [05_Inference_&_Optimization](./05_Inference_&_Optimization) | 推理与优化 | KV Cache、FlashAttention、量化、采样、Scaling Law |
| [06_Agent_&_Applications](./06_Agent_&_Applications) | Agent 与应用 | Agent、Harness、MCP、RAG、Prompt Engineering、Skills |
| [07_Frontier_&_Interpretability](./07_Frontier_&_Interpretability) | 前沿与可解释性 | 新兴发现、可解释性研究（J-space & J-lens 等） |

## 目录导航

### 01_Foundation_&_Mechanisms：基础与机制

- [Tokenizer 分词器](./01_Foundation_&_Mechanisms/Tokenizer.md)
- [Transformer 架构](./01_Foundation_&_Mechanisms/Transformer_Architecture.md)
- [位置编码](./01_Foundation_&_Mechanisms/Positional_Encoding.md)
- [归一化方法](./01_Foundation_&_Mechanisms/Normalization.md)
- [激活函数](./01_Foundation_&_Mechanisms/Activation_Function.md)
- [Dropout](./01_Foundation_&_Mechanisms/Dropout.md)
- [损失函数](./01_Foundation_&_Mechanisms/Loss_Function.md)
- [MHA、MQA、GQA、MLA 注意力机制对比](./01_Foundation_&_Mechanisms/MHA_MQA_GQA_MLA.md)

### 02_Architecture_Evolution：架构演进

- [为什么主流大模型采用 Decoder-Only 架构](./02_Architecture_Evolution/Why_Decoder_Only.md)
- [BERT 家族](./02_Architecture_Evolution/BERT_Family.md)
- [LLaMA 家族](./02_Architecture_Evolution/LLaMA_Family.md)
- [Mamba 家族](./02_Architecture_Evolution/Mamba_Family.md)
- [混合专家模型 (MoE)](./02_Architecture_Evolution/Mixture_of_Experts.md)
- [扩散模型](./02_Architecture_Evolution/Diffusion_Models.md)
- [扩散语言模型 (Diffusion LLM)](./02_Architecture_Evolution/Diffusion_LLM.md)

### 03_Multimodal_&_Cross_Domain：多模态与跨领域

- [多模态大模型](./03_Multimodal_&_Cross_Domain/Multimodal_Model.md)
- [视觉语言模型](./03_Multimodal_&_Cross_Domain/Vision_Language_Models.md)
- [时序预测模型](./03_Multimodal_&_Cross_Domain/Time_Series_Models.md)
- [推荐大模型](./03_Multimodal_&_Cross_Domain/Recommendation_Model.md)

### 04_Training_&_Alignment：训练与对齐

- [有监督微调 (SFT)](./04_Training_&_Alignment/SFT.md)
- [Adapter Tuning & PEFT](./04_Training_&_Alignment/Adapter_Tuning_&_PEFT.md)
- [强化学习与对齐](./04_Training_&_Alignment/Reinforcement_Learning_&_Alignment.md)
- [分布式训练](./04_Training_&_Alignment/Distributed_Training.md)
- [混合精度训练](./04_Training_&_Alignment/Mixed_Precision_Training.md)

### 05_Inference_&_Optimization：推理与优化

- [KV Cache 优化](./05_Inference_&_Optimization/KV_Cache_Optimization.md)
- [FlashAttention](./05_Inference_&_Optimization/FlashAttention.md)
- [量化技术](./05_Inference_&_Optimization/Quantization.md)
- [采样策略](./05_Inference_&_Optimization/Sampling.md)
- [Scaling Law](./05_Inference_&_Optimization/Scaling_Law.md)

### 06_Agent_&_Applications：Agent 与应用

- [Agent 概述与分类](./06_Agent_&_Applications/Agent.md)
- [Harness 执行框架](./06_Agent_&_Applications/Harness.md)
- [MCP (Model Context Protocol)](./06_Agent_&_Applications/MCP.md)
- [RAG 检索增强生成](./06_Agent_&_Applications/RAG.md)
- [Prompt Engineering](./06_Agent_&_Applications/Prompt_Engineering.md)
- [Skills](./06_Agent_&_Applications/Skills.md)
- [OpenClaw](./06_Agent_&_Applications/OpenClaw.md)

### 07_Frontier_&_Interpretability：前沿与可解释性

- [J-space 与 J-lens（全局工作空间）](./07_Frontier_&_Interpretability/J-space_&_J-lens.md)

## 项目结构

```text
My-LLM-Journey/
├── 01_Foundation_&_Mechanisms/
│   ├── Activation_Function.md
│   ├── Dropout.md
│   ├── Loss_Function.md
│   ├── MHA_MQA_GQA_MLA.md
│   ├── Normalization.md
│   ├── Positional_Encoding.md
│   ├── Tokenizer.md
│   └── Transformer_Architecture.md
├── 02_Architecture_Evolution/
│   ├── BERT_Family.md
│   ├── Diffusion_LLM.md
│   ├── Diffusion_Models.md
│   ├── LLaMA_Family.md
│   ├── Mamba_Family.md
│   ├── Mixture_of_Experts.md
│   └── Why_Decoder_Only.md
├── 03_Multimodal_&_Cross_Domain/
│   ├── Multimodal_Model.md
│   ├── Recommendation_Model.md
│   ├── Time_Series_Models.md
│   └── Vision_Language_Models.md
├── 04_Training_&_Alignment/
│   ├── Adapter_Tuning_&_PEFT.md
│   ├── Distributed_Training.md
│   ├── Mixed_Precision_Training.md
│   ├── Reinforcement_Learning_&_Alignment.md
│   └── SFT.md
├── 05_Inference_&_Optimization/
│   ├── FlashAttention.md
│   ├── KV_Cache_Optimization.md
│   ├── Quantization.md
│   ├── Sampling.md
│   └── Scaling_Law.md
├── 06_Agent_&_Applications/
│   ├── Agent.md
│   ├── Harness.md
│   ├── MCP.md
│   ├── OpenClaw.md
│   ├── Prompt_Engineering.md
│   ├── RAG.md
│   └── Skills.md
├── 07_Frontier_&_Interpretability/
│   └── J-space_&_J-lens.md
├── assets/
├── LICENSE
└── README.md
```
