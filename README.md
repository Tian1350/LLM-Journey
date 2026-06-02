# LLM-Journey

记录我从理论到工程，系统理解大语言模型的学习路径与实践笔记。

## 知识地图

| 章节 | 主题 | 内容 |
|---|---|---|
| [01_Foundation](./01_Foundation) | 基础理论 | Transformer、位置编码、归一化等核心模型基础 |
| [02_Architecture_Evolution](./02_Architecture_Evolution) | 架构演进 | MHA/MQA/GQA/MLA、LLaMA 系列、视觉语言模型等架构发展 |
| [03_Training_&_Alignment](./03_Training_&_Alignment) | 训练与对齐 | SFT、PEFT、RLHF、DPO、分布式训练 |
| [04_Inference_&_Engineering](./04_Inference_&_Engineering) | 推理与工程 | KV Cache、vLLM、PagedAttention、量化技术 |
| [05_Interview_&_Projects](./05_Interview_&_Projects) | 面试与项目 | 项目复盘、高频问题与工程实践 |
| [06_Agent_&_Applications](./06_Agent_&_Applications) | Agent 与应用 | Agent、MCP、Skills 等应用层能力 |

## 目录导航

### 01_Foundation：基础理论

- [Transformer 架构深度剖析](./01_Foundation/Transformer_Architecture.md)
- [位置编码数学推导](./01_Foundation/Positional_Encoding.md)
- [归一化方法](./01_Foundation/Normalization.md)

### 02_Architecture_Evolution：架构演进

- [MHA、MQA、GQA、MLA 注意力机制对比](./02_Architecture_Evolution/MHA_MQA_GQA_MLA.md)
- [LLaMA 系列](./02_Architecture_Evolution/LLaMA_Family.md)
- [视觉语言模型](./02_Architecture_Evolution/Vision_Language_Models.md)

### 03_Training_&_Alignment：训练与对齐

- [SFT & PEFT](./03_Training_&_Alignment/SFT_&_PEFT.md)
- [RLHF & DPO](./03_Training_&_Alignment/RLHF_&_DPO.md)
- [分布式训练](./03_Training_&_Alignment/Distributed_Training.md)

### 04_Inference_&_Engineering：推理与工程

- [KV Cache 优化](./04_Inference_&_Engineering/KV_Cache_Optimization.md)
- [vLLM & PagedAttention](./04_Inference_&_Engineering/vLLM_&_PagedAttention.md)
- [量化技术](./04_Inference_&_Engineering/Quantization.md)

### 05_Interview_&_Projects：面试与项目

- [AI4Science 项目](./05_Interview_&_Projects/My_AI4Science_Project.md)
- [VRP 项目](./05_Interview_&_Projects/My_VRP_Project.md)
- [高频面试题](./05_Interview_&_Projects/High_Freq_Questions.md)

### 06_Agent_&_Applications：Agent 与应用

- [Agent 概述与分类](./06_Agent_&_Applications/Agent.md)
- [MCP 概述](./06_Agent_&_Applications/MCP.md)
- [Skills 概述](./06_Agent_&_Applications/Skills.md)

## 项目结构

```text
My-LLM-Journey/
├── 01_Foundation/
│   ├── Normalization.md
│   ├── Positional_Encoding.md
│   └── Transformer_Architecture.md
├── 02_Architecture_Evolution/
│   ├── LLaMA_Family.md
│   ├── MHA_MQA_GQA_MLA.md
│   └── Vision_Language_Models.md
├── 03_Training_&_Alignment/
│   ├── Distributed_Training.md
│   ├── RLHF_&_DPO.md
│   └── SFT_&_PEFT.md
├── 04_Inference_&_Engineering/
│   ├── KV_Cache_Optimization.md
│   ├── Quantization.md
│   └── vLLM_&_PagedAttention.md
├── 05_Interview_&_Projects/
│   ├── High_Freq_Questions.md
│   ├── My_AI4Science_Project.md
│   └── My_VRP_Project.md
├── 06_Agent_&_Applications/
│   ├── Agent.md
│   ├── MCP.md
│   └── Skills.md
├── assets/
├── LICENSE
└── README.md
```
