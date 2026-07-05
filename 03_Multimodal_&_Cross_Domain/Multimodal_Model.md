# 多模态大模型（MLLM）

## 什么是多模态大模型

多模态大模型是指能够接收和/或生成多种模态数据（文本、图像、音频、视频等）的大语言模型。核心目标是让 LLM 不止"看懂文字"，还能理解图片、听懂语音、生成图像等。

需要区分两种能力：
- **多模态理解**：接收非文本输入，输出文本分析（如 GPT-4o 看图回答问题）
- **多模态生成**：根据文本/其他模态生成非文本输出（如根据描述生成图片、视频）

目前大部分 MLLM 侧重理解，少数同时具备生成能力。

---

## 典型架构

多模态理解类 MLLM 的主流架构可以抽象为三个模块：

```
原始输入（图片/音频/视频）
      │
      ▼
[Modality Encoder]  ── 将原始数据编码为特征序列
      │
      ▼
[Projection / Adapter]  ── 将特征对齐到 LLM 的 embedding 空间
      │
      ▼
[LLM Backbone]  ── 拼接文本 token 和视觉 token，统一处理
      │
      ▼
文本输出
```

### 1. Modality Encoder（模态编码器）

将原始信号转为高维特征表示。通常使用在大规模数据上预训练好的编码器：

| 模态 | 常用编码器 | 输出形式 |   
|------|-----------|---------|
| 图像 | ViT（CLIP 预训练）、SigLIP | 一组 patch token 序列 |
| 音频 | Whisper encoder | 帧级特征序列 |
| 视频 | ViT 逐帧 + 时序采样/池化 | 多帧 token 序列 |

关键点：编码器通常是冻结的或轻微微调的，它的质量直接决定了模型能"看到"多少信息。

### 2. Projection / Adapter（投影层）

模态编码器的输出维度和 LLM 的 embedding 维度通常不一致，需要一个映射模块来对齐。常见方案：

| 方案 | 做法 | 代表模型 |
|------|------|---------|
| 线性投影 | 一层或两层 MLP | LLaVA |
| Cross-attention（Perceiver） | 用少量可学习 query 从编码器特征中提取信息 | Flamingo、Qwen-VL |
| Q-Former | BERT 式结构 + cross-attention 抽取固定数量 token | BLIP-2、InternVL |
| 卷积下采样 | 降低 token 数量减少 LLM 计算压力 | LLaVA-1.5（用了 MLP + 池化） |

这一层的设计直接影响送入 LLM 的视觉 token 数量——token 太多会占用上下文窗口，太少会丢失细节。

### 3. LLM Backbone

接收拼接后的多模态 token 序列（文本 token + 视觉 token），用标准的自回归方式处理并生成文本输出。

常用底座：LLaMA、Qwen、InternLM、Vicuna 等开源模型。

---

## 训练流程

多模态模型的训练通常分阶段：

| 阶段 | 目标 | 训练数据 | 哪些模块更新 |
|------|------|---------|------------|
| 预训练对齐 | 让投影层学会将视觉特征映射到 LLM 能理解的空间 | 大规模图文对（如 CC3M） | 只训投影层，冻结编码器和 LLM |
| 指令微调 | 让模型学会按指令完成多模态任务 | 多模态指令数据（问答、描述、推理） | 投影层 + LLM（可能全参数或 LoRA） |

部分模型会加入第三阶段：高质量数据的 RLHF/DPO 对齐。

---

## 多模态生成

除了理解，一些模型还具备生成非文本内容的能力：

| 方案 | 做法                                       | 代表 |
|------|------------------------------------------|------|
| 外挂生成模型 | LLM 输出文本描述Projector → 调用独立的模态生成模型（如扩散模型） | 早期方案，模态间相对割裂 |
| 统一 token 化 | 把图像/音频也量化为离散 token，和文本一起自回归生成            | Chameleon（Meta）、Gemini |
| 输出投影 + 解码器 | LLM 输出特殊 embedding → 解码为图像/音频            | Emu、SEED |

统一 token 化是目前最受关注的方向——真正实现"any-to-any"的模态转换，但训练代价也最高。

---

## 代表模型

| 模型 | 机构 | 架构特点 |
|------|------|---------|
| GPT-4o | OpenAI | 原生多模态（文本+图像+音频），统一模型 |
| Gemini | Google | 原生多模态训练，支持超长上下文 |
| LLaVA | 学界 | ViT + 线性投影 + Vicuna/LLaMA，简洁高效 |
| Qwen-VL / Qwen2-VL | 阿里 | ViT + Cross-attention + Qwen，支持动态分辨率 |
| InternVL | 上海AI Lab | 大 ViT（6B）+ InternLM，强调视觉编码器能力 |
| BLIP-2 | Salesforce | 冻结编码器 + Q-Former + 冻结 LLM，训练极轻量 |

---

## 参考资料

- [Visual Instruction Tuning - LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485)
- [BLIP-2 (Li et al., 2023)](https://arxiv.org/abs/2301.12597)
- [Flamingo (Alayrac et al., 2022)](https://arxiv.org/abs/2204.14198)
- [A Survey on Multimodal Large Language Models (Yin et al., 2024)](https://arxiv.org/abs/2306.13549)
- [Qwen-VL (Bai et al., 2023)](https://arxiv.org/abs/2308.12966)
