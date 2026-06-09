# 视觉语言模型（VLM）

## 概述

VLM 是多模态大模型中最核心的一个子方向——处理图像（视觉）和文本（语言）两种模态的联合理解与生成。视觉是人类获取信息的主要通道，也是 AI 应用场景最丰富的模态之一。

VLM 要解决的核心问题：**如何让语言模型"看懂"图片**——即把视觉信息转化为 LLM 能理解的表示。

---

## Vision Transformer（ViT）

ViT 是几乎所有现代 VLM 的视觉编码器基础。

### 架构

```
输入图像 (H×W×3)
    │
    ▼
切分为 N 个 Patch (每个 P×P×3)
    │
    ▼
每个 Patch 经线性投影 → Patch Embedding (维度 d)
    │
    ▼
加上位置编码 + 拼接 [CLS] token
    │
    ▼
标准 Transformer Encoder (多层 Self-Attention + FFN)
    │
    ▼
输出：[CLS] token 表征 或 所有 patch token 序列
```

关键设计：
- 把图像切成 $N = (H/P) \times (W/P)$ 个 patch，每个 patch 展平后通过线性投影映射为 $d$ 维向量——这一步等效于一个 stride=P 的卷积操作
- 加上可学习的位置编码（或 2D 正弦编码），让模型知道每个 patch 在图像中的空间位置
- 拼接 [CLS] token，一个可学习的嵌入向量，用来表征图像的全局特征
- 之后就是标准的 Transformer Encoder，和处理文本的方式一致

### 为什么 VLM 普遍选择 ViT

1. **架构统一性**：ViT 和 LLM 都是 Transformer，输出都是 token 序列，对接自然——视觉 token 和文本 token 可以直接拼在一起送入 LLM
2. **Scaling 表现好**：ViT 的性能随模型规模和数据量稳定提升，适合大规模预训练
3. **生态成熟**：CLIP、SigLIP、DINOv2 等预训练好的 ViT checkpoint 丰富，可以直接拿来用

---

## VLM 发展路线

### CLIP（2021, OpenAI）

CLIP 不是一个生成模型，而是一个**对齐模型**——学习图像和文本之间的语义对应关系。

**架构：**
- Image Encoder：ViT（或 ResNet）
- Text Encoder：Transformer

**训练方式：对比学习**

给定一个 batch 的图文对 $(I_i, T_i)$，训练目标是让匹配的图文对相似度高，不匹配的相似度低：

$$\mathcal{L} = -\frac{1}{N}\sum_i \log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_j \exp(\text{sim}(I_i, T_j)/\tau)}$$

**CLIP 的意义：**
- 在 4 亿图文对上训练，学到了极其通用的视觉-语言对齐空间
- 零样本分类：把类别名转化为文本 prompt（如"a photo of a dog"），和图像 embedding 算相似度
- 更重要的是：CLIP 预训练的 ViT 成为了后续几乎所有 VLM 的默认视觉编码器

注意：CLIP 的训练目标是图文对比（ITC），不是图像描述生成，也不是目标检测。它学的是全局语义对齐，不是细粒度理解。

### BLIP / BLIP-2 / InstructBLIP（2022-2023, Salesforce）

**BLIP** 的核心贡献是统一了理解和生成任务，同时提出了数据自举（CapFilt）策略：

- 用模型自己生成图像描述（Captioner），再用另一个模型过滤低质量描述（Filter）
- 用清洗后的数据反过来继续训练，迭代提升数据质量

训练目标三合一：
| 任务 | 目标 | 作用 |
|------|------|------|
| ITC（Image-Text Contrastive） | 图文对比对齐 | 学习全局对齐 |
| ITM（Image-Text Matching） | 判断图文是否匹配（二分类） | 学习细粒度交互 |
| LM（Language Modeling） | 给定图像生成描述文本 | 学习生成能力 |

**BLIP-2** 的关键改进——Q-Former：

解决的问题：ViT 输出的 patch token 太多（如 256 个），直接送入 LLM 占用大量上下文。

做法：用一组少量的可学习 query 向量（如 32 个），通过 cross-attention 从 ViT 输出中"提取"关键视觉信息，压缩为固定数量的视觉 token。

```
ViT 输出 (256 tokens) → Q-Former (32 learned queries × self-attn × cross-attn) → 32 个视觉 token → LLM
```

好处：视觉编码器和 LLM 都可以冻结，只训 Q-Former（参数很少），训练成本极低。

**InstructBLIP**（2023）——在 BLIP-2 基础上引入指令微调：

BLIP-2 的 Q-Former 虽然能提取视觉信息，但它不理解用户的具体意图——不管你问什么，它提取的视觉 token 都是一样的。InstructBLIP 的改进：

- 把用户指令（instruction）也送入 Q-Former 的 self-attention，让 query 向量在提取视觉信息时"知道用户想要什么"
- 在 26 个视觉-语言数据集上做指令微调，覆盖问答、描述、推理等多种任务类型

```
BLIP-2:        ViT → Q-Former(queries) → 视觉 token → LLM(instruction + 视觉 token)
InstructBLIP:  ViT → Q-Former(queries + instruction) → 指令感知的视觉 token → LLM(instruction + 视觉 token)
```

关键区别：Q-Former 的提取过程变成了指令条件化的（instruction-aware），不同的问题会导致提取不同的视觉信息。这在需要关注图像特定区域的任务上（如"图中左边的人在做什么"）提升明显。

### LLaVA（2023, 学界）

LLaVA 的哲学是"简单即有效"：

- 用 CLIP ViT 作为视觉编码器
- **一个线性投影层**（后续版本改为两层 MLP）直接把 ViT 输出映射到 LLM embedding 空间
- 拼上文本 token，送入 LLaMA/Vicuna 做自回归生成

```
图像 → CLIP ViT → MLP 投影 → 视觉 token
                                    ↓
                          [视觉 token, 文本 token] → LLM → 输出
```

对比 BLIP-2 的 Q-Former，LLaVA 证明了：当 LLM 足够强时，复杂的对齐模块不是必需的——简单的线性投影 + 足够好的指令微调数据就够了。

LLaVA 的训练数据也是一个重要贡献：用 GPT-4 对图像描述生成多轮对话数据，开创了"用强模型生成多模态指令数据"这条路。

---

## 技术路线对比

| 模型 | 视觉编码器 | 对齐方式 | LLM | 核心取舍 |
|------|-----------|---------|-----|---------|
| CLIP | ViT | 对比学习 | 无 | 只做对齐，不做生成 |
| BLIP-2 | 冻结 ViT | Q-Former | 冻结 LLM | 训练成本极低，但 Q-Former 可能成为信息瓶颈 |
| LLaVA | 冻结/微调 ViT | MLP 线性投影 | 微调 LLM | 简单直接，依赖 LLM 能力和数据质量 |
| Qwen-VL | ViT (自训) | Cross-attention | Qwen | 动态分辨率，支持细粒度定位 |
| InternVL | 大 ViT (6B) | MLP | InternLM | 强调视觉编码器本身的能力上限 |

---

## 参考资料

- [An Image is Worth 16x16 Words: ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [Learning Transferable Visual Models From Natural Language Supervision - CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [BLIP-2 (Li et al., 2023)](https://arxiv.org/abs/2301.12597)
- [InstructBLIP: Towards General-purpose Vision-Language Models (Dai et al., 2023)](https://arxiv.org/abs/2305.06500)
- [Visual Instruction Tuning - LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485)
- [Qwen-VL (Bai et al., 2023)](https://arxiv.org/abs/2308.12966)
