# 推荐大模型

## 背景

传统推荐系统依赖 ID embedding + 协同过滤/深度模型，存在冷启动困难、跨域迁移差、缺乏可解释性等问题。LLM 的出现带来了新思路：利用语言模型的通用知识和推理能力来增强或重构推荐系统。

---

## 推荐系统基本流程

```
数据采集 → 特征工程 → 召回 → 粗排 → 精排 → 后处理（混排/规则/策略） → 展示
```

各阶段职责：

| 阶段 | 输入规模 | 做什么 |
|------|---------|--------|
| **召回** | 百万/千万级物品 → 数百~数千 | 用轻量策略快速筛选候选集（协同过滤、向量召回、图召回等） |
| **粗排** | 数百~数千 → 数百 | 用中等复杂度的模型初步打分排序 |
| **精排** | 数百 → 数十 | 用复杂模型精确预测点击率/转化率，按得分排序 |
| **后处理** | 数十 → 最终展示 | 多样性打散、业务规则（如新品加权、广告插入）、合规过滤 |

特征工程中，用户侧特征（历史行为、画像）和物品侧特征（属性、内容）被编码为稠密向量。这一步类似于 NLP 中的 tokenization + embedding，但对象是用户行为序列和物品属性。

---

## 推荐大模型的主要流派

### 流派一：LLM 增强传统推荐（LLM4Rec）

保留传统推荐架构，用 LLM 增强其中某些环节：

| 增强方式 | 做法 | 代表工作 |
|---------|------|---------|
| 特征增强 | 用 LLM 生成物品的文本描述/标签/知识，丰富物品表征 | KAR（用 LLM 生成开放世界知识作为推荐特征） |
| 重排与解释 | 传统模型召回候选，LLM 根据对话上下文重排并生成推荐理由 | Chat-REC |
| 用户意图理解 | 用 LLM 理解用户的自然语言偏好描述，转化为结构化偏好 | — |
| 冷启动 | 利用 LLM 的世界知识为新物品/新用户生成初始表征 | — |
| LLM 微调为推荐模型 | 将推荐任务包装为自然语言指令，微调 LLM 做排序/评分 | TALLRec、InstructRec |

**Chat-REC** 的思路：传统推荐模型负责召回候选集，LLM 作为交互层——接收用户的自然语言对话，结合候选集做重排和解释。好处是推荐变得可交互（"为什么推荐这个？""我不想看恐怖片"），缺点是 LLM 推理延迟高，难以用于实时大规模 serving。

### 流派二：生成式推荐（Generative Recommendation）

彻底用生成模型的范式重构推荐——把"从候选集中选"变成"直接生成推荐物品"。

**核心问题**：LLM 生成的是 token 序列，怎么让它生成"物品"？

**解决方案：语义 ID（Semantic ID）**

传统推荐中每个物品有一个无语义的数字编号（Item ID = 12345），模型只能通过训练学到这个 ID 的含义。语义 ID 的做法是把每个物品编码为一段有意义的离散 token 序列，这样：

- 语义相近的物品 → ID 序列有共同前缀 → 模型可以泛化
- 生成式模型可以自回归地逐 token 生成物品 ID

| 对比 | Item ID | Semantic ID |
|------|---------|-------------|
| 形式 | 单个无语义数字（12345） | 离散 token 序列（[3, 17, 42, 8]） |
| 语义性 | 无，纯查找表 | 有，相似物品共享 token 前缀 |
| 泛化能力 | 无法泛化到未见过的 ID | 组合式泛化（新 token 序列 = 新物品） |

**如何生成语义 ID：RQ-VAE（Residual Quantized VAE）**

```
物品多模态特征 → Encoder → 连续语义向量 → 多级残差量化 → 离散 token 序列（语义 ID）
```

为什么不直接用连续向量做推荐，而要量化为离散 ID？
- 自回归生成需要离散 token 空间（像生成文字一样逐个生成）
- 离散化后可以构建层次化的 token 树，加速检索
- 连续向量的精确匹配不如离散 ID 的生成范式灵活

**TIGER**（Google, 2023）：

完整流程：RQ-VAE 为每个物品生成语义 ID → 用用户行为序列（历史交互物品的语义 ID 序列）训练一个 Transformer → 自回归生成下一个推荐物品的语义 ID。

本质上把推荐问题转化为了序列到序列的生成问题。

**OneRec**（快手, 2024）：

端到端的生成式推荐模型：
- 统一的 Tokenizer 处理多模态物品特征，采用了 QARM 的思路
- MoE + Transformer 架构
- 直接从用户历史序列生成推荐，不需要传统的召回/排序多阶段流水线

---

## 两种流派对比

| 维度 | LLM4Rec | 生成式推荐 |
|------|---------|-----------|
| 对传统架构的改动 | 小，LLM 作为增强模块嵌入 | 大，重构整个推荐范式 |
| 部署难度 | 低，可渐进式引入 | 高，需要重建基础设施 |
| 实时性 | 受限于 LLM 推理延迟 | 可优化（模型相对轻量） |
| 冷启动能力 | 强（利用 LLM 世界知识） | 中等（依赖语义 ID 的泛化） |
| 工业落地 | 已有应用（对话式推荐、解释生成） | 仍在早期探索阶段 |

---

## 参考资料

- [Recommender Systems with Generative Retrieval - TIGER (Rajput et al., 2023)](https://arxiv.org/abs/2305.05065)
- [Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System (Gao et al., 2023)](https://arxiv.org/abs/2303.14524)
- [Towards Open-World Recommendation with Knowledge Augmentation from LLMs - KAR (Xi et al., 2023)](https://arxiv.org/abs/2306.10933)
- [TALLRec: An Effective and Efficient Tuning Framework to Align LLM with Recommendation (Bao et al., 2023)](https://arxiv.org/abs/2305.00447)
- [A Survey on Large Language Models for Recommendation (Wu et al., 2024)](https://arxiv.org/abs/2305.19860)
- [OneRec (Kuaishou, 2024)](https://arxiv.org/abs/2502.18965)