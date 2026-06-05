# BERT 家族

## BERT 的核心思路

BERT（Bidirectional Encoder Representations from Transformers）是 Encoder-Only 架构的代表，2018 年由 Google 提出。

关键设计：

- **双向注意力**：每个 token 能同时看到左右上下文，获得更丰富的语义表征
- **预训练任务**：MLM（Masked Language Model，随机遮盖 15% 的 token 让模型预测）+ NSP（Next Sentence Prediction，判断两句话是否相邻）
- **两阶段范式**：先在大规模无标注语料上预训练，再在下游任务上微调——这一范式深刻影响了后续所有工作

---

## 主要变体

| 模型 | 相对 BERT 的改进 | 核心动机 |
|------|-----------------|----------|
| **RoBERTa** | 去掉 NSP；更大数据量+更长训练；动态掩码（每个 epoch 重新生成 mask） | NSP 任务收益不明确，静态掩码让模型反复见同样的 mask 模式导致欠拟合 |
| **ALBERT** | 用 SOP（Sentence Order Prediction）替代 NSP；Embedding 矩阵分解；跨层参数共享 | 大幅缩减参数量（ALBERT-xxlarge 参数仅 BERT-large 的 1/10），同时保持甚至提升性能 |

补充说明：
- RoBERTa 的结论是"BERT 本身被严重 undertrained"，在同等模型大小下多训就能大幅涨点。
- ALBERT 的参数共享不等于计算量减少——推理时每层仍然要跑，只是权重相同，所以它省显存但不省时间。

---

## 为什么 Encoder-Only 没有展现涌现能力

GPT 系列在 scale up 后出现了 in-context learning、chain-of-thought 等涌现能力，而 BERT 系列即使做大也没有类似现象。原因不是单一的，而是多个因素叠加：

1. **预训练目标的本质差异**
   - MLM 是"填空"——给定上下文预测被遮盖的局部 token，本质上是一个分类任务（从词表中选一个词）。
   - GPT 的 next-token prediction 是"续写"——模型必须学会生成连贯的长序列，这迫使它建模更长距离的依赖和推理链。

2. **无法自然地 prompting**
   - Few-shot prompting 的形式是"给前缀，续写答案"，这和自回归生成完全对齐。
   - BERT 的推理方式是"在 [MASK] 位置预测一个 token"，无法处理开放式生成，也很难把 few-shot 示例自然地编码进去。

3. **无法实现 Chain-of-Thought**
   - CoT 依赖模型逐步生成中间推理步骤，这需要自回归的序列生成能力。
   - BERT 只能并行地预测各个 [MASK] 位置，没有"逐步展开思考"的机制。

4. **Scale up 的路径不同**
   - 实践上，BERT 系列最大做到了几亿参数级别（BERT-large 3.4 亿），社区没有像 GPT 那样持续推到千亿级。这部分是因为 MLM 的 scaling 收益递减更快，部分是因为判别式任务不需要那么大的模型。

简单说：BERT 的训练目标决定了它擅长"理解"而非"生成"，而目前观察到的涌现能力几乎都依赖于自回归生成这一前提。

---

## 参考资料

- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](https://arxiv.org/abs/1907.11692)
- [ALBERT: A Lite BERT for Self-supervised Learning (Lan et al., 2019)](https://arxiv.org/abs/1909.11942)
