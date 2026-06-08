# 为什么大模型都选了 Decoder-Only？

Transformer 原始论文提出了 Encoder-Decoder 结构，但后续大模型几乎清一色走向了 Decoder-Only。下面先回顾三种架构路线，再分析 Decoder-Only 胜出的原因。

---

## 三种架构路线

| 架构 | 注意力方式                 | 典型模型 | 擅长场景 |
|------|-----------------------|----------|----------|
| Encoder-Only | 双向（全可见）               | BERT, RoBERTa | 分类、NER、语义匹配等判别式任务 |
| Encoder-Decoder | Encoder双向 + Decoder因果 | T5, BART | 翻译、摘要等 seq2seq 任务 |
| Decoder-Only | 因果掩码（单向）              | GPT 系列, LLaMA, DeepSeek | 开放式文本生成、通用对话 |

**Encoder-Only** 通过双向注意力获得丰富的上下文表征，适合做"理解"类任务（打标签、判断关系），但它的预训练目标是 MLM（掩码语言模型），不具备自回归生成能力。

**Encoder-Decoder** 保留了完整的 Transformer 结构，编码端理解输入，解码端生成输出。在输入输出有明确边界的任务（翻译、摘要）上表现优异，但参数被拆分到两个模块，scale up 时效率不如单栈结构。

**Decoder-Only** 只保留不带交叉注意力的解码器，通过因果掩码让每个 token 只能看到左侧上下文，天然适配自回归生成。

---

## Decoder-Only 胜出的核心原因

1. **与 Scaling Law 高度契合**
   - 单栈结构意味着所有参数都在做同一件事（next-token prediction），增大模型规模时收益更直接，不存在 Encoder/Decoder 之间的容量分配问题。

2. **训练效率高**
   - 因果掩码让序列中每个位置都可以作为一个训练样本（预测下一个 token），同一条文本提供 N-1 个监督信号，数据利用率极高。
   - Encoder-Decoder 只有解码端产生 loss，编码端参数的梯度信号相对间接。

3. **In-Context Learning 的天然载体**
   - Decoder-Only 的推理方式就是"给前缀，续写后文"，这与 few-shot prompting 的形式完全一致——把示例拼在 prompt 里，模型自然地基于前缀生成答案，不需要额外的架构适配。

4. **工程简洁性**
   - 只有一个模块，推理时 KV Cache 管理更简单，serving 系统更容易优化（连续批处理、投机解码等技术实现成本更低）。

---

## 一句话总结

Decoder-Only 能赢不是因为它"理解能力更强"，而是因为在 scale up 的大趋势下，它的结构最简单、训练信号最密集、工程实现最友好，同时通过足够大的模型容量和足够长的上下文窗口，弥补了单向注意力在"理解"上的理论短板。

---

## 参考资料

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)
- [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization? (Wang et al., 2022)](https://arxiv.org/abs/2204.05832)