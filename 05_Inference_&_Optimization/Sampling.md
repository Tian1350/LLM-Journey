# LLM 推理中的采样策略

## 为什么需要采样

LLM 每一步的输出是词表上的一个概率分布（经 softmax 得到）。要生成文本，就要在每一步从这个分布中"选"一个 token——这个选择过程就是采样。

不同的采样策略在两个目标之间权衡：
- **确定性/质量**：选高概率 token，输出连贯但可能单调重复
- **多样性/创造性**：给低概率 token 一些机会，输出丰富但可能离题

---

## 贪心解码（Greedy Decoding）

最简单的策略：每一步直接选概率最高的 token。

$$\text{token} = \arg\max_i P(x_i)$$

特点：完全确定（同样输入永远同样输出），但容易陷入重复循环，且"局部最优"不等于"整体最优"——某一步的最高概率词可能导致后续生成质量下降。

适用：需要确定性结果的任务（如代码生成、事实问答），常配合 temperature=0 使用。

---

## Beam Search（束搜索）

不只跟踪一条路径，而是同时保留 $k$ 条概率最高的候选序列（beam width = k），最后选整体概率最高的序列。

相比贪心，beam search 考虑了序列级别的全局概率，缓解了"一步走错满盘皆输"。但在开放式生成中，它倾向于产生通用、安全、重复的文本，所以现代 LLM 的对话生成很少用，更多用于翻译、摘要等有明确目标的任务。

---

## 温度采样（Temperature Sampling）

引入温度系数 $T$，在 softmax 之前把 logits 除以 $T$，再按调整后的概率随机采样：

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

温度的作用是调节分布的"尖锐程度"：

| 温度 | 效果 |
|------|------|
| $T \to 0$ | 分布变尖锐，趋近贪心解码（确定性） |
| $T = 1$ | 保持原始分布 |
| $T > 1$ | 分布变平缓，低概率词机会增大（更随机、更有创造性） |

注意：温度本身不裁剪任何 token，只是重新调整概率。通常和 Top-k / Top-p 配合使用。

---

## Top-k 采样

只保留概率最高的 $k$ 个 token，将它们的概率重新归一化后采样，其余 token 概率置零。

优点：排除了长尾的低质量 token。
缺点：$k$ 是固定的，无法适应分布形状——当分布很尖锐时（一个词几乎占满概率），固定 k 会引入不必要的低概率词；当分布很平缓时，固定 k 又可能切掉合理的候选。

---

## Top-p 采样（核采样，Nucleus Sampling）

按概率从大到小累加，直到累计概率 $\geq p$，保留这个最小集合（称为"核"），归一化后采样。

$$\text{保留最小的集合 } V_p \text{ 使得} \sum_{x_i \in V_p} P(x_i) \geq p$$

相比 Top-k 的优势：**候选集大小是动态的**。
- 分布尖锐时（模型很确定）：少数几个 token 就累积到 p，候选集小
- 分布平缓时（模型不确定）：需要更多 token 才到 p，候选集大

这种自适应性让 Top-p 成为目前最常用的采样策略。典型值 $p = 0.9$ 或 $0.95$。

---

## 其他常用机制

| 机制 | 作用 |
|------|------|
| **Repetition Penalty** | 对已生成过的 token 降低概率，抑制重复 |
| **Frequency/Presence Penalty** | 按 token 出现频率/是否出现施加惩罚（OpenAI API 提供） |
| **Min-p** | 设定一个相对于最高概率的阈值，动态裁剪，比 Top-p 更鲁棒 |

---

## 策略对比与选择

| 策略 | 确定性 | 多样性 | 典型场景 |
|------|--------|--------|---------|
| 贪心 / T=0 | 完全确定 | 无 | 代码、数学、事实问答 |
| Beam Search | 确定 | 低 | 翻译、摘要 |
| 温度采样 | 可调 | 可调 | 通用，配合 Top-p |
| Top-k | 随机 | 中 | 早期 GPT-2 常用 |
| Top-p（核） | 随机 | 中-高 | 对话、创意写作（主流默认） |

实践中常见组合：`temperature + top-p`，例如 `temperature=0.7, top_p=0.9`。需要可复现结果时用 `temperature=0`（等价贪心）。

---

## 参考资料

- [The Curious Case of Neural Text Degeneration - Nucleus Sampling (Holtzman et al., 2019)](https://arxiv.org/abs/1904.09751)
- [Hierarchical Neural Story Generation - Top-k (Fan et al., 2018)](https://arxiv.org/abs/1805.04833)
- [Hugging Face - How to generate text: decoding methods](https://huggingface.co/blog/how-to-generate)
