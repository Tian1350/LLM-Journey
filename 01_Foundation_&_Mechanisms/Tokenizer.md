# Tokenizer（分词器）

## 是什么

LLM 无法直接处理原始文本，它处理的是 token（词元）——离散的整数 ID。Tokenizer 就是负责在"文本"和"token ID 序列"之间双向转换的工具：

```
文本 "Hello world"
  │ encode（分词 + 查表）
  ▼
[15496, 995]        # token ID 序列
  │ 送入模型 → 模型输出 token ID
  ▼ decode（查表 + 拼接）
文本 "Hello world"
```

Tokenizer 是在训练前就确定好的（在语料上"训练"出一套词表和合并规则），它不参与模型的梯度更新，但它的设计直接影响模型的词表大小、序列长度和对不同语言的处理能力。

---

## 三种分词粒度

| 粒度 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **Word（词级）** | 按单词/空格切分 | 语义完整，序列短 | 词表巨大；无法处理未登录词（OOV）；不适合中文等无空格语言 |
| **Character（字符级）** | 按单个字符切分 | 词表极小；无 OOV 问题 | 序列过长；单个字符语义弱，模型学习负担重 |
| **Subword（子词级）** | 介于两者之间，把词拆成有意义的子词片段 | 平衡词表大小和序列长度；能组合出未见过的词 | 需要额外的训练过程得到切分规则 |

子词级是现代 LLM 的标准选择。核心思想：常见词保留为整体（如 "the"、"ing"），罕见词拆成子词片段（如 "tokenization" → "token" + "ization"），既控制了词表规模，又能通过子词组合泛化到新词。

---

## 主流子词算法

### BPE（Byte-Pair Encoding）

从字符开始，**统计相邻 token 对的出现频率，反复合并最高频的一对**，直到达到目标词表大小。

```
初始：l o w e r
统计发现 "e r" 最高频 → 合并为 "er"：l o w er
下一轮 "l o" 最高频 → 合并为 "lo"：lo w er
... 直到词表大小达标
```

代表：GPT-2 使用 **Byte-level BPE**——在字节（byte）而非字符层面做 BPE。好处是任何 Unicode 字符都能用 256 个字节表示，彻底消除 OOV 问题（任何文本都能被编码），GPT 系列、LLaMA 都采用这一方案。

### WordPiece

和 BPE 类似也是自底向上合并，但**合并准则不是频率，而是似然增益**：选择合并后能最大化语料似然的 token 对。

直观理解：优先合并那些"经常一起出现、且合并后更有意义"的对，而不是单纯最高频的对。评分近似为 $\frac{\text{freq}(AB)}{\text{freq}(A) \times \text{freq}(B)}$。

代表：BERT 系列。

### Unigram

思路和 BPE/WordPiece 相反，是**自顶向下做减法**：
1. 先构建一个超大的候选子词集合
2. 用概率语言模型评估每个子词的重要性
3. 迭代删除对整体似然损失最小的子词，直到词表缩到目标大小

Unigram 保留了每个词的多种可能切分及其概率，推理时可以选最优切分。代表：ALBERT、T5（配合 SentencePiece）。

### SentencePiece

严格说 **SentencePiece 不是一种分词算法，而是一个工具/框架**——它内部可以用 BPE 或 Unigram 作为具体算法。

它的关键特性：
- **把空格也当作普通字符处理**（用特殊符号 `▁` 表示空格），因此不依赖预先的空格分词，能统一处理中文、日文等无空格语言
- **端到端可逆**：encode 再 decode 能完美还原原文（包括空格），不像传统分词器会丢失空格信息
- 语言无关，直接在原始文本上训练

代表：LLaMA、T5、多语言模型广泛使用。

---

## 算法对比

| 算法 | 方向 | 合并/删除准则 | 代表模型 |
|------|------|--------------|---------|
| BPE | 自底向上合并 | 最高频 token 对 | GPT 系列、LLaMA |
| WordPiece | 自底向上合并 | 最大化似然增益 | BERT |
| Unigram | 自顶向下删减 | 最小化似然损失 | T5、ALBERT |
| SentencePiece | 框架（含 BPE/Unigram） | 视所选算法而定 | LLaMA、T5、多语言模型 |

---

## 一些实践中的坑

- **数字和代码**：很多 tokenizer 把长数字拆得很碎（如 "12345" 拆成多个 token），影响算术能力；代码中的缩进、符号也会消耗大量 token
- **中文效率**：早期以英文为主训练的 tokenizer，一个汉字可能占 2-3 个 token，导致中文文本的 token 数远多于等长英文，变相抬高了成本和上下文占用
- **词表大小权衡**：词表越大，单条文本的 token 数越少（序列短、推理快），但 embedding 层和输出层的参数量越大。现代大模型词表通常在 3 万到 15 万之间（如 LLaMA 3 用了 128K 词表以提升多语言和编码效率）

---

## 参考资料

- [Neural Machine Translation of Rare Words with Subword Units - BPE (Sennrich et al., 2015)](https://arxiv.org/abs/1508.07909)
- [Japanese and Korean Voice Search - WordPiece (Schuster & Nakajima, 2012)](https://research.google/pubs/pub37842/)
- [Subword Regularization - Unigram (Kudo, 2018)](https://arxiv.org/abs/1804.10959)
- [SentencePiece: A simple and language independent subword tokenizer (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226)
