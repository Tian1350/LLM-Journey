# Diffusion LLM（扩散语言模型）

## 是什么

Diffusion LLM（DLLM）是用扩散模型的范式来做文本生成的语言模型。和主流的自回归（Autoregressive, AR）LLM 最根本的区别在于生成方式：

- **自回归 LLM**：从左到右逐个生成 token，每个 token 依赖前面所有已生成的 token
- **扩散 LLM**：把待生成区域初始化为全 [MASK]，并行地、迭代地去噪/填充，逐步得到完整文本

> 关键区别：AR 是"串行地写"，DLLM 是"并行地改"。

需要澄清一点：文本是离散的，不能像图像那样加高斯噪声。扩散语言模型用的是**掩码扩散（Masked Diffusion）**——"加噪"等于把 token 替换成 [MASK]，"去噪"等于预测被 mask 的 token。这和图像扩散的连续高斯噪声不是一回事，但"逐步破坏 → 学习反向恢复"的思想一致。

---

## 生成流程

以掩码扩散为例：

**训练时：**
```
干净文本 → 按某个比例随机 mask 掉部分 token → 模型预测被 mask 的 token
```
mask 比例从 0% 到 100% 变化（对应不同的"噪声强度"），模型学会在任意 mask 程度下还原文本。

**推理时（带 prompt 的条件生成）：**

关键点：**prompt 部分始终保持可见、永不被 mask，只有待生成的回答区从全 [MASK] 开始迭代**。prompt 充当"条件信号"，引导去噪方向——这也是为什么不同 prompt 会得到不同结果，而不是千篇一律。

```
初始：[Prompt: 写一句关于猫的话] [MASK][MASK][MASK]...[MASK]
       └──── 固定，作为条件 ────┘ └──── 回答区，待去噪 ────┘
  │
  ▼ 第 1 步：模型看着 prompt，对回答区所有位置同时预测，锁定高置信 token
[Prompt...] [MASK] 猫  [MASK] 老鼠 [MASK]
  │
  ▼ 第 2 步：保留已确定的，重新预测剩余 [MASK]
[Prompt...] 一只 猫 在 追 老鼠 [MASK]
  │
  ▼ 迭代若干步直到没有 [MASK]
[Prompt...] 一只 猫 在 追 老鼠 。
```

每一步都是对回答区并行预测（借助双向注意力同时看到 prompt 和已填入的 token），通过多步迭代逐渐"锁定"高置信 token、修正低置信位置。生成步数是可调的——步数越多质量越好，步数越少速度越快。

> 补充：如果连 prompt 都不给（回答区和输入区全 [MASK]），那就是**无条件生成**——模型从数据分布中采样出一段合理文本，此时结果才是"不针对特定输入"的。平时带 prompt 的推理属于**条件生成**，prompt 决定了输出。

---

## DLLM vs 自回归 LLM

| 维度 | 自回归 LLM | 扩散 LLM |
|------|-----------|---------|
| 生成方向 | 严格从左到右 | 并行，全局同时生成 |
| 上下文 | 单向（只能看左边） | 双向（能看到整个序列） |
| 生成步数 | 等于序列长度（N 个 token N 步） | 可调（几步到几十步，与长度解耦） |
| 纠错能力 | 弱，生成的 token 无法回改 | 强，后续步骤可以修改前面的 token |
| KV Cache | 天然适用，推理高效 | 不直接适用（每步都要重算整个序列） |
| 生成长度 | 天然可变长 | 通常需要预设长度或额外机制 |
| 成熟度 | 高，生态完善 | 早期探索阶段 |

**DLLM 的潜在优势：**

1. **并行解码可能更快**：AR 生成 1000 个 token 必须跑 1000 次前向；DLLM 理论上几十步就能生成整段，长文本场景有加速潜力
2. **双向上下文**：每个位置都能看到全局，对需要全局规划的任务（如结构化输出）更友好
3. **可修改/可纠错**：不像 AR"一言既出驷马难追"，DLLM 可以在迭代中推翻之前的决定

**DLLM 的劣势：**

1. **质量仍落后**：同等规模下，DLLM 的语言建模质量目前普遍不如顶尖 AR 模型
2. **无法直接用 KV Cache**：AR 的 KV Cache 极大加速了推理，DLLM 每步都要重新计算整个序列，单步成本高
3. **长度处理麻烦**：AR 遇到 [EOS] 自然停止，DLLM 通常要预设生成长度或引入额外机制

---

## 在代码编程上的优势

代码场景恰好能发挥 DLLM 的长处：

1. **填空/中间插入（Infilling）天然契合**
   - 写代码经常需要"在已有代码中间补一段"，AR 模型做中间填充比较别扭（需要专门的 FIM 训练），而 DLLM 的掩码机制天生就是填空——把要补的位置设为 [MASK] 即可

2. **全局结构感知**
   - 代码有强结构约束（括号匹配、变量作用域、类型一致），双向上下文让 DLLM 在生成时能同时看到前后文，更容易保持全局一致性

3. **可迭代修改**
   - 代码修改常是"改一处、连带调整多处"，DLLM 的迭代去噪范式适合这种"反复修订直到自洽"的过程

4. **并行生成长代码块**
   - 生成一整个函数/文件时，并行解码的速度优势更明显

---

## 代表模型

### LLaDA 系列

**LLaDA（Large Language Diffusion with mAsking，2025）** 是首个在规模上（8B）证明扩散范式可以与自回归 LLM 竞争的工作。

核心做法：
- 用掩码扩散做预训练——随机 mask 输入 token，训练模型预测被 mask 的部分
- 架构仍是标准 Transformer，但**去掉了因果掩码**（因为不需要从左到右），使用双向注意力
- 推理时从全 [MASK] 序列出发，多步迭代解码

意义：LLaDA-8B 在多项 benchmark 上接近同规模的 LLaMA3-8B，说明扩散语言模型不是玩具，具备 scale up 的潜力。它挑战了"大模型必须是自回归"的固有认知。

### 其他工作

- **Mercury（Inception Labs，2025）**：商业化的扩散代码模型，主打生成速度（宣称比 AR 快数倍）
- **DiffuLLaMA / DiffuGPT**：把已有的自回归模型改造/适配为扩散模型的尝试
- **SEDD（Score Entropy Discrete Diffusion）**：离散扩散的理论改进，用 score entropy 目标提升质量

---

## 参考资料

- [Large Language Diffusion Models - LLaDA (Nie et al., 2025)](https://arxiv.org/abs/2502.09992)
- [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution - SEDD (Lou et al., 2023)](https://arxiv.org/abs/2310.16834)
- [Structured Denoising Diffusion Models in Discrete State-Spaces - D3PM (Austin et al., 2021)](https://arxiv.org/abs/2107.03006)
- [Scaling up Masked Diffusion Models on Text (Nie et al., 2024)](https://arxiv.org/abs/2410.18514)
