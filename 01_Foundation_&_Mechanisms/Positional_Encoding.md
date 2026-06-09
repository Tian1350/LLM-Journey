# 位置编码

## 为什么需要位置编码

注意力机制本身是置换不变的（permutation invariant）：把输入 token 打乱顺序，注意力的输出只是跟着重排，计算结果不会"察觉"顺序变化。但语言是有序的——"猫追狗"和"狗追猫"意思完全不同。位置编码的作用就是把序列顺序信息注入模型。

---

## 绝对位置编码

给序列中每个位置分配一个固定的向量，与 token embedding 相加后送入模型。

### 正弦/余弦位置编码（Sinusoidal PE）

原始 Transformer 的方案：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**设计动机：**

- 不同维度使用不同频率的正余弦波，低维变化快（捕捉局部位置差异），高维变化慢（捕捉全局位置趋势）
- 任意位置 $pos+k$ 的编码可以表示为 $pos$ 编码的线性变换（通过三角恒等式），模型可以通过线性操作学到相对位置关系
- 值域有界（[-1, 1]），对未见过的更长序列有一定外推能力

**局限：**

- 位置信息与语义信息通过加法混合，模型需要自己学会区分哪些是"位置成分"、哪些是"语义成分"
- 注意力分数 $q \cdot k$ 展开后，位置-位置、位置-语义、语义-语义三项纠缠在一起，模型无法显式利用相对位置信息

### 可学习位置编码（Learned PE）

将位置向量作为可训练参数，训练中和其他参数一起优化。BERT、GPT-2 采用此方案。优点是模型可以自己决定位置的最优表示；缺点是无法外推到训练时未见过的长度。

---

## 相对位置编码

不给每个位置一个绝对向量，而是在注意力计算中直接编码 token 之间的相对距离。

### RoPE（Rotary Position Embedding）

RoPE 是目前主流开源模型（LLaMA、Qwen、DeepSeek 等）的标准选择。

**核心思路：** 对 Q 和 K 向量施加一个与位置相关的旋转操作，使得两个 token 的注意力分数只依赖于它们的相对位置差。

具体地，将 $d$ 维向量视为 $d/2$ 个二维子空间，在每个子空间中按位置角度旋转：

$$\tilde{q}_m = R(\theta_m) \cdot q, \quad \tilde{k}_n = R(\theta_n) \cdot k$$

其中 $R(\theta)$ 是二维旋转矩阵，$\theta_m$ 与位置 $m$ 和维度频率有关。计算注意力时：

$$\tilde{q}_m^T \tilde{k}_n = q^T R(\theta_n - \theta_m)^T k$$

结果只依赖相对位置 $n - m$，与绝对位置无关。

**RoPE 的优势：**
- 注意力分数天然只依赖相对位置，不需要额外的偏置项
- 可以结合 NTK-aware 插值、YaRN 等技术扩展到训练长度之外
- 实现高效——旋转操作等价于按元素乘一组正余弦值，无额外参数

### ALiBi（Attention with Linear Biases）

不修改 Q/K 向量，而是在注意力分数上直接减去一个与距离成正比的惩罚：

$$\text{score}(i, j) = q_i^T k_j - m \cdot |i - j|$$

其中 $m$ 是每个头预设的斜率（不可学习）。距离越远惩罚越大，自然实现局部偏好。ALiBi 外推能力强，但表达力不如 RoPE 灵活。

---

## 各方案对比

| 方案 | 编码方式 | 是否可外推 | 代表模型 |
|------|---------|-----------|---------|
| Sinusoidal | 绝对，加到 embedding | 有限 | 原始 Transformer |
| Learned | 绝对，加到 embedding | 不可 | BERT, GPT-2 |
| RoPE | 相对，旋转 Q/K | 可（需配合插值） | LLaMA, Qwen, DeepSeek |
| ALiBi | 相对，注意力分数偏置 | 强 | BLOOM, MPT |

---

## 参考资料

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
- [Train Short, Test Long: Attention with Linear Biases (Press et al., 2021)](https://arxiv.org/abs/2108.12409)
- [Eleuther AI - Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/)
