# 大语言模型 (LLM) 中的核心激活函数

在 Transformer 架构与大语言模型中，激活函数的选择对模型的非线性表达能力、训练稳定性及最终性能起着决定性作用。以下是对你提到几种激活函数的原理剖析与纠偏。

## 1. GELU (Gaussian Error Linear Unit)

GELU（高斯误差线性单元）是 GPT 系列、BERT 等早期经典模型中广泛使用的激活函数。

* **核心思想**：GELU 的设计思想融合了 Dropout、Zoneout 和 ReLU 的特性，引入了随机正则化的概念。GELU的思想是让输入 $x$ 乘一个随机的伯努利变量 $m$，由于输入 $x$ 大致满足标准正态分布，设定伯努利分布的参数 $p$ 为累积分布函数，可以满足激活函数的要求（越大越保留）。
* **基本原理**：通过对上述随机过程取数学期望，可以得到一个确定的激活函数，即将其与标准正态分布的累积分布函数相乘。直观来说，随着输入 $x$ 的降低，其被“归零”或抑制的概率会逐渐升高；由于引入了正态分布特性，它的激活曲线比 ReLU 更加平滑。
* **计算公式**（工业界常用的快速近似版本）：
$$\text{GELU}(x) = 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)\right)\right)$$

## 2. SwiGLU (Swish Gated Linear Unit)

SwiGLU 是目前开源 LLM（如 LLaMA 系列、ChatGLM 等）的标配。它是 GLU（门控线性单元）的变体，使用 Swish 函数替代了传统的 Sigmoid 门控。

* **前置知识Swish 函数**：公式为 $ \text{Swish}(x) = x \cdot \sigma(\beta x) $，其中 $\sigma$ 为 Sigmoid 函数， $\beta$ 为可学习的参数。
* **SwiGLU 公式**：在 LLM 的前馈网络（FFN）实际应用中，通常会省略偏置项以提升计算和存储效率。其对输入进行两组不同的线性变换，一组经过 Swish 激活后作为“门控”，控制另一组的信息流：
$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_2)$$
* **核心优势**：
    1.  **更平滑的梯度流动**：去除了 ReLU 的硬截断，在负值区域也保留了微弱的梯度。
    2.  **增强的信息选择能力**：门控机制赋予了模型动态过滤和放缩特征的能力；Swish是一个非单调的激活函数，信息表达更细腻。
    3.  **更强的特征表达**：实验证明，相比于标准的 GELU，SwiGLU 在同等计算量下能显著提升语言建模的收敛速度和最终性能。

## 3. Softmax (归一化指数函数)

在 LLM 中，Softmax 通常**不作为**隐藏层的FFN激活函数。它主要扮演多分类概率转换或权重分配的角色。

* **主要应用场景**：
    1.  **输出层**：将模型最后一层输出的 Logits 转换为庞大词表上所有 Token 的概率分布，配合交叉熵损失函数进行预测。
    2.  **注意力机制**：将 Query 和 Key 的点积结果转化为注意力权重，确保所有特征的权重之和为 1。
* **计算公式**：对于输入向量中的第 $i$ 个元素 $x_i$，其公式为：
$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$
*(其中 $K$ 为类别的总数或序列的长度)*

---

## 参考资料

- [Gaussian Error Linear Units (GELUs) (Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1606.08415)
- [Searching for Activation Functions (Ramachandran et al., 2017)](https://arxiv.org/abs/1710.05941)
- [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)