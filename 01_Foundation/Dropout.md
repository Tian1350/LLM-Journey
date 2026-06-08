# Dropout概述

## Dropout 到底在做什么？

Dropout 是一种极其简单却高效的正则化手段，主要用于缓解模型的过拟合问题。

在每次训练的前向传播中，它会以一定概率（如 $p=0.1$）随机将部分神经元的**输出（激活值）置为 0**。 
> **避坑指南**：这里置零的是神经元的“输出信号”，而不是底层的“权重参数”。

**本质作用**：
这种随机丢弃使得模型在每次训练时，实际上都在学习一个不同的、更瘦的**子网络**。这就强迫模型不能过度依赖某几个特定的神经元特征，必须让所有神经元都学到有用的信息，从而提升了模型的泛化能力和鲁棒性。

---

## Dropout 在 Transformer 中的具体位置

在标准的 Transformer 架构中，Dropout 主要放置在以下三个关键节点，分别承担着不同的任务：

### 1. Embedding 层之后
* **具体位置**：在词嵌入（Token Embeddings）与位置编码（Positional Encodings）相加之后，进入 Encoder/Decoder 之前。
* **设计目的**：防止模型在训练初期对某些特定的Token特征维度产生过度依赖，强迫模型提取更具泛化性的输入特征。

### 2. Attention 计算内部
* **具体位置**：在得到注意力分数矩阵后，在与Values相乘之前应用。
* **设计目的**：随机屏蔽掉一部分 Token 之间的注意力连接。这相当于告诉模型：不要每次都死盯着那几个高分 Token 看，尝试去挖掘一下上下文中其他 Token 的价值。

### 3. 子层输出与残差连接之间
* **具体位置**：在MHA的输出投影层之后，以及FFN的输出之后，**紧挨着加回到Residual Connection之前**应用。
* **设计目的**：这是最典型的应用位置。它通过对新生成的非线性特征进行随机掩码，既保留了残差流中的原始信息，又防止了网络对每一层新提取的高维信息过度依赖。

---

## Zoneout：一种用于RNN的正则化手段

Zoneout是专门针对RNN设计的一种正则化手段，是一种时间维度的Dropout。它会在每个时刻，以一定概率随机令部分隐藏单元保持上一刻的值。

**核心本质**：
在时间展开的网络中，动态且随机地引入了**恒等映射**，相当于时间维度上的残差连接。

---

## 参考资料

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., 2014)](https://jmlr.org/papers/v15/srivastava14a.html)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations (Krueger et al., 2016)](https://arxiv.org/abs/1606.01305)