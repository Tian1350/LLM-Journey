# Transformer 模型在各领域的损失函数总结

## 1. LLM的损失函数

LLM 的核心任务是**自回归语言建模**（Next-token Prediction）。其本质是在每一步预测时，将庞大的词表（Vocabulary）看作候选类别，执行一个维度极高的多分类任务。

* **损失函数**：**交叉熵损失函数（Cross-Entropy Loss）**。
* **原理解析**：模型输出词表中每个 token 的概率分布，交叉熵用于衡量该预测分布与真实标签（One-hot 编码）之间的差异。在实际训练中，为了防止过拟合和提高泛化能力，有时也会结合标签平滑（Label Smoothing）。
* **公式**：$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^{V} y_c \log(p_c)$，其中 $V$ 为词库大小，$y_c$ 为真实标签，$p_c$ 为模型预测的概率。

## 2. Transformer 在其他领域的应用与损失函数

Transformer 架构跨界到不同模态后，损失函数的设计主要取决于**具体的下游任务类型**（分类、回归、匹配等）。

### 2.1 计算机视觉 (CV)
* **自监督学习与图像生成（如 MAE, DiT）**：
    * 任务目标是重建被掩码的像素值或预测扩散过程中的噪声，属于回归任务。
    * **损失函数**：**均方误差（MSE, $L_2$ Loss）** 或 **平均绝对误差（MAE, $L_1$ Loss）**。
* **目标检测（如 DETR）**：
    * **边框回归（BBox Regression）**：目前主流采用 **$L_1$ Loss** 结合 **GIoU Loss（Generalized Intersection over Union）**。
    * **分类识别**：采用 **交叉熵损失**；若存在严重的类别或正负样本不平衡，通常会替换为 **Focal Loss**。

### 2.2 时间序列预测 (TSF)
* **确定性预测（点预测）**：
    * 任务是预测未来的具体数值（连续变量）。
    * **损失函数**：最基础的是 **均方误差（MSE）**。为了提高对异常值（Outliers）的鲁棒性，也常采用 **$L_1$ Loss** 或 **Huber Loss**。
* **概率性预测（区间预测）**：
    * 任务是输出预测的置信区间或概率分布（如应用于金融或气象预测）。
    * **损失函数**：通常使用 **分位数损失（Quantile Loss）** 或 **负对数似然（Negative Log-Likelihood, NLL）**。

### 2.3 自然语言处理 (NLP) 的其他任务
* **双向掩码建模（如 BERT）**：同样使用**交叉熵损失**来预测被 Mask 的特定词汇。
* **文本表示与对比学习（如 Sentence-BERT, CLIP 的文本端）**：使用 **InfoNCE Loss**（本质也是一种基于交叉熵的变体，用于在特征空间中拉近正样本、推远负样本）。

---

## 参考资料

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [End-to-End Object Detection with Transformers (Carion et al., 2020)](https://arxiv.org/abs/2005.12872)
- [Masked Autoencoders Are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377)
- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (Zhou et al., 2021)](https://arxiv.org/abs/2012.07436)
