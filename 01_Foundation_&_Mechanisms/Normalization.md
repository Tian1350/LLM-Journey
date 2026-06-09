# Normalization

## 为什么需要归一化

深层网络训练时，每一层的输入分布会随着前面层参数的更新不断漂移（Internal Covariate Shift）。归一化的作用是把每层输入拉回到一个相对稳定的分布，带来两个直接好处：

- 允许使用更大的学习率，加速收敛
- 减少对参数初始化的敏感度，训练更稳定

---

## 归一化方式对比

### BatchNorm

沿 batch 维度归一化：对同一个特征，跨 batch 内所有样本计算均值和方差。

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

- 可学习参数：$\gamma$（缩放）和 $\beta$（偏移），每个特征维度各一对，共 $2 \times d$ 个参数
- 推理时使用训练阶段累积的滑动平均统计量
- 问题：batch size 太小时统计量不稳定；变长序列中不同样本的同一位置语义不同，跨样本统计没有意义

### LayerNorm

沿特征维度归一化：对单个样本的所有特征计算均值和方差，不依赖 batch 内其他样本。

$$\mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$$
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

- 可学习参数：$\gamma$ 和 $\beta$，共 $2 \times d$ 个
- 适合 NLP 场景：每个 token 独立归一化，不受 batch 大小和序列长度影响
- 原始 Transformer、BERT、GPT-2 均使用 LayerNorm

### RMSNorm

在 LayerNorm 基础上去掉均值偏移（re-centering），只保留缩放（re-scaling）：

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}, \quad y = \frac{x}{\text{RMS}(x) + \epsilon} \cdot \gamma$$

- 可学习参数：只有 $\gamma$，共 $d$ 个（比 LayerNorm 少一半）
- 动机：Zhang & Sennrich (2019) 发现 LayerNorm 的效果主要来自缩放操作，均值偏移对收敛帮助甚微，去掉后计算更快且效果持平
- LLaMA、Qwen、DeepSeek 等主流模型的标准选择

> **为什么 $\gamma$ 和 $\beta$ 是基于不同样本同一特征的？**
>
> 可学习参数属于模型的全局参数，它是对该特征维度的一种全局校准，用来恢复本来的特征表达能力。

---

## Pre-Norm vs Post-Norm

归一化放在子层（attention/FFN）的前面还是后面，对训练行为影响很大：

```
Post-Norm:  x → SubLayer → Add(x, ·) → Norm → 输出
Pre-Norm:   x → Norm → SubLayer → Add(x, ·) → 输出
```

| 方式 | 特点 | 代表模型 |
|------|------|---------|
| Post-Norm | 残差分支经过 Norm 后才回主路，深层梯度衰减快，需要 warmup 和小学习率 | 原始 Transformer、BERT |
| Pre-Norm | 残差连接直接加回主路（不经过 Norm），梯度流更顺畅，训练稳定性好 | GPT-3、LLaMA、所有现代大模型 |

Pre-Norm 训练更稳定的原因：残差主路上没有归一化操作阻挡梯度，信号可以从最后一层直接回传到第一层。代价是有研究指出 Pre-Norm 在同等深度下的表征能力可能略弱于 Post-Norm（因为各层对残差流的贡献被"稀释"），但在大模型的实践中这个差异可以忽略。

---

## 参考资料

- [Layer Normalization (Ba et al., 2016)](https://arxiv.org/abs/1607.06450)
- [Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467)
- [On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)](https://arxiv.org/abs/2002.04745)
