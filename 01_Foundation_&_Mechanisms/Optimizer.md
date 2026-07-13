# 优化器（Optimizer）

优化器负责在训练中根据梯度更新模型参数，目标是最小化损失函数。核心矛盾始终是：**如何选择更新的方向和步长**，让模型又快又稳地收敛。

优化器的演进主线有两条，最后在 Adam 系列汇合：

```
SGD ──加动量──→ Momentum ──────────────────┐
                                            ├──→ Adam ──解耦权重衰减──→ AdamW
AdaGrad ──限制窗口──→ RMSprop ──────────────┘
（自适应学习率主线）
```

---

## SGD（随机梯度下降）

每次迭代只用一个或一小批（mini-batch）样本估计梯度并更新参数：

$$\theta_{t+1} = \theta_t - \eta \cdot g_t, \quad g_t = \nabla_\theta L(\theta_t)$$

其中 $\eta$ 是学习率，$g_t$ 是当前梯度。相比用全量数据的批梯度下降，SGD 计算成本低，且梯度噪声反而有助于跳出尖锐的局部极小。

**局限**：所有参数共用一个固定学习率；在损失面狭长弯曲（ill-conditioned）的区域会来回震荡、收敛慢。

### Momentum（动量）

引入动量项累积历史梯度方向，像"惯性"一样加速收敛、抑制震荡：

$$v_t = \mu v_{t-1} + g_t, \quad \theta_{t+1} = \theta_t - \eta v_t$$

$\mu$（通常 0.9）是动量系数。在梯度方向一致时累积加速，在方向频繁翻转时相互抵消，从而更平稳。

### NAG（Nesterov 加速梯度）

Momentum 的改进：先按动量"预走一步"，在预测位置计算梯度，相当于有了"前瞻性"，能更早修正方向。

$$v_t = \mu v_{t-1} + \nabla_\theta L(\theta_t - \eta \mu v_{t-1})$$

---

## 自适应学习率系列

核心思想：**为每个参数单独调整学习率**，而不是全局共用一个。

### AdaGrad

累积历史梯度的平方和，用它来缩放学习率——梯度大（频繁更新）的参数学习率变小，梯度小（不常更新）的参数学习率保持较大：

$$G_t = G_{t-1} + g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot g_t$$

**优点**：适合稀疏特征（不常出现的特征能获得大更新）。
**缺点**：$G_t$ 单调递增，累积到后期学习率趋近于 0，训练提前"停滞"。

### RMSprop

针对 AdaGrad 学习率无限衰减的问题，把"累加所有历史"改为"指数移动平均"，只关注近期梯度：

$$E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho) g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} \cdot g_t$$

$\rho$（通常 0.9）是衰减率。学习率不再单调递减，适合非平稳目标（如 RNN 训练）。

### AdaDelta

和 RMSprop 几乎同时提出、思路相近，同样用梯度平方的滑动窗口。特别之处是进一步用参数更新量的滑动平均来替代全局学习率，理论上**连初始学习率都不需要设**。

---

## Adam 系列

### Adam（Adaptive Moment Estimation）

**结合了 Momentum（一阶矩）和 RMSprop（二阶矩）**，是深度学习最常用的优化器。

一阶矩（动量）和二阶矩（梯度平方的滑动平均）：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

由于 $m_0, v_0$ 初始化为 0，早期估计会偏向 0，需做偏差修正：

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

最终更新：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

默认超参数 $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$。优点是收敛快、对超参数相对鲁棒；缺点是在某些任务上泛化不如调好的 SGD+Momentum，理论上存在不收敛的构造反例（AMSGrad 对此做了修正）。

### AdamW

Adam 里的 L2 正则（weight decay）实际上被自适应学习率缩放了，导致正则强度对不同参数不一致。**AdamW 把权重衰减从梯度中解耦出来**，直接作用在参数上：

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

其中 $\lambda \theta_t$ 是独立的权重衰减项。这个看似微小的改动显著改善了泛化，**现在几乎所有大模型（GPT、LLaMA 等）都用 AdamW**。

### Nadam

把 NAG 的"前瞻动量"融入 Adam——在动量项上应用 Nesterov 的预测思想，理论上收敛更快，但实际收益有限，用得不多。

### RAdam（Rectified Adam）

Adam 训练初期由于二阶矩估计样本太少、方差过大，容易走偏（这也是为什么实践中常配合 warmup）。RAdam 引入一个整流项动态修正早期的方差，前期自动退化为带动量的 SGD，稳定后再切换到 Adam，相当于"内置了自适应 warmup"。

---

## 其他优化器（简述）

- **ASGD（平均 SGD）**：用参数的历史平均值作为最终结果，在凸问题上有更好的理论收敛性
- **Rprop**：只用梯度的**符号**（不用大小）决定更新方向，每个参数独立自适应步长。只适合全量梯度（full-batch），不适合 mini-batch，因此深度学习中少用
- **Lion**（2023）：Google 用符号搜索发现的优化器，只用动量的符号更新，显存占用比 Adam 少（不需存二阶矩），在大模型上有竞争力
- **Muon**（2024）：面向大模型的新优化器，对权重矩阵的动量做正交化处理，在部分预训练场景下比 AdamW 更快

---

## 如何选择

| 场景 | 推荐 |
|------|------|
| 大模型预训练/微调 | AdamW（事实标准） |
| CV 任务、追求极致泛化 | SGD + Momentum（调好后常优于 Adam） |
| RNN、非平稳目标 | RMSprop / Adam |
| 稀疏特征 | AdaGrad / Adam |
| 显存受限的大模型 | Lion（省一份二阶矩显存） |

在实践中，面对不确定的场景，**可以首选 AdamW**，它对超参数最不敏感、性能最均衡。

---

## 参考资料

- [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization - AdamW (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101)
- [On the Variance of the Adaptive Learning Rate and Beyond - RAdam (Liu et al., 2019)](https://arxiv.org/abs/1908.03265)
- [Adaptive Subgradient Methods - AdaGrad (Duchi et al., 2011)](https://www.jmlr.org/papers/v12/duchi11a.html)
- [Symbolic Discovery of Optimization Algorithms - Lion (Chen et al., 2023)](https://arxiv.org/abs/2302.06675)
