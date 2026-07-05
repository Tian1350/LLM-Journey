# Diffusion Models

## 核心直觉

扩散模型是一类生成模型。它的思路可以先用一句话理解：

> 训练时学会“从带噪数据里恢复干净数据”，生成时从随机噪声开始，反复去噪，最后得到样本。

这里有两个过程：

- **前向过程**：把真实数据逐步加噪，直到它接近标准高斯噪声。
- **反向过程**：训练神经网络一步步预测如何去掉噪声，从噪声还原出数据。

在图像生成里，真实数据通常是图片；在视频、音频、3D 或分子生成里，数据形式会变，但“逐步破坏数据，再学习反向恢复”的思想基本不变。

需要注意的是，扩散模型不是“简单把噪声擦掉”的图像修复算法。它学的是数据分布的反向生成过程：模型不是记住某张图片，而是学习在每个噪声强度下，怎样把样本往真实数据分布的方向推回去。

---

## 基本公式

DDPM 里常见的前向加噪写法是：

$$
q(x_t \mid x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}x_0,\ (1-\bar{\alpha}_t)I\right)
$$

也就是可以直接从干净样本 $x_0$ 得到任意时刻的带噪样本：

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

训练时最常见的目标是让网络预测加入的噪声：

$$
\mathcal{L} = \mathbb{E}_{x_0,t,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t,t)\|^2\right]
$$

这就是为什么很多教程会说“扩散模型在预测噪声”。但这只是参数化方式之一，不是唯一方式。模型也可以预测：

- `epsilon`：预测噪声，DDPM 里最常见；
- `x0`：预测干净样本；
- `v`：预测 velocity，Stable Diffusion 2.x、Imagen Video 等工作中常见；
- score：预测 $\nabla_x \log p_t(x)$，对应 score-based generative modeling 的视角。

这些形式在理论上关系很近，工程上会影响训练稳定性、采样质量和不同噪声调度下的表现。

---

## DDPM：把扩散模型做成可训练系统

DDPM（Denoising Diffusion Probabilistic Models）把扩散生成写成一个清晰的概率模型：前向过程固定为逐步加高斯噪声，反向过程由神经网络学习。

一个常见误解是：

> DDPM 的核心不是 U-Net，而是“固定前向加噪 + 学习反向去噪”的概率建模和训练目标。

U-Net 只是图像任务里非常合适的网络骨架。它擅长保留多尺度空间信息，所以在早期图像扩散模型中几乎成了默认选择。但换成 Transformer、MLP 或其他结构，仍然可以是扩散模型。

DDPM 的优点很明显：

- 训练目标简单，通常就是 MSE；
- 生成质量高，模式覆盖比 GAN 更稳；
- 不需要对抗训练，训练过程更容易调。

代价也很明显：

- 采样慢，早期 DDPM 常需要几百到上千步去噪；
- 每一步都要跑一次去噪网络，生成延迟高；
- 高分辨率图像上直接在像素空间扩散，计算和显存压力很大。

---

## DDIM：少走几步的采样器

DDIM（Denoising Diffusion Implicit Models）主要解决采样慢的问题。它没有推翻 DDPM 的训练方式，而是提出了一种更高效的采样路径，可以用更少步数生成图像。

直观地说，DDPM 的反向采样是随机的马尔可夫过程，每一步都带随机性；DDIM 可以构造非马尔可夫、甚至确定性的采样过程。在实际使用中，这意味着：

- 同一个训练好的 DDPM 模型，可以换 DDIM sampler 加速采样；
- 采样步数可以从上千步降到几十步；
- 确定性采样方便做图像编辑、插值和复现。

DDIM 之后，采样器成为扩散模型工程里的重要组件。今天常见的 Euler、Heun、DPM-Solver、UniPC 等，本质上都是在回答同一个问题：怎样用更少的神经网络调用，走出更好的反向轨迹。

---

## Conditioning：文本、类别和控制信号怎么进来

扩散模型本身只负责“从噪声生成样本”。如果希望它按类别、文本、草图、深度图或姿态生成，就需要把条件信息喂给去噪网络。

常见做法包括：

- **Class conditioning**：把类别标签嵌入后加到时间嵌入或网络层里。
- **Text conditioning**：用文本编码器得到 token 表示，再通过 cross-attention 让图像特征读取文本信息。
- **Image/control conditioning**：把边缘图、深度图、pose、mask 等作为额外条件输入。

Stable Diffusion 里的文本到图像生成，关键就在于 cross-attention：U-Net 在去噪过程中，不只看当前 noisy latent 和时间步，也会读取文本编码器输出的语义信息。

---

## Classifier-Free Guidance

Classifier-Free Guidance（CFG）是现代文本到图像扩散模型里非常关键的技巧。

训练时，模型会随机丢掉一部分条件，让同一个网络同时学会：

- 有条件去噪：$\epsilon_\theta(x_t, t, c)$
- 无条件去噪：$\epsilon_\theta(x_t, t, \varnothing)$

采样时，把两者做一个外推：

$$
\epsilon_{\text{cfg}} =
\epsilon_\theta(x_t,t,\varnothing)
+ s \cdot \left(\epsilon_\theta(x_t,t,c)-\epsilon_\theta(x_t,t,\varnothing)\right)
$$

其中 `s` 是 guidance scale。它越大，模型越听 prompt，但也更容易出现过饱和、细节崩坏、构图僵硬等问题。

所以 CFG 不是“免费增强质量”的开关。它是在 prompt 对齐和自然度之间做权衡。

---

## Latent Diffusion 与 Stable Diffusion

直接在像素空间做扩散很贵。Latent Diffusion Models（LDM）的关键想法是：先用 Autoencoder 把图像压缩到 latent space，在 latent 上做扩散，最后再解码回像素图像。

流程大致是：

```text
图像 x -> VAE Encoder -> latent z
latent z 上加噪/去噪
去噪后的 z -> VAE Decoder -> 图像 x
```

Stable Diffusion 就是 LDM 路线最有代表性的系统之一。它通常包含三块：

| 组件 | 作用 |
|---|---|
| VAE / Autoencoder | 在像素空间和 latent space 之间转换 |
| Text Encoder | 把 prompt 编成语义表示 |
| Denoising U-Net | 在 latent space 中逐步去噪 |

相比像素空间扩散，latent diffusion 的优势是：

- latent 分辨率更低，计算量小很多；
- 训练和采样成本下降；
- 更容易在消费级 GPU 上部署；
- 配合文本条件和 CFG 后，生成质量与可控性都比较好。

它的代价是，最终图像质量会受 VAE 压缩能力影响。非常细的文字、纹理和几何结构，可能在压缩和解码过程中损失或变形。

---

## DiT：把 U-Net 换成 Transformer

DiT（Diffusion Transformer）把去噪网络从 U-Net 换成 Transformer。更准确地说，它不是“把卷积层机械替换成 Transformer”，而是把图像或 latent 切成 patch，把这些 patch 当作 token，用 Transformer block 做去噪建模。

DiT 的基本结构可以理解为：

```text
noisy latent/image -> patchify -> Transformer blocks -> predict noise / velocity
```

时间步和条件信息会通过特殊的 conditioning 机制注入 Transformer，比如 adaptive layer norm。这样模型可以根据当前噪声强度和文本/类别条件调整去噪行为。

DiT 的意义在于：扩散模型的生成质量不只依赖“扩散过程”，也强依赖去噪网络的 scaling 能力。Transformer 在大规模数据和大模型训练上已经被验证过，所以 DiT 打开了扩散模型向更大模型扩展的路线。

不过 DiT 也不是简单地“全面优于 U-Net”：

- 小模型、小数据或低分辨率场景下，U-Net 的卷积归纳偏置仍然很有用；
- Transformer 通常更吃数据和算力；
- 高效训练和推理需要配合 patch size、attention 优化、latent space 等设计。

---

## Score、SDE 和 Flow Matching 的关系

扩散模型还有另一条常见表述：score-based generative modeling。它把不同噪声强度下的数据分布看成一族分布，训练模型估计 score：

$$
\nabla_x \log p_t(x)
$$

生成时通过 SDE 或 ODE 从噪声分布走回数据分布。这个视角和 DDPM 不是两套完全无关的东西，而是对同一类生成过程的不同数学描述。

近几年很多新模型又转向 Flow Matching、Rectified Flow、Consistency Models 等路线。它们和传统 DDPM 不完全一样，但目标相似：学习一条从简单分布到数据分布的生成路径，并尽量减少采样步数。

可以粗略理解为：

- DDPM/DDIM：经典扩散路线，逐步去噪；
- Score/SDE：连续时间视角，强调 score 和随机微分方程；
- Flow Matching/Rectified Flow：学习更直接的传输路径，常用于减少采样步数；
- Consistency Models：希望一步或少步生成，同时保持质量。

这些方法的边界在工程实现上会互相借鉴，所以现代“扩散模型”这个词经常被宽泛地用来指一整个去噪/流式生成家族。

---

## 常见误解

**扩散模型不是只能生成图像。**
图像是最成功的应用之一，但扩散思想也可以用于音频、视频、3D、分子结构、机器人轨迹等连续数据。

**扩散模型不是必须用 U-Net。**
DDPM 论文和 Stable Diffusion 大量使用 U-Net，是因为它适合图像；DiT 证明 Transformer 也可以作为强去噪网络。

**Stable Diffusion 不只是“DDPM + 压缩”。**
更准确地说，它是 latent diffusion 框架下的文本条件生成系统，包含 VAE、文本编码器、latent-space denoising U-Net、cross-attention 和 CFG 等关键组件。

**采样步数少不一定质量差。**
好的 sampler、distillation 或 flow-based 训练可以在少步下保持质量。但步数越少，模型和采样器设计越关键。

**Guidance scale 不是越大越好。**
过高的 CFG 会让图像更贴 prompt，但也可能牺牲自然度、细节和多样性。

---

## 演进脉络

| 阶段 | 代表方法 | 解决的问题 |
|---|---|---|
| DDPM | Denoising Diffusion Probabilistic Models | 建立稳定可训练的扩散生成框架 |
| DDIM / fast samplers | DDIM、DPM-Solver 等 | 减少采样步数，提高生成速度 |
| Guided diffusion | Classifier guidance、CFG | 提高条件控制能力 |
| Latent Diffusion | LDM、Stable Diffusion | 降低高分辨率图像生成成本 |
| Diffusion Transformer | DiT、后续大规模 T2I 模型 | 用 Transformer 提升 scaling 能力 |
| Flow / Consistency | Rectified Flow、Consistency Models | 进一步减少采样步数 |

如果只记一条主线，可以这样看：

```text
DDPM 解决“怎么稳定训练”
DDIM/采样器解决“怎么少走几步”
LDM/Stable Diffusion 解决“怎么便宜地做高分辨率”
DiT 解决“去噪网络怎么继续 scale”
Flow/Consistency 继续压缩“生成需要多少步”
```

---

## 参考资料

- Jonathan Ho, Ajay Jain, Pieter Abbeel, [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), 2020.
- Jiaming Song, Chenlin Meng, Stefano Ermon, [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502), 2020.
- Yang Song et al., [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456), 2021.
- Jonathan Ho, Tim Salimans, [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598), 2022.
- Robin Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), 2022.
- William Peebles, Saining Xie, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748), 2022.
- Yaron Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), 2022.
