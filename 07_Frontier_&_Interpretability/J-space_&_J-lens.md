# J-space 与 J-lens

> 来源：Anthropic《Verbalizable Representations Form a Global Workspace in Language Models》（transformer-circuits.pub，2026 年 7 月）。
> 本文基于该研究的公开介绍与多篇二手解读整理，部分技术细节以论文原文为准。

## 一句话概括

Anthropic 在 Claude 内部发现了一个类似认知科学中"全局工作空间"的结构——**J-space**，并提出了一种基于雅可比矩阵（Jacobian）的解释工具 **J-lens**，用来读出模型在"说出口"之前，其内部到底在"想"什么。

---

## 背景：全局工作空间理论

"全局工作空间"（Global Workspace）是认知神经科学的经典理论（Baars、Dehaene 等提出）：大脑中存在一个中央"工作台"，不同的专用模块（视觉、听觉、记忆等）各自处理信息，但只有被"广播"到这个工作台上的信息，才会进入意识、被整合和报告出来。

Anthropic 的发现是：**大语言模型里似乎也存在类似的机制**——不是所有内部表征都平等，有一部分表征处在一个可被模型"读取并表达"的公共空间里，另一些则停留在局部、无法被语言化。

---

## 什么是 J-space

J-space（J 代表 Jacobian）指模型残差流（residual stream）中一个特定的**子空间**，落在这个子空间里的表征是**可言语化的（verbalizable）**——即模型能够读取它、并将其体现在输出的 token 上。

关键点：
- 残差流是高维的，但并非每个方向都能影响输出。J-space 是那些**真正能传导到最终输出**的方向所张成的空间
- 它扮演的角色类似全局工作空间里的"公共黑板"：写在上面的信息才能被"读出来"
- 处在 J-space 之外的内部状态，可能参与了中间计算，但不会直接反映到模型说出的话里——这就是所谓的"沉默的思考"

这带来一个重要视角：**模型内部的表征 ≠ 模型说出来的内容**。有些信息模型"想到了"但没有（或无法）表达，J-space 划出了这两者的边界。

---

## 什么是 J-lens

J-lens（Jacobian Lens，雅可比透镜）是配套的解释工具，用来把内部激活投影到"可读出"的方向上，看模型在某一层、某一步实际上"表达"了什么。

### 与 Logit Lens 的对比

早期的 **Logit Lens** 是直接用模型的 unembedding 矩阵，把中间层的残差流硬投影到词表上，看"如果现在就输出会是什么"。它的问题是忽略了后续网络层的非线性变换——中间层的表征还要经过很多层处理才变成输出，直接套 unembedding 并不准确。

**J-lens 的改进**：不用固定的 unembedding，而是计算**输出对残差流的雅可比矩阵**（Jacobian，即输出 logits 关于该处激活的一阶导数），用这个雅可比来投影。

直观理解：
- 雅可比刻画的是"此处激活的微小变化，会如何传导并改变最终输出"
- 它自然地包含了后续所有层的线性化影响，比 Logit Lens 更忠实地反映"这个内部状态实际能表达出什么"
- 因此 J-lens 读出的方向，正好张成了 J-space

> 一句话：Logit Lens 问"现在直接输出是什么"，J-lens 问"这个内部状态最终会怎样影响输出"。后者才对应真正"可言语化"的信息。

---

## 主要发现与意义

综合公开介绍，这项研究的几个要点：

1. **可言语化表征构成一个全局工作空间**
   - 模型确实把"能报告的信息"集中在一个相对低维、共享的空间里，行为上呼应了认知科学的全局工作空间理论

2. **存在"未被言语化"的内部状态**
   - 模型的一些内部表征停留在 J-space 之外，参与计算却不直接显现在输出中。媒体常把这描述为"Claude 的隐藏思考"

3. **对可控性/可解释性的价值**
   - J-space 给了一个更精确的"读心"接口：想知道模型真实在处理什么，看 J-space 里的投影比看它嘴上说的更可靠
   - 对齐与安全上有潜在用途，例如检测模型是否"意识到自己在被测试"、是否有未表达的意图

---

## 需要澄清的误解

- **这不是"AI 有意识"的证明**。全局工作空间理论在认知科学里常和"意识"关联，但这项研究说的是**表征结构和信息可达性**，不是主观体验。大量二手报道用"Claude 是否有意识"做标题属于过度演绎
- **"隐藏思考"不等于"藏心机"**。J-space 之外的状态更多是计算的中间产物，把它拟人化为"心里想着却不说"的意图是不准确的
- **J-lens 是解释工具，不是读心术**。它给的是对输出影响的线性近似，有适用范围和误差，不能当作对模型"想法"的完整读取

---

## 参考资料

- [Anthropic - A global workspace in language models](https://www.anthropic.com/research/global-workspace)
- [Transformer Circuits - Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/)
- [LessWrong - A global workspace in language models](https://www.lesswrong.com/posts/3PaLrzxagpbnNtPLT/a-global-workspace-in-language-models)
- 全局工作空间理论背景：Baars (1988), Dehaene et al. (Global Neuronal Workspace)
