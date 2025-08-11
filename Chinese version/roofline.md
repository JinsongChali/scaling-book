---
layout: distill
title: "关于Roofline分析的一切"
# permalink: /main/
description: "当我们在硬件上运行算法时，我们会受到三个方面的限制：计算机执行数学运算的速度（操作数/秒）、移动数据的可用带宽（字节/秒）以及存储数据的总内存（字节）。这些"roofline"约束让我们能够确定给定计算的上限和下限时间。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 1

previous_section_url: ".."
previous_section_name: "第0部分：引言"

next_section_url: ../tpus
next_section_name: "第2部分：TPU"

bibliography: main.bib

giscus_comments: true

authors:
  - name: Jacob Austin
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind
  - name: Sholto Douglas
    url: "https://x.com/_sholtodouglas"
  - name: Roy Frostig
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Anselm Levskaya
    url: "https://anselmlevskaya.com/"
  - name: Charlie Chen
    url: "https://x.com/charliexychen"
  - name: Sharad Vikram
    url: "https://sharadvikram.com/"
  - name: Federico Lebron
    url: "https://fedelebron.com/"
  - name: Peter Choy
    url: "https://x.com/pchoy95"
  - name: Vinay Ramasesh
    url: "https://x.com/vinayramasesh"
  - name: Albert Webson
    url: "https://representation.ai/"
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  
  - name: 时间都花在哪里了？
  - subsections:
    - name: "可视化roofline"
    - name: "矩阵乘法"
    - name: "网络通信roofline"
  - name: 几个练习题

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## 时间都花在哪里了？

让我们从一个极其简单的问题开始：*为什么一个算法需要50毫秒而不是50秒或5毫秒*？模型内部究竟发生了什么会耗费大量时间，我们应该预期它需要多长时间？

**计算：** 深度学习模型本质上是一堆矩阵乘法，每个都由浮点乘法和加法"操作"（FLOP）组成。我们的加速器速度决定了计算这些操作需要多长时间：

$$\begin{equation}
T_\text{math} = \frac{\text{计算 FLOP}}{\text{加速器 FLOP/s}}
\end{equation}$$

例如，NVIDIA H100 可以执行大约 9.89e14 bfloat16<d-footnote>bf16是<a href="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format">bfloat16</a>的简称，这是机器学习中经常使用的16位浮点格式。</d-footnote> FLOP/s，而TPU v6e可以执行 9.1e14 FLOP/s。<d-footnote>H100和B200通常只能达到声称峰值FLOP的80-85%左右，而TPU在正常使用中可以接近95%。</d-footnote> 这意味着在H100上执行1e12 FLOP大约需要`1e12 / 9.89e14 = 1.01ms`，在TPU v6e上需要`1e12 / 9.1e14 = 1.1ms`。<d-footnote>请注意，这些芯片的价格不同，这个比较没有按成本进行标准化。</d-footnote>

**芯片内通信：** *在加速器内部*，张量需要在片上内存（HBM）和计算核心之间传输。你会看到这个连接的带宽被称为"HBM带宽"<d-footnote>NVIDIA也称其为"内存带宽"。</d-footnote> 在H100上，[这大约是3.35TB/s](https://www.nvidia.com/en-us/data-center/h100/)，在TPU v6e上[这大约是1.6TB/s](https://cloud.google.com/tpu/docs/v6e)。

**芯片间通信：** 当我们将模型*分布在多个加速器上*时，张量经常需要在它们之间传输。在我们的硬件上通常有几个选择（ICI、DCN和PCIe），每个都有不同的带宽。

无论是芯片内还是芯片间的通信，我们都以字节/秒为单位进行测量，并用以下方式估算总通信时间：

$$\begin{equation}
T_\text{comms} = \frac{\text{通信字节数}}{\text{网络/内存带宽字节/s}}
\end{equation}$$

通常（但不总是如此），单芯片内的计算可以与芯片内和芯片间的通信重叠。这意味着**我们可以使用计算和通信时间的最大值来下限估计训练和推理时间**。我们也可以**用它们的和来上限估计**。在实践中，我们针对最大值进行优化，因为代数更简单，我们通常可以通过重叠通信和计算接近这个界限。如果我们以最大值为优化目标，那么下限和上限相差最多2倍，因为$T_\text{math} + T_\text{comms} \leq 2 * \max(T_\text{math}, T_\text{comms})$。然后我们通过建模"重叠区域"和开销来提高超出此范围的准确性，这可以通过分析您的特定模型和目标系统来了解。

$$\begin{equation}
T_\text{lower}=\max(T_\text{math}, T_\text{comms})
\end{equation}$$

$$\begin{equation}
T_\text{upper} = T_\text{math} + T_\text{comms}
\end{equation}$$

如果我们假设可以完美地重叠通信和计算，当$T_\text{math} > T_\text{comms}$时，我们看到硬件的完全利用。我们称这种情况为"计算受限"。当$T_\text{comms} > T_\text{math}$时，我们往往是"通信受限"的，我们的加速器FLOP/s中至少有一部分被浪费在等待数据传递上。判断一个操作是计算受限还是通信受限的一种方法是查看其"*算术强度*"或"*操作强度*"。

**定义：** 算法的算术强度由其执行的总FLOP数与需要通信的字节数（无论是芯片内还是芯片间）的比值给出。

$$\begin{equation}
\text{算术强度} = \frac{\text{计算 FLOP}}{\text{通信字节数}}
\end{equation}$$

算术强度衡量给定操作的"每字节FLOP数"。初步而言，当我们的算术强度很高时，$T_\text{math}$相对于$T_\text{comms}$很大，我们通常使用大部分可用FLOP。相反情况下，我们在通信上花费更多时间并浪费FLOP。发生这种交叉的点是我们硬件的"峰值算术强度"，即峰值加速器FLOP/s与加速器带宽的比值。

$$\begin{align*}
T_\text{math} > T_\text{comms} \Leftrightarrow \frac{\text{计算 FLOP}} {\text{加速器 FLOP/s}} > \frac{\text{通信字节数}}{\text{带宽字节/s}} & \\[0.5em]
\Leftrightarrow \frac{\text{计算 FLOP}}{\text{通信字节数}} > \frac{\text{加速器 FLOP/s}}{\text{带宽字节/s}} & \\[0.5em]
\Leftrightarrow \text{强度}(\text{计算}) > \text{强度}(\text{加速器}) & \\
\end{align*}$$

量$\text{强度}(\text{加速器})$是我们的加速器实现其峰值FLOP/s的算术强度。**对于TPU v5e MXU，这大约是240 FLOP/字节**<d-footnote>MXU是TPU上的矩阵乘法单元。我们在这里特指这个是因为TPU还有其他加速器如VPU，负责元素级操作，具有不同的峰值FLOP/s。</d-footnote>，因为TPU可以执行`1.97e14` FLOP/s并从HBM加载`8.2e11`字节/s。这意味着如果一个算法的算术强度低于240<d-footnote>这仅在算法从HBM加载其权重并在MXU中运行时才为真。正如我们将在下一节中讨论的，我们有时可以将参数存储在具有更高带宽的VMEM中。许多算法也在具有不同性能特征的VPU中运行。</d-footnote> FLOP/字节，它将受到字节加载的限制，因此我们无法充分利用我们的硬件。让我们看一个这样的例子：

**<span style="color:#7ab5ff">例子（点积）</span>：** 要计算bfloat16精度下两个向量的点积，`x • y: bf16[N], bf16[N] → bf16[1]`，我们需要从内存加载$x$和$y$，每个都有$2 * N = 2N$字节，执行$N$次乘法和$N-1$次加法，并将$2$字节写回HBM
$$\begin{equation}
\text{强度}(\text{点积}) = \frac{\text{总 FLOP}}{\text{总字节数}} = \frac{N + N - 1}{2N + 2N + 2} = \frac{2N - 1}{4N + 2} \rightarrow \frac{1}{2}
\end{equation}$$

当$N\rightarrow\infty$时。所以点积的算术强度是$\frac{1}{2}$，或者换句话说，点积每加载一字节执行0.5次浮点操作。这意味着我们的算术强度低于我们硬件的算术强度，我们将受到通信限制。<d-footnote>上面的240这个数字在这里不是正确的比较，因为正如你将在下一节看到的，点积是在VPU而不是MXU上执行的。TPU v5p VPU大约可以执行7e12 FLOP/秒，所以其临界强度大约是3，这意味着我们在这里仍然有些受通信限制。无论如何，我们强度低且恒定的事实意味着在大多数硬件上很难成为计算受限的。</d-footnote>

### 可视化roofline

我们可以使用**roofline图**来可视化内存和计算之间的权衡，该图将算法在我们硬件上可达到的峰值FLOP/s（吞吐量）（y轴）与该算法的算术强度（x轴）进行绘图。这里是一个示例对数-对数图：

{% include figure.liquid path="assets/img/roofline-improved.png" class="img-fluid" caption="<b>图：</b> 一个示例roofline图，显示了两个具有不同算术强度的算法（算法1和算法2）以及它们在不同带宽（BW1和BW2）下相应的理论峰值吞吐量。在红色区域，算法在两个带宽下都受带宽限制，浪费了硬件峰值FLOP/s的一部分。黄色区域仅在较低带宽（BW1）下受带宽限制。绿色区域在所有带宽下都受计算限制。在这里，我们使用加速器的峰值FLOP/s，增加带宽或提高强度不会产生任何好处。" %}

上图中，随着强度增加（从左到右移动），我们最初看到算法性能（以FLOP/s为单位）的线性增加，直到达到硬件的临界算术强度，在TPU v5e的情况下是240。任何强度较低的算法都将受到带宽（BW）限制并受到峰值内存带宽的限制（红色显示）。右侧的任何算法都将充分利用我们的FLOP（绿色显示）。这里，算法1受通信限制，只使用了总硬件FLOP/s的一部分。算法2受计算限制。我们通常可以通过增加其算术强度或增加可用的内存带宽（从BW1移动到BW2）来改善算法的性能。

### 矩阵乘法

让我们看看我们即将最喜欢的算法：矩阵乘法（又名matmul）。我们写$X * Y \rightarrow Z$，其中$X$的形状是$\text{bf16}[B, D]$，$Y$的形状是$\text{bf16}[D, F]$，$Z$的形状是$\text{bf16}[B, F]$。要做matmul我们需要加载$2DF + 2BD$字节，执行$2BDF$ FLOP，并写回$2BF$字节。<d-footnote>技术上我们执行$BF \times (2D - 1)$ FLOP，但这已经足够接近了。这来自$BDF$次乘法和$BF * (D-1)$次加法。第4节有更多细节。</d-footnote> <d-footnote>虽然matmul的输出技术上是float32，我们通常在复制回HBM之前向下转换为bfloat16。</d-footnote> 因此：

$$\begin{equation}
\text{强度}(\text{matmul}) = \frac{2BDF}{2BD + 2DF + 2BF} = \frac{BDF}{BD + DF + BF}
\end{equation}$$

如果我们假设"批次大小"$B$相对于$D$和$F$较小，我们可以得到一个很好的简化。然后我们得到

$$\begin{equation}
\frac{BDF}{BD + DF + BF} \approxeq \frac{BDF}{DF} = B
\end{equation}$$

$$\begin{equation}
\text{强度}(\text{matmul}) > \text{强度}(\text{TPU}) \implies B > \frac{1.97e14}{8.20e11} = 240
\end{equation}$$

这对于Transformer matmul来说是一个合理的假设，因为对于我们的大多数模型，我们的本地**token**批次大小$B < 1024$，但$D$和$F > 8000$。因此，当我们的本地批次大小大于240个token时，我们变成计算受限的，这是一个非常简单的规则！

<p markdown=1 class="takeaway">**要点：** 对于bfloat16 matmul在大多数TPU上成为计算受限，我们需要本地token批次大小大于240。<d-footnote>请注意，这_不是_通常意义上的批次大小，通常意义上的批次大小指的是序列中的批次大小。实际上，大多数roofline纯粹依赖于token的数量，无论它们属于相同还是不同的序列。例如，如果你在128个GPU上有512个序列的4096个token的批次大小，你有总批次大小`512 * 4096 = 2M`个token，以及16k个token的本地批次大小。</d-footnote></p>

这带来了一些我们将在下面的问题中探索的显著注意事项，特别是关于量化的（例如，如果我们量化我们的激活但仍然做全精度FLOP），但这是一个很好的记住规则。对于GPU，这个数字稍高（接近300），但同样的结论通常成立。当我们[将大matmul分解为较小的matmul](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#your-first-matrix-multiplication-kernel)时，tile大小也很重要。<d-footnote>当我们做大矩阵乘法时，我们需要将其分解为适合VMEM/SMEM/TMEM（更高带宽的片上内存）的较小tile。这导致我们多次加载块，所以我们不再只加载$O(N^2)$字节。考虑一个$(m, k) \cdot (k, n)$ matmul，tile大小为$bm$、$bk$、$bn$。设$tm = m / bm$等。那么总FLOP是$2 \cdot tm \cdot tn \cdot tk \cdot bm \cdot bk \cdot bn$，总字节数是$2 \cdot tm \cdot tn \cdot (tk \cdot (bm \cdot bk + bk \cdot bn) + 2 \cdot bm \cdot bn)$。忽略最后一项，我们有强度$bm \cdot bn / (bm + bn)$，这与上面的类似。</d-footnote> 我们将在[下一节](../tpus)中讨论较低级别的GPU和TPU细节。

### 网络通信roofline

到目前为止我们讨论的所有roofline都是内存带宽roofline，_都在单个芯片内_。这不应该被当作规则。实际上，我们在本书中关心的大多数roofline涉及芯片间通信：通常是涉及跨多个TPU分片的矩阵的矩阵乘法。

举一个有些人为的例子，假设我们想要乘以两个大矩阵$X\sim \text{bfloat16[B, D]}$和$Y \sim \text{bfloat16[D, F]}$，它们在2个TPU/GPU之间平均分割（沿$D$维度）。要做这个乘法（正如我们将在[第3节](../sharding)中看到的），我们可以在每个TPU上乘以每个矩阵的一半（在TPU 0上`A = X[:, :D // 2] @ Y[:D // 2, :]`，在TPU 1上`B = X[:, D // 2:] @ Y[D // 2:, :]`），然后将结果"部分和"复制到另一个TPU并将它们加在一起。假设我们可以在每个方向复制`4.5e10`字节，并在每个芯片上执行`1.97e14` FLOP/s。$T_\text{math}$和$T_\text{comms}$是什么？

$T_\text{math}$显然是之前的一半，因为每个TPU在做一半的工作，即<d-footnote>我们忽略了将两个部分和加在一起所需的FLOP（另外DF次加法），但这基本上是可忽略的。</d-footnote>

$$T_\text{math} = \frac{2BDF}{2 \cdot \text{加速器 FLOP/s}} = \frac{BDF}{1.97e14}$$

现在$T_\text{comms}$呢？这现在指的是芯片间的通信时间！这只是发送的总字节数除以网络带宽，即

$$T_\text{comms} = \frac{2BF}{\text{网络带宽}} = \frac{2BF}{4.5e10}$$

因此，当$$\text{强度}(\text{matmul (2-芯片)}) > \text{强度}(\text{关于芯片间网络的TPU})$$或等价地当$\frac{BDF}{2BF} = \frac{D}{2} > \frac{1.97e14}{4.5e10} = 4377$或$D > 8755$时，我们变成计算受限（现在关于芯片间网络）。注意，与之前不同，临界阈值现在取决于$D$而不是$B$！试着想想为什么会这样。这只是一个这样的例子，但我们强调这种roofline对于知道何时可以跨多个TPU并行化操作是关键的。

## 几个练习题

**问题1 [int8 matmul]：** 假设我们想要以int8精度（每个参数1字节）而不是bfloat16执行matmul $X[B, D] \cdot_D Y[D, F] \rightarrow Z[B, F]$。<d-footnote>这里和整本书中我们将使用符号$A \cdot_D B$来表示乘法正在对D维度执行收缩。这是einsum符号的滥用。</d-footnote>

1. 需要从内存加载多少字节？需要将多少字节写回内存？
2. 执行了多少总操作？
3. 算术强度是多少？
4. $T_\text{math}$和$T_\text{comms}$的roofline估计是什么？整个操作运行时间的合理上限和下限是什么？

假设我们的HBM带宽是`8.1e11`字节/s，我们的int8峰值操作/s是`3.94e14`（大约是bfloat16的2倍）。

{% details 点击这里查看答案。 %}

1. 因为我们以int8存储参数，每个参数1字节，所以我们从HBM加载$$BD + DF$$字节，写回$$BF$$字节。
2. 这与bfloat16中相同，但理论上int8操作/s应该更快。所以这仍然是$2BDF$ FLOP。
3. 算术强度是$$2BDF / (BD + DF + BF)$$。如果我们对$$B \ll D$$和$$B \ll F$$做同样的假设，我们得到算术强度$$2B$$，意味着我们的规则变成$B > \text{HBM int8 算术强度} / 2$。使用给定的数字，这个int8强度是`3.94e14 / 8.1e11 = 486`，所以规则是$B > 486 / 2 = 243$。注意这基本上没有改变！
4. $$T_\text{math} = 2BDF / 3.94e14$$和$$T_\text{comms} = (BD + DF + BF) / 8.1e11$$，所以合理的下限是$$\max(T_\text{math}, T_\text{comms})$$，上限是$$T_\text{math} + T_\text{comms}$$。

{% enddetails %}

**问题2 [int8 + bf16 matmul]：** 在实践中，我们经常对权重和激活进行不同的量化，所以我们可能以非常低的精度存储权重但保持激活（和计算）在更高精度。假设我们想要将权重量化为int8但保持激活（和计算）为bfloat16。在什么批次大小下我们变成计算受限？假设`1.97e14` bfloat16 FLOP/s。

*提示：这特指`bfloat16[B, D] * int8[D, F] -> bfloat16[B, F]`，其中$B$是"批次大小"。*

{% details 点击这里查看答案。 %}

再次假设B较小，我们有2BDF bfloat16 FLOP但只有DF个权重（而不是bfloat16中的2DF）。这意味着我们在$$2B > 240$$或$$B > 120$$时变成计算受限。这要低得多，意味着如果我们能做int8权重量化（这相当容易做到）但仍然做bfloat16 FLOP，我们在效率上获得有意义的胜利（虽然int8操作会更好）。

{% enddetails %}

**问题3：** 采用问题2的设置，为$F = D = 4096$和$F = D = 1024$制作峰值FLOP与$B$的roofline图。*使用准确的加载字节数，不是近似值。*

{% details 点击这里查看答案。 %}

这里是相关图表：

{% include figure.liquid path="assets/img/roofline-plot-q3.png" class="img-fluid img-small" %}

注意两个模型最终都达到峰值硬件FLOP/s，但较大的D/F更早达到。D=F=1024几乎使临界批次大小翻倍。生成此图的代码在这里：

```py
import matplotlib.pyplot as plt
import numpy as np

bs = np.arange(1, 512)

def roofline(B, D, F):
  total_flops = 2*B*D*F
  flops_time = total_flops / 1.97e14
  comms_time = (2*B*D + D*F + 2*B*F) / 8.2e11
  total_time = np.maximum(flops_time, comms_time)
  return total_flops / total_time

roofline_big = roofline(bs, 4096, 4096)
roofline_small = roofline(bs, 1024, 1024)

plt.figure(figsize=(8, 4))
plt.plot(bs, roofline_big, label='F=D=4096')
plt.plot(bs, roofline_small, label='F=D=1024')
plt.legend()
plt.xlabel('批次大小')
plt.ylabel('TPU v5e上的峰值bfloat16 FLOP/s')
plt.grid()
```

{% enddetails %}

**问题4：** 如果我们想要执行$\text{int8[B, D]} *_D \text{int8[B, D, F]} \rightarrow \text{int8[B, F]}$，其中我们想象每个批次元素有不同的矩阵，这个操作的算术强度是什么？

{% details 点击这里查看答案。 %}

让我们先看看总FLOP和通信。

1. 总FLOP：FLOP基本相同，因为我们在做相同数量的$$BD \times DF$$ matmul（这在第4节中讨论更多）。所以这只是$$2BDF$$。
2. 总通信：我们这里有更多通信：$$BD + BDF + BF$$。
3. 因此，我们的算术强度现在实际上是$$2BDF / (BD + BDF + BF)$$。由于$$BDF$$主导分母，这大约是$$2$$。所以不是依赖于批次大小，这本质上是常数。这很糟糕，因为这意味着无论如何我们基本上总是会受到通信限制。

{% enddetails %}

**问题5 [GPU的内存Roofline]：** 使用[NVIDIA为H100提供的规格表](https://www.nvidia.com/en-us/data-center/h100/)，计算矩阵乘法变成计算受限的批次大小。*注意Tensor Core FLOP数字是真实值的两倍，因为它们只能通过结构化稀疏性实现。*

{% details 点击这里查看答案。 %}

从规格表中，我们看到报告的bfloat16 FLOP值是`1.979e15` FLOP/s，带有星号注释"with sparsity"。没有稀疏性的真实值是这个的一半，意味着接近`1e15` FLOP/s。内存带宽是3.35TB/s，或`3.35e12`字节/秒。因此$B_\text{crit}$是`1e15 / 3.35e12 = 298`，与TPU相当类似。

{% enddetails %}

<h3 markdown=1 class="next-section">第1部分就到这里！对于第2部分，看看真实的TPU如何处理FLOP和通信，[点击这里](../tpus)。</h3>