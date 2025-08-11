---
layout: distill
title: "在TPU上服务LLaMA 3-70B"
# permalink: /main/
description: "让我们深入了解如何在TPU v5e上服务LLaMA 3-70B模型。在理论峰值性能下，不同模型的服务成本是多少？它们的KV缓存有多大？我们应该使用什么批次大小？在推理期间，参数和激活是如何分片的？让我们通过一些粗略估算来计算生产环境中的延迟和吞吐量。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 8

previous_section_url: "inference"
previous_section_name: "第7部分：推理"

next_section_url: profiling
next_section_name: "第9部分：性能分析"

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
  - name: "LLaMA服务的情况如何？"
  - subsections:
    - name: "考虑吞吐量"
    - name: "预填充怎么办？"
  - name: "可视化延迟吞吐量权衡"
  - name: "实践问题"

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

*本节将探讨服务LLaMA-3需要什么以及如何高效地完成。与之前的"应用"章节一样，尝试在查阅答案之前用纸笔自己计算出答案！*

## LLaMA服务的情况如何？

让我们回顾一下LLaMA 3-70B的参数（参考[第6节](../applied-training)）：

| **超参数**              | **值** |
| --------------------------- | :-------: |
| $$n_\text{layers}$$ (L)     |    80     |
| $$d_\text{model}$$ (D)      |   8,192   |
| $$d_{ff}$$ (F)              |  28,672   |
| $$n_\text{heads}$$ (N)      |    64     |
| $$n_\text{kv heads}$$ (K)   |     8     |
| $$d_\text{qkv}$$ (H)        |    128    |
| $$n_\text{embeddings}$$ (V) |  128,256  |

让我们从一个简单的问题开始：**我们应该在什么硬件上服务？** 答案基本上是，无论哪个在FLOPs/美元上最便宜的。<d-footnote>这并不总是正确的，有时更多的HBM或ICI带宽比FLOPs更关键，但这是一个好的启发式规则。</d-footnote> 因此，我们通常希望在TPU v5e上服务，这是我们当前的专用推理芯片（成本来自[Google Cloud定价](https://cloud.google.com/tpu/pricing)，截至2025年2月）：

| **TPU类型** | **bfloat16 FLOPs/s** | **Google Cloud USD / 小时** | **FLOPs / $** |
| ------------ | :------------------: | :-------------------------: | :-----------: |
| H100         |        9.9e14        |            $10.8            |    3.3e17     |
| v5p          |       4.59e14        |            $4.2             |    3.9e17    |
| v5e          |       1.97e14        |            $1.2             |  **5.8e17**  |

每个TPU v5e有16GB的HBM，这要求我们相当激进地分片我们的模型。让我们开始思考一些对我们可能重要的基本量：

**问题：** LLaMA 3-70B的KV缓存每个token有多大？*你可以假设我们用int8存储它们。这决定了我们在给定拓扑上的批次大小可以有多大。*

{% details 想好了再点这里！ %}

LLaMA 3-70B有8个KV头，所以每个token的大小是`2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`。

**注意这有多大！** 如果我们有32k token的序列长度（这很常见），这使用`162e3 * 32,768 = 5.3GB / 序列`。对于BS=240，这是1.3TB！由于TPU v5e只有每个16GB，我们需要大约`(70e9 + 1.3e12) / 16e9 = 86`个TPU v5e芯片才能装下这么多内存。还要注意与70GB模型参数相比，这是多么大。

{% enddetails %}

**问题：** 假设我们想要以批次大小32和8192序列长度服务L3 70B，所有东西（参数和KV）都用int8。这将使用多少总内存？我们可以服务的最小切片是什么？

{% details 答案 %}

由于我们的KV是int8的`160e3`字节，我们的总KV内存是`160e3 * 8192 * 32 = 41.9e9`字节。我们的参数是`70e9`字节，因为每个参数1字节。因此，我们的总内存使用量是`41.9e9 + 70e9 = 112GB`。

我们可以使用的最小切片将有`112e9 / 16e9 = 7`个TPU，或者（舍入到偶数大小），TPU v5e `4x2`。这将是紧凑的配合，我们可能无法完全适应考虑其他开销，所以我们可能需要至少一个`4x4`（或降低批次大小）。

{% enddetails %}

**问题：** 在这种批次大小和量化在TPU v5e `4x2`上，我们期望每个解码步骤的延迟大约是多少？吞吐量（tokens / sec / chip）呢？`4x4`怎么样？*假设我们在bfloat16中执行FLOPs，一切都完全分片。*

{% details 答案 %}

我们可以调用上一节的公式

$$\begin{align*}
\tiny \text{理论步骤时间（通用）} = \underbrace{\frac{\text{批次大小} \times \text{KV缓存大小}}{\tiny \text{总内存带宽}}}_{\text{注意力（总是带宽受限）}} + \underbrace{\max\left(\frac{2 \times \text{批次大小} \times \text{参数数量}}{\text{总FLOPs/s}}, \frac{\text{参数大小}}{\text{总内存带宽}}\right)}_{\tiny \text{MLP（可以是计算受限）}}
\end{align*}$$

这里我们的临界批次大小大约是120，因为我们的参数是int8，但我们的FLOPs是bfloat16。我们也可以手动计算RHS最大值，但这基本上是我们已经做过几次的计算。**所以我们在matmul和FLOPs的内存受限区域都很好。**

严格看内存带宽，我们的步骤时间基本上是`(KV大小 + 参数大小) / (8 * HBM带宽) = 112e9 / (8 * 8.1e11) = 17ms`。**所以理论上我们的步骤时间大约是17ms。** 我们的吞吐量将是`32 / .017 = 1882 tokens / sec`，或`1882 / 8 = 235 tokens / sec / chip`。

这里有一个警告，就是检查我们的matmul是否可能是ICI受限的。我们可以为此专用2个轴，所以理论上当$Y > 2 * F / 2550 = 2 * 28672 / 2550 = 22$时，我们是ICI受限的，所以我们很好！

如果我们在`4x4`上运行，我们在ICI方面仍然很好，所以我们的延迟会降到`17 / 2 = 8.5ms`，但我们每芯片的吞吐量会保持相同。

{% enddetails %}

### 考虑吞吐量

让我们花一点时间纯粹考虑吞吐量。当我们优化吞吐量时，我们希望是计算受限的，这意味着我们接近利用所有TPU MXU容量。通常这意味着我们希望批次大小尽可能大，所以我们做尽可能多的工作。

**问题：** 在TPU v5e上，使用bfloat16权重和激活，我们的批次大小需要多大才能在matmul中是计算受限的？如果我们做int8权重但在bfloat16中执行FLOPs怎么样？int8权重配int8 FLOPs呢？

{% details 答案 %}

如第7节所讨论的，对于任何$B \ll D, F$的bfloat16 matmul，我们有

$$\begin{equation*}
T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM带宽}} = 240
\end{equation*}$$

当我们的权重是int8时，我们在分母中失去了因子2，所以我们有$2BDF / DF = 2B > 240$，或等价地$B > 120$，是之前临界批次大小的一半。这对我们真的很有帮助！当我们做int8权重和int8 FLOPs时，我们必须使用TPU FLOPs/s的int8值，从bfloat16的1.97e14变为3.94e14，几乎翻倍。这意味着我们回到了大约$B > 240$的起点。

int8权重和bfloat16 FLOPs的情况很常见，因为无损量化参数通常比做低精度算术更容易。

{% enddetails %}

**问题：** 我们可以在最小的TPU v5e拓扑上服务LLaMA 3-70B，使用bfloat16、int8和int4（KV和参数都）与8k上下文的是什么？*你可以认为这个KV缓存非常小。*

{% details 答案 %}

这很简单！如果我们接受一个很小的批次大小，那么唯一的限制就是将参数内存装入HBM，即只是`ceil(参数数量 * sizeof(dtype) / 每个TPU的HBM`，或`ceil(70e9 * sizeof(dtype) / 16e9)`四舍五入到最近的合理拓扑（2的某个倍数）：

| dtype | 参数大小 | KV大小 / token（字节） | 最小TPU v5e | 实际最小切片 | KV缓存的剩余HBM | 8k时的KV缓存数量 |
| :---: | :--------: | :---------------------: | :----------: | :--------------: | :-------------------------: | :----------------: |
| bf16  |   140GB    |          324kB          |     8.75     |  4x4 = 16个芯片  |             116             |         43         |
| int8  |    70GB    |          162kB          |     4.38     |  4x2 = 8个芯片   |             68              |         52         |
| int4  |    45GB    |          81kB           |     2.81     |  2x2 = 4个芯片   |             19              |         67         |

这很酷！它告诉我们，如果我们想要，我们可以在TPU v5e 2x2上装下LLaMA 70B。但是你会注意到KV缓存的数量很小。这是我们的批次大小！这意味着我们将得到糟糕的FLOPs利用率。我们很乐意使用更大的拓扑以便将批次大小推到240。

{% enddetails %}

**问题：** 假设我们使用适合这些拓扑的最大批次大小，我们可以期望每个生成步骤的延迟是多少？

{% details 答案 %}

这也很简单，因为我们选择批次大小来填满所有HBM！这只是一个关于将一个完整TPU v5e的字节数加载到MXU需要多长时间的问题。这只是`v5e HBM / v5e HBM内存带宽 = 16GB / 8.2e11 = 19ms`，所以这是**19ms / 步骤**。假设我们的生成平均长度为512个token，那大约是每次解码9秒。注意，我们可以通过较小的批次大小获得边际更好的延迟，例如，如果我们只看int4中的模型参数，我们的最小延迟大约是10ms / 步骤，因为HBM不再满了。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：我们总是可以通过询问从HBM将模型的所有参数加载到MXU需要多长时间来下界解码延迟。当我们的KV缓存很小时，你可以将每层视为只是逐块加载权重然后丢弃它们。除非我们使用大批次大小或大量设备间通信，这通常是一个合理的界限（在1.5倍以内）。当我们的批次大小更大时，我们需要对KV缓存加载进行建模，因为它主导参数。</p>

同样，在FLOPs受限区域（例如训练或大批次推理）中，我们可以使用$$\text{总FLOPs} / (N \cdot C) = 2 \cdot \text{参数数量} \cdot B / (N \cdot C)$$下界，它假设没有通信。

**问题：** 对于每一种，这给我们什么吞吐量每芯片（就查询/芯片而言）？*你可以假设我们的中位解码长度是512个token。*

{% details 答案 %}

这是一个重要问题，因为它与成本/token完全相关。

在我们关于中位解码长度的假设下，我们的吞吐量只是$$B / (\text{每步延迟} \cdot \text{中位步数} \cdot N) \approxeq 43 / (0.019 * 512 * N)$$。这给我们大约$$(4.42 / N)$$ QPS，所以插入$$N$$我们得到：

|  dtype   | QPS / 芯片 |
| :------: | :--------: |
| bfloat16 |    0.27    |
|   int8   |    0.66    |
|   int4   |    1.72    |

注意这相当乐观，因为它完全忽略了前向传递的工作内存（分配给激活和注意力的内存）。这在Flash Attention下并非完全荒谬，但也不现实。真实数字可能大约是这个的1/2。对于绝对最大吞吐量，我们可能希望将芯片数量增加一倍以上，并显著增加批次大小。

{% enddetails %}

**问题：** 如果我们将上述每个示例的拓扑翻倍，我们的峰值吞吐量会如何变化？

{% details 答案 %}

如果我们在bfloat16中使用4x8切片，我们将有186GB剩余用于KV缓存，这将让我们将批次大小提升到161。然后由于我们的步骤时间保持相同，我们的吞吐量将是`16.54 / 芯片数量`，或

|       dtype       | QPS / 芯片 |
| :---------------: | :--------: |
| bfloat16 (在4x8上) |    0.51    |
|   int8 (在4x4上)   |    1.03    |
|   int4 (在2x4上)   |    2.06    |

进一步增加会带来更大的收益！重要的收获是**在所有情况下，最小拓扑不是最高性能拓扑**，如果我们受到KV缓存大小的限制。

{% enddetails %}

**问题：** 现在让我们深入分片问题。假设我们想要在TPU v5e 4x8上以bfloat16服务。我们在TPU v5e 4x8上生成期间对模型使用什么分片？我们能避免通信受限吗？

{% details 答案 %}

如上一节所讨论的，我们在生成期间真的只有一个分片选项：模型并行性。我们在变为通信受限之前可以做多少？如我们在上一节中讨论的，我们的模型大致在

$$Y > \frac{F \cdot n_\text{轴数}}{2550}$$

时变为通信受限

对于LLaMA 3-70B，我们有`F = 28,672`，所以如果我们做2轴模型分片，这给我们大约$$Y = 28672 \cdot 2 / 2550 = 22$$，所以一般来说我们可以扩展到大约16个芯片而不会通信受限，这让我们使用`4x4`但不是`4x8`。通常，由于我们不能完美重叠计算，即使这个估计也过于乐观。

**要点：我们实际上不能在纯模型并行性的4x8上服务。** 我们在这里能做的最好的是4x2或*也许*4x4。

然而，如我们所讨论的，当我们的批次大小很小时，我们经常可以做更多模型并行性而不会显著伤害吞吐量，因为我们的模型是内存带宽受限而不是FLOPs受限。我们之前说过这个值大约是$Y=F / (8\cdot B)$，所以如果我们做批次大小64，理论上我们可以达到`Y = 28,672 / (8 * 64) = 56`的模型并行性，然后才会变为ICI受限。为了健全性检查，我们可以看单个matmul的$T_\text{ici comms}$，$T_\text{hbm comms}$和$T_\text{math}$。我们清楚地有：

$$\begin{align*}T_\text{ici comms} = \frac{2BD}{W_\text{ici}} && T_\text{hbm comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{align*}$$

对于`4x8`，这将给我们$T_\text{ici comms}$ = `(2 * 64 * 8192) / 9e10 = 11us`，$T_\text{hbm comms}$ = `(2 * 8192 * 28,672) / (32 * 8.1e11) = 18us`，$T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`，所以理论上我们仍然是HBM带宽受限，这很好！*注意从`4x4`扩展到`4x8`可能从吞吐量角度来看没有帮助，但它会减少我们的延迟！

如果我们看int8和int4配置，我们*可以*用纯模型并行性做到这些。所以我们已经达到了量化实际上给我们除了更快FLOPs之外的有意义优势的点：它让我们在变为通信受限之前使用更大的批次大小。**所以这个故事的结局是我们不能在4x8上实现峰值吞吐量，但对于int8和int4配置，我们可以做纯模型并行性*。

{% enddetails %}

<p markdown=1 class="takeaway">**提示**：有用模型并行性的最大量取决于$$d_{ff}$$和您分片模型的轴数。最大值通常在8到32之间，取决于模型大小。您可以超出此限制以改善延迟，但会有一些吞吐量成本。</p>

### 预填充怎么办？

我们在这里大多忽略了预填充，因为它简单得多。让我们将几个概念结合起来，思考端到端的图片。

**问题：** 假设我们在预填充期间实现40%的FLOPs利用率。在16个TPU v5e芯片上，长度8192的预填充将需要多长时间？

{% details 答案 %}

在8k token时，我们完全是计算受限的，所以我们只需要考虑FLOPs。我们知道我们的模型有`70e9`个参数，所以每次前向传递使用`2 * 70e9 * B`FLOPs。假设40% MFU（FLOPs利用率），这给我们大约`2 * 70e9 * 8192 / (16 * 1.97e14 * 0.4) = 0.91s`的运行时间。与我们之前看过的数字相比，这实际上相当多！

{% enddetails %}

**问题：** 假设我们有8192个token的中位预填充长度和4096个token的中位解码长度。假设我们有32的生成批次大小。平均每步有多少序列完成解码？平均每步从我们的KV缓存中逐出多少token？

{% details 答案 %}

这有点直接。由于我们有4096个token的中位解码长度，一个序列将大约每1/4096个token完成一次。给定批次大小32，这意味着我们每步有`32 / 4096`个序列被逐出。由于我们的KV缓存长度大约是`8192 + 4096`，这是每步逐出`32 * (8192 + 4096) / 4096 = 96`个token。一般公式是$B * (P + G) / G$，其中$P$和$G$是预填充和生成长度。

{% enddetails %}

**问题：** 假设我们做分解服务，中位预填充长度为8192，中位解码长度为512。假设在bfloat16中计算的上述预填充和生成延迟。您需要什么比例的预填充：生成服务器来保持两者都充分饱和。

{% details 答案 %}

这是一个有趣的问题。设$P$为预填充服务器数量，$G$为生成服务器数量。所以一般来说，这是一个流水线问题，我们以`P / 预填充延迟`的速率输入序列，以`B * G / (生成延迟 * 中位解码长度)`的速率消费它们。我们计算出每个预填充步骤`910ms`，每个解码步骤在批次大小43（我们称之为32）时`19ms`。因此我们需要`P / 0.91 = 32 * G / (0.019 * 512)`或`P = 3G`，即我们需要大约3倍的预填充服务器比生成服务器！

{% enddetails %}

## 可视化延迟吞吐量权衡

继续使用LLaMA 70B，让我们实际看看生成期间不同批次大小的延迟和吞吐量。如我们在前一节中对PaLM模型所显示的，这给我们一个吞吐量/延迟的帕累托前沿。让我们假设16路张量并行性，因为这是我们在MLP块中保持计算受限时可以使用的合理界限。我们在这里将使用TPU v5e 4x4拓扑。**滑块控制序列长度，所以你可以看到更大KV缓存的效果。**

<div class="l-page">
  <iframe src="{{ 'assets/plotly/pareto.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

* **看到成本和延迟之间的权衡是多么戏剧性。** 以每token延迟翻倍的成本，我们可以实现每token成本大约100倍的减少。此外，我们的延迟可以在低批次大小的5.5ms到非常大批次的20ms之间的任何地方。
* 注意在2k上下文时，吞吐量在大约1 token / ms / chip时有效平稳，当它达到BS 120峰值线时（这里是120，因为我们做int8权重但bf16 FLOPs）。然而，随着序列长度增加，我们不能再在内存中适应这个批次大小，所以我们永远不会到达完全饱和的点。
* 注意在相同吞吐量下，大批次大小的延迟要高得多，因为KV加载变得主导（而不是参数加载）。

我们可以通过将成本和延迟的来源分解为参数加载时间、KV加载时间和FLOPs时间来更好地理解这一点。红色区域是我们期望在MLP块中计算受限的区域。

<div class="l-page">
  <iframe src="{{ 'assets/plotly/latency_breakdown_log.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

这讲述了一个故事。你可以看到，最初，参数加载代表了延迟的绝大部分，直到批次大小变得足够大，FLOPs和KV加载变得更重要。值得注意的是，在所有大于2048的序列长度上，我们在KV缓存加载上花费的时间比FLOPs多！**所以虽然我们可以通过增加批次大小来改善硬件利用率，但在长上下文长度下，KV加载总是主导总步骤时间。**

<p markdown=1 class="takeaway">**要点：** 对于LLaMA 3-70B，我们在几乎所有这些配置中都强烈受KV缓存内存带宽限制（和HBM限制），突出了减少KV缓存大小对生成吞吐量的重要性。还要注意这里延迟/吞吐量权衡仍然是多么戏剧性。</p>

{% details 这个代码很简单。 %}

这是计算这些峰值线的代码：

```py
import numpy as np

num_chips = 16  # 我们固定16为我们做的总模型并行性数量
param_size = 70e9  # int8意味着每个参数1字节
sequence_length = 8192  # 可以变化这个

hbm_bandwidth = 8.20E+11  # v5e
flops = 1.97E+14  # v5e

param_size = bytes_per_param * param_count

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80
    
def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(max_num_chips: int = 16):
  # for num_chips in topo_sizes:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  num_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(num_chips <= max_num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(num_chips, sequence_length, param_size)  # 获取可以适应的最大批次大小
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_count * batch_sizes / (num_chips * flops)  # 在2阶意义上大致正确

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # 对于生成总是带宽受限

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

注意我们如何非常明确地将延迟分解为两个来源：KV加载和参数加载，以及延迟如何受FLOPs或通信的限制，无论哪个更大。

{% enddetails %}

## 实践问题

这里有一些实践问题。其中一些重复了上面已经解决的内容，但可能在教学上有用。

**问题1：** LLaMA 3-405B每个前向传递每token使用多少FLOPs？假设我们是FLOPs受限的，在TPU v5e的N个芯片上单次前向传递的下界是多少？如果我们是通信受限呢？*忽略模型不适合单个芯片的事实。*

**问题2：** 假设我们想要使用int8权重和int8 KV缓存以BS240服务LLaMA 3-8B。有多少字节被(a)模型参数(b) KV缓存和(c)峰值工作激活（大致）使用？我们可以运行的最小拓扑是什么？

**问题3：** 你将如何在TPU v5e上服务LLaMA 3-405B？假设int8权重和bfloat16 FLOPs。假设我们有15ms / token的严格限制，我们可以实现的最高吞吐量配置是什么？理论最小步骤时间是多少？

<h3 markdown=1 class="next-section">第8部分到此结束！对于第9部分，深入XLA和TPU分析，点击[这里](../profiling)。</h3>