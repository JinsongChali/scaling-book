---
layout: distill
title: "关于Transformer推理的一切"
# permalink: /main/
description: "在Transformer上执行推理与训练有很大不同。部分原因是推理增加了一个新的考虑因素：延迟。在本节中，我们将从模型采样单个新token开始，一直到作为推理引擎的一部分在多个加速器片段上高效扩展大型Transformer。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 7

previous_section_url: "../applied-training"
previous_section_name: "第6部分：训练LLaMA"

next_section_url: "../applied-inference"
next_section_name: "第8部分：服务LLaMA"

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
  - name: "Transformer推理基础"
  - subsections:
    - name: "我们实际想要优化什么？"
    - name: "线性操作：什么是我们的瓶颈？"
    - name: "注意力呢？"
    - name: "LLM延迟和吞吐量的理论估计"
    - name: "内存呢？"
    - name: "为LLaMA 2-13B建模吞吐量和延迟"
  - name: "提高生成吞吐量和延迟的技巧"
  - name: "在多个加速器上分布推理"
  - subsections:
    - name: "预填充"
    - name: "生成"
    - name: "分片KV缓存"
  - name: "设计有效的推理引擎"
  - subsections:
    - name: "连续批处理"
    - name: "前缀缓存"
    - name: "让我们看一个实现：JetStream"
  - name: "练习题"
  - name: "附录"
  
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

## Transformer推理基础

所以你已经训练了一个Transformer，你想用它来生成一些新序列。_归根结底，基准分数的提高和损失曲线的下降只是某些有趣事情在实际应用中是否会发生的代理！_<d-footnote>从历史上看，你可以在Transformer上做大量研究而无需接触推理——LLM损失、多选基准可以在没有适当的KV缓存或生成循环实现的情况下高效运行。这意味着，特别是在研究代码库中，推理代码路径中通常有很多低垂的果实。</d-footnote>

采样在概念上很简单。我们输入一个序列，我们最喜欢的Transformer将输出$$\log p(\text{下一个token}_i \vert \text{之前的tokens})$$，即所有可能的下一个token的对数概率。我们可以从这个分布中采样并获得一个新token。追加这个token并重复这个过程，我们就获得了一个token序列，它是提示的延续。

{% include figure.liquid path="assets/img/naive-inference.png" class="img-fluid" caption="<b>图：</b>从Transformer进行朴素采样。蓝色的logits给我们一个下一个token的分布，我们可以从中采样。注意每一步都重新处理整个前缀，导致算法的$\Theta(n^2)$运行时间。" %}

我们刚刚描述了Transformer采样的朴素实现，虽然它有效，**但我们在实践中从不这样做**，因为我们每次生成token时都重新处理整个序列。这个算法在FFW上是$$O(n^2)$$，在注意力机制上是$$O(n^3)$$来生成$$n$$个token！

**我们如何避免这个问题？** 我们可以保存每次前向传递的一些中间激活，而不是每次都进行完整的前向传递，这让我们避免重新处理之前的token。具体来说，由于给定token在点积注意力期间只关注之前的token，我们可以简单地将每个token的键和值投影写入一个称为**KV缓存**的新数据结构。一旦我们为过去的token保存了这些键/值投影，未来的token可以简单地计算它们的$$q_i \cdot k_j$$乘积，而无需对早期token执行任何新的FLOPs。太棒了！

考虑到这一点，推理有两个关键部分：

* <b style="color: red;">预填充</b>：给定一个长提示，我们同时处理提示中的所有token，并将生成的激活（具体来说，是键值投影）保存在**"KV缓存"**中。我们还保存最后一个token的logits。
* <b style="color: blue;">生成</b>：给定KV缓存和之前的logits，我们从logits中增量采样一个token，将该token反馈到Transformer中，并为下一步生成一组新的logits。我们还将该新token的KV激活追加到KV缓存中。我们重复这个过程，直到我们遇到特殊的`<EOS>` token或达到某个最大长度限制。

这是使用KV缓存进行采样的图表：

{% include figure.liquid path="assets/img/cached-inference.png" class="img-fluid" caption="<b>图：</b>使用KV缓存进行高效Transformer采样的图表。<b style=\"color: red;\">预填充</b>处理我们的提示并将所有每token键值激活保存在缓存中。<b style=\"color: blue;\">生成</b>获取此缓存（和最后token的logits），采样一个新token，并通过模型传递该新token，关注KV缓存并将新token的键值投影保存回缓存。这是MLP块中的$O(n)$算法。" %}

通过使用KV缓存进行采样，我们已将生成$n$个token的时间复杂度降低到FFW上的$$O(n)$$和注意力上的$$O(n^2)$$，因为我们从不重新处理之前的token。然而，生成序列仍然需要许多前向传递——这就是当你查询Gemini或ChatGPT并且结果流式返回给你时发生的事情。每个token（通常）是对大型模型的单独（但部分缓存）的Transformer调用。

我们很快就会看到，<b style="color: red;">预填充</b>和<b style="color: blue;">生成</b>是非常不同的野兽——Transformer推理是伪装的两个任务！与训练相比，KV缓存也是一个新颖且重要的复杂性来源。

### 我们实际想要优化什么？

在我们进一步进行之前，值得强调推理的一个全新方面：延迟。虽然在训练期间我们只关心吞吐量（每秒处理的总token数），但在推理期间我们必须担心我们生成token的速度（包括**首个token时间（TTFT）**和**每token延迟**）。例如：

* **离线批量推理**用于评估和数据生成只关心推理的批量成本，对单个样本的延迟视而不见。
* **聊天界面/流式任务**需要以低成本大规模运行，同时具有低TTFT并生成token的速度足以超过人类阅读速度。
* **边缘推理**（例如你笔记本电脑上的`llama.cpp`）一次只需要服务一个用户，以尽可能低的延迟，可能有严格的硬件限制。

最大化硬件利用率仍然至关重要，有助于成本和TTFT，但与训练不同，它并不*必然*在所有上下文中为个人用户带来更好的体验。加速器、系统和模型架构级别的许多优化在延迟、吞吐量、上下文长度甚至模型质量之间进行权衡。

### Transformer的更细粒度视图

到目前为止，我们主要将Transformer视为前馈块的堆栈。虽然从FLOPs和内存的角度来看这通常是合理的，但它不足以正确建模推理。<d-footnote>你会在本节中注意到的一件事是，推理比训练要求更高。我们通常有更少的FLOPs，更少的批处理机会，以及对延迟的更高敏感性。KV缓存也极大地复杂化了推理。</d-footnote>正如我们在[第4部分](../transformers)中看到的，Transformer前向传递的主要组件是：

1. **一堆线性操作**，包括MLP（$W_{in}$，$W_{out}$）和注意力QKV投影和输出投影（$W_Q$，$W_K$，$W_V$和$W_O$）。这些都涉及从HBM读取参数和一批激活，进行一些FLOPs，并将结果写回HBM。
2. **点积注意力**。我们需要从HBM读取一批键值投影和一批查询激活，进行一些内积和一些softmax操作，并将注意力结果写回HBM。
3. **其他所有内容**，包括应用层归一化、激活函数、token采样、更新KV缓存和位置嵌入。这些确实需要一些FLOPs，但被上述内容所主导，或融合到其中。

在接下来的几节中，我们将在预填充和生成的上下文中查看这些中的每一个，并询问什么可能成为我们性能的瓶颈。在单个加速器内，我们是计算受限还是内存受限？我们想强调预填充与生成的答案会有多么不同。

### 线性操作：什么是我们的瓶颈？

我们所有的线性操作在概念上都是相同的，无论它们是在MLP块还是注意力中。它们的算术强度取决于批量大小。我们在[第1节](../roofline)中做了这个数学，但值得重复。让我们看一个$\text{bf16[B, D]}$批次乘以$\text{bf16[D, F]}$矩阵的单个矩阵乘法。这可能是大的MLP块或较小的注意力投影之一（$W_Q$，$W_K$，$W_V$，$W_O$）。要进行这个矩阵乘法，我们需要将这两个数组从HBM加载到MXU中，进行乘法，然后将结果写回HBM。如前所述，我们有：

$$T_\text{math} = \frac{\text{总FLOPs}}{\text{TPU FLOPs/s}} = \frac{2BDF}{\text{TPU FLOPs/s}}$$

$$T_\text{comms} = \frac{\text{总字节数}}{\text{HBM带宽}} = \frac{2BD + 2FD + 2BF}{\text{HBM带宽}}$$

TPU可以在进行计算时通过加载来重叠这些，所以要成为计算受限，我们需要$$T_\text{math} \geq T_\text{comms}$$，或：

$$\frac{2BDF}{2BD + 2DF + 2BF} \geq \frac{\text{TPU FLOPs/s}}{\text{HBM带宽}} = \frac{1.97E+14}{8.20E+11} = 240$$

其中RHS是我们硬件的算术强度。现在让我们假设$D$和$F$与$B$相比非常大（通常我们的批次最多500，而$D$和$F > 10k$），我们可以通过使用$\small{2BD + 2DF + 2BF \approxeq 2DF}$这个事实来简化分母，这给我们

$$\begin{align*}
\frac{2BDF}{2BD + 2DF + BF} \approxeq \frac{2BDF}{2DF} \geq \frac{\text{TPU FLOPs/s}}{\text{HBM带宽}} \\
= \frac{1.97E+14}{8.20E+11} \implies B \geq 240 = B_{\text{crit}}
\end{align*}$$

<p markdown=1 class="takeaway">**要点：**要在任何矩阵乘法上成为计算受限，我们的总token批量大小必须大于$B_\text{crit}$，这取决于硬件和量化。对于TPU v5e上的bf16激活，这是240个token。这适用于我们Transformer中的任何简单矩阵乘法（例如MLP块或注意力投影）。</p>

在训练期间，我们将在所有矩阵乘法期间具有高强度，因为我们在非常大的批次上重用相同的权重。**这种高算术强度延续到预填充，因为用户提示通常是数百个（如果不是数千个）token长。**正如我们之前看到的，TPUv5e的硬件算术强度是240，所以如果将长度超过240个token的序列输入到在bf16上运行的密集模型中，我们预计会受到计算限制，一切都很好。短于此的提示在技术上可以批处理在一起以实现更高的利用率，但这通常是不必要的。

<p markdown=1 class="takeaway">**要点：**在预填充期间，所有矩阵乘法基本上总是受计算限制的。因此，简单地最大化硬件利用率或MFU（模型FLOPs利用率）就足以最大化每芯片吞吐量（成本）和延迟（以TTFT的形式）。除非提示非常短，否则在每个提示级别进行批处理只会增加延迟，而预填充吞吐量的改进很小。</p>

然而，在生成期间，对于每个请求，我们只能一次进行一个token的前向传递，因为步骤之间存在顺序依赖性！因此，我们只能（轻松地）通过将多个请求批处理在一起，在批次维度上并行化来实现良好的利用率。我们稍后会更多地讨论这个，但实际上在不影响延迟的情况下将许多并发请求批处理在一起是困难的。出于这个原因，**使用生成来饱和硬件FLOPs要困难得多。**

<p markdown=1 class="takeaway">**要点：**我们的总token批量大小必须大于$$B_{\text{crit}}$$，以便生成在线性/前馈操作上受计算限制（对于TPU v5e上的bf16参数为240）。因为生成是串行发生的，逐个token，这需要我们将多个请求批处理在一起，这很难！</p>

*值得注意的是这有多大！*生成批量大小为240意味着240个并发请求同时生成，对于密集模型有240个单独的KV缓存。这意味着在实践中很难实现，除了在一些批量推理设置中。相比之下，在预填充期间推送超过240个token是相当常规的，尽管随着稀疏性增加需要一些注意。

**请注意，这个确切的数字将因量化和硬件的种类而不同。**加速器通常可以在较低精度下提供更多算术。例如，如果我们有int8参数但在bf16中进行计算，临界批量大小降至120。使用int8激活和int8参数，它又跳回到240，因为TPUv5e可以提供400 TOPs/s的int8 x int8。

### 注意力呢？

当我们查看点积注意力操作时，事情变得更加复杂，特别是因为我们必须考虑KV缓存。让我们只看一个具有纯多头注意力的注意力头。在单个Flash Attention融合中，我们<d-footnote>我们在这里相当简化了，忽略了应用softmax、掩码等中的非矩阵乘法FLOPs。它们应该与计算或HBM读取重叠，但在某些TPU代上可能不容易做到。这些细节不会改变主要信息，即KV缓存通常是内存受限的。</d-footnote>：

1. 从HBM读取形状为$\text{bf16[B, T, D]}$的$Q$激活。
2. 从HBM读取KV缓存，这是一对$\text{bf16[B, S, D]}$张量。
3. 在$$QK$$矩阵乘法中执行$2BSTD$ FLOPs。使用Flash Attention，我们不需要将$\text{bf16[B, S, T]}$注意力矩阵写回HBM。
4. 在注意力$$AV$$矩阵乘法中执行$2BSTD$。
5. 将生成的$\text{bf16[B, T, D]}$张量写回HBM。

把它们放在一起，我们得到：

$$\text{多头注意力算术强度} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T}$$

对于预填充，$S=T$因为我们正在进行自注意力，所以这简化为$T^2 / 2T = T / 2$。这很好，因为这意味着**预填充期间注意力的算术强度是$\Theta(T)$**。这意味着很容易成为注意力的计算受限。只要我们的序列长度相当大，我们就会很好！

但由于生成具有微不足道的序列维度，并且$B$和$D$维度抵消，我们可以进行近似：

$$S \gg T = 1 \implies \frac{ST}{S+T} \approx 1$$

这很糟糕，因为这意味着我们无法做任何事情来提高生成期间注意力的算术强度。我们在加载大量KV缓存的同时只进行很少的FLOPs。**所以我们在注意力期间基本上总是受内存带宽限制！**

<p markdown=1 class="takeaway">**要点：**在预填充期间，对于任何合理的序列长度（大约$\gt 480$个token），注意力通常是计算受限的，而在生成期间，我们的算术强度低且恒定，所以我们总是受内存带宽限制。</p>

*从概念上讲，这是为什么？*主要是，我们在模型的线性部分受计算限制，因为参数（内存带宽密集型组件）被许多批处理项重用。然而，每个批处理项都有自己的KV缓存，所以更大的批量大小意味着更多的KV缓存。我们几乎*总是*在这里受内存限制，除非架构被积极调整。

这也意味着一旦参数内存与KV缓存内存相当，你将从增加批量大小中获得递减的吞吐量回报。递减回报对你的伤害程度取决于单个序列的参数与KV缓存字节的比率，即大致比率$2DF / SHK$。由于$HK\approx D$，这大致取决于$F$与$S$（序列长度）的比率。这也取决于使KV缓存更小的架构修改（我们稍后会说更多）。

### LLM延迟和吞吐量的理论估计

从这个数学中，我们可以在优化时获得相当好的步骤时间界限。**（注意：如果读者从整个章节中只记住一件事，那就是以下内容）。**对于生成期间的小批量大小（这很常见），我们可以通过假设我们在注意力和MLP块中都受内存带宽限制来下界我们的每步延迟：

$$\begin{equation*}
\text{理论最小步骤时间} = \frac{\text{批量大小} \times \text{KV缓存大小} + \text{参数大小}}{\text{总内存带宽}}
\end{equation*}$$

类似地，对于吞吐量：

$$\begin{equation*}
\text{理论最大Tokens/s} = \frac{\text{批量大小} \times \text{总内存带宽}}{\text{批量大小} \times \text{KV缓存大小} + \text{参数大小}}
\end{equation*}$$

最终，随着我们的批量大小增长，FLOPs开始主导参数加载，所以在实践中我们有更一般的方程：

$$\begin{align}
\tiny \text{理论步骤时间（一般）} = \underbrace{\frac{\text{批量大小} \times \text{KV缓存大小}}{\tiny \text{总内存带宽}}}_{\text{注意力（总是带宽受限）}} + \underbrace{\max\left(\frac{2 \times \text{批量大小} \times \text{参数数量}}{\text{总FLOPs/s}}, \frac{\text{参数大小}}{\text{总内存带宽}}\right)}_{\tiny \text{MLP（可以是计算受限）}}
\end{align}$$

其中注意力组件（左）从不受计算限制，因此不需要FLOPs屋顶线。这些对于粗略计算相当有用，例如

<b markdown=1 style="color: #57cf57;">快速测验：</b>假设我们想在TPU v5e 4x4片上使用int8和bf16 FLOPs从30B参数密集模型中生成批量大小为4个token的步骤，具有8192上下文和100 kB / token KV缓存。这个操作的合理延迟下界是什么？如果我们想采样256个token的批次呢？

{% details 点击这里查看答案。 %}

**答案：**在int8中，我们的参数将使用30e9字节，根据给定的规格，我们的KV缓存将使用`100e3 * 8192 = 819MB`。我们有16个芯片，每个具有`8.1e11`字节/秒的带宽和`1.97e14` bf16 FLOPs/s。根据上述方程，由于我们的批量大小很小，我们预计我们的步骤时间至少为`(4 * 819e6 + 30e9) / (16 * 8.1e11) = 2.5 ms`。在256个token时，我们将很好地进入MLP块的计算受限区域，所以我们的步骤时间大约为`(256 * 819e6) / (16 * 8.1e11) + (2 * 256 * 30e9) / (16 * 1.97e14) = 21ms`。

{% enddetails %}

如你所见，这里在吞吐量和延迟之间有明确的权衡。小批次快但不能很好地利用硬件。大批次慢但高效。这是为一些较旧的PaLM模型计算的延迟-吞吐量帕累托前沿（来自[ESTI论文](https://arxiv.org/pdf/2211.05102)<d-cite key="esti"></d-cite>）：

{% include figure.liquid path="assets/img/latency-cost.png" class="img-fluid" caption="<b>图：</b>几个PaLM模型的成本（读：吞吐量）与延迟的帕累托前沿。注意芯片数（C）和批量大小（B）如何沿着帕累托前沿移动你，除了绿点（PaLM 540B的C:32 B:16），其中可用内存阻止了设置支持良好的批量大小并导致吞吐量受损。注意吞吐量通常在批量大小240后趋于平坦。int8权重提供了更好的延迟-吞吐量帕累托最优，但不是更好的最大吞吐量。" %}

我们不仅通过批量大小作为旋钮来权衡延迟和吞吐量，如果我们发现自己受HBM限制，我们也可能更喜欢更大的拓扑而不是更小的拓扑，这样我们可以容纳更大的批次。[下一节](../applied-inference)更详细地探讨了这一点。

<p markdown=1 class="takeaway">**要点：**如果你关心生成吞吐量，请使用尽可能大的每芯片批量大小。任何高于TPU算术强度（$B_\text{crit}$，通常为120或240）的每芯片批量大小都将最大化吞吐量。你可能需要增加拓扑来实现这一点。较小的批量大小将允许你以吞吐量为代价改善延迟。</p>

{% details 从硬件的角度来看，这有一些警告。点击这里查看一些细节。 %}

这都相当理论化。在实践中，由于几个原因，我们通常不会看到尖锐的屋顶线：

* 我们假设HBM读取将与FLOPs完美重叠是不现实的，因为我们的编译器（XLA）是有缺陷的。
* 对于分片模型，XLA也经常无法有效地将我们模型分片矩阵乘法的ICI通信与FLOPs本身重叠，所以我们经常在线性上开始在$$\text{BS}=32$$上受到延迟打击。
* 大于理论屋顶线的批量大小仍然会看到吞吐量的一些改进，因为重叠不完美，但这是一个很好的启发式。

{% enddetails %}

### 内存呢？

我们花了一些时间查看带宽和FLOPs，但没有查看内存。由于我们的新数据结构KV缓存，推理时的内存图景看起来很不同。对于本节，让我们选择一个真实的模型（LLaMA 2-13B）来演示事情看起来有多不同：


| 超参数                    | 值     |
| ------------------------ | ------ |
| n\_layers (L)            | 40     |
| d\_model (D)             | 5,120  |
| ffw\_multiplier (F // D) | 2.7    |
| n\_heads (N)             | 40     |
| n\_kv\_heads (K)         | 40     |
| d\_qkv (H)               | 128    |
| n\_embeddings (V)        | 32,000 |

推理期间什么在使用内存？嗯，显然，我们的参数。计算这些，我们有：

| 参数             | 公式                                                                                                                      | 大小（字节）                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| FFW参数          | d_model<sup>2</sup> x ffw\_multiplier x 3（用于gelu \+ 输出投影）x n\_layers                                              | 5,120 x 5,120 x 2.7 x 3 x 40 \= **8.5e9**                        |
| 词汇表参数       | 2（输入和输出嵌入）x n\_embeddings x d\_model                                                                            | 2 x 32,000 x 5,120 \= **0.3e9**                                  |
| 注意力参数       | \[2（*q和输出*）x d\_model x n\_heads x d\_qkv \+ 2（*用于k和v*）x d\_model x n\_kv\_heads x d\_qkv\] x n\_layers       | (2 x 5,120 x 40 x 128 \+ 2 x 5,120 x 40 x 128\) x 40 \= **4.2e9** |

将这些参数相加，我们得到8.5e9 + 4.2e9 + 0.3e9 = **13e9总参数**，正如预期的那样。正如我们在前面的章节中看到的，在训练期间，我们可能将参数存储在bfloat16中，优化器状态存储在float32中。这可能使用大约100GB的内存。与我们的梯度检查点相比，这相形见绌，它可以使用几TB。

**推理有什么不同？**在推理期间，我们存储一份参数副本，比如说在bfloat16中。这使用26GB——在实践中，我们通常可以通过量化做得更好。没有优化器状态或梯度需要跟踪。因为我们不进行检查点（为反向传递保留激活），所以我们的激活占用对于预填充<d-footnote>特别是由于Flash Attention，它避免了实现我们的注意力矩阵</d-footnote>和生成都可以忽略不计。如果我们预填充8k个token，单个激活只使用大约`8,192 x 5,120 x 2字节 = 80MB`的内存。更长的预填充可以分解为许多较小的前向传递，所以对于更长的上下文也不是问题。生成使用的token更少，所以激活可以忽略不计。

**主要区别是KV缓存**。这些是所有过去token的键和值投影，大小仅受最大允许序列长度的限制。$$T$$个token的总大小是

$$\text{KV缓存大小} = 2 \cdot \text{每浮点字节} \cdot H \cdot K \cdot L \cdot T$$

其中$$H$$是每个头的维度，$$K$$是KV头的数量，$$L$$是层数，2来自存储键和值。

**这可以很快变得很大**，即使批量大小和上下文长度适中。对于LLaMA-13B，bf16中单个8192序列的KV缓存是

$$8192\ (T) \times 40\ (K) \times 128\ (H) \times 40\ (L) \times 2\ (\text{字节}) \times 2 = 6.7 \text{GB}$$

**仅4个这样的缓存就超过了我们参数的内存使用！**要明确的是，LLaMA 2在更长的上下文中没有针对KV缓存大小进行优化（并不总是这么糟糕，因为通常$K$要小得多，如LLaMA-3），但这仍然具有说明性。我们不能在内存或延迟估计中忽视这些。

### 为LLaMA 2-13B建模吞吐量和延迟

让我们看看如果我们尝试在8xTPU v5es上以不同批量大小完全高效地执行生成，直到之前导出的最大理论吞吐量的临界批量大小（240）会发生什么。

| 批量大小                           |      1 |      8 |     16 |     32 |     64 |    240 |
| :-------------------------------- | -----: | -----: | -----: | -----: | -----: | -----: |
| KV缓存内存（GiB）                 |    6.7 |   53.6 |  107.2 |  214.4 |  428.8 |   1608 |
| 总内存（GiB）                     |   32.7 |   79.6 |  133.2 |  240.4 |  454.8 |   1634 |
| 理论步骤时间（ms）                |   4.98 |  12.13 |  20.30 |  36.65 |  69.33 | 249.09 |
| 理论吞吐量（tokens/s）            | 200.61 | 659.30 | 787.99 | 873.21 | 923.13 | 963.53 |

8x TPU v5es给我们128GiB的HBM，6.5TiB/s的HBM带宽（每个0.82TiB/s）和1600TF/s的计算。

对于这个模型，增加批量大小确实给我们更好的吞吐量，但我们迅速遭受递减回报。我们在批量大小16之后OOM，需要一个数量级更多的内存才能接近240。更大的拓扑可以改善延迟，但我们在每芯片吞吐量上遇到了瓶颈。

假设我们保持参数总数相同，但神奇地使KV缓存缩小5倍（比如，使用1:5 [GMQA](#提高生成吞吐量和延迟的技巧)，这意味着我们有8个KV头在40个Q头上共享——有关更多详细信息，请参见下一节）。

| 批量大小                           |      1 |        8 |       16 |       32 |       64 |      240 |
| :-------------------------------- | -----: | -------: | -------: | -------: | -------: | -------: |
| KV缓存内存（GiB）                 |   1.34 |    10.72 |    21.44 |    42.88 |    85.76 |    321.6 |
| 总内存（GiB）                     |  27.34 |    36.72 |    47.44 |    68.88 |   111.76 |    347.6 |
| 理论步骤时间（ms）                |   4.17 |     5.60 |     7.23 |    10.50 |    17.04 |    52.99 |
| 理论吞吐量（tokens/s）            | 239.94 | 1,429.19 | 2,212.48 | 3,047.62 | 3,756.62 | 4,529.34 |

使用较小的KV缓存，我们仍然有递减回报，但每芯片的理论吞吐量继续扩展到批量大小240。我们可以容纳更大的64批次，所有批量大小的延迟也始终更好。延迟、最大吞吐量和最大批量大小都得到了显著改善！事实上，后来的LLaMA代使用了这个确切的优化——LLaMA-3 8B有32个查询头和8个KV头（[来源](https://huggingface.co/MaziyarPanahi/Llama-3-13B-Instruct-v0.1/blob/dfdeb40bdb2c149dfa399ea2be0d56eb120f0831/config.json)）。

<p markdown=1 class="takeaway">**要点：**除了参数，KV缓存的大小对模型的最终推理性能有很大影响。我们希望通过架构决策和运行时优化的组合来控制它。</p>

## 提高生成吞吐量和延迟的技巧

自原始的[Attention is All You Need论文](https://arxiv.org/abs/1706.03762)以来，已经开发了许多技术来使模型更高效，通常专门针对KV缓存。一般来说，较小的KV缓存使得在不影响延迟的情况下更容易增加生成步骤的批量大小和上下文长度，并使Transformer周围的系统（如请求缓存）的生活更轻松。忽略对质量的影响，我们可能会看到：

**分组多查询注意力（又名GMQA，GQA）：**我们可以减少KV头的数量，并在注意力机制中与许多Q头共享它们。在极端情况下，可以在所有Q头之间共享单个KV头。这将KV缓存减少了Q:KV比率相对于纯MHA的因子，并且已经观察到模型的性能对这种变化相对不敏感。

{% include figure.liquid path="assets/img/gmqa.png" class="img-fluid" %}

这也有效地增加了注意力计算的算术强度（参见[第4节](../transformers)中的问题4）。

**混合一些局部注意力层：**局部注意力将上下文限制为小到中等大小的最大长度。在训练时间和预填充时间，这涉及将注意力矩阵掩码为对角线条带而不是三角形。这有效地限制了局部层的KV缓存的最大长度大小。通过将一些局部层与一些全局层混合到模型中，在长于局部窗口的上下文中，KV缓存的大小大大减少。

**跨层共享KV：**模型可以学习以某种模式跨层共享相同的KV缓存。虽然这确实减少了KV缓存大小，并在增加批量大小、缓存、离线存储等方面提供了好处。共享的KV缓存可能需要从HBM多次读取，*所以它不一定改善步骤时间。*

{% include figure.liquid path="assets/img/kv-sharing.png" class="img-fluid" caption="
 <b>左：</b>纯全局注意力的多层。<b>右：</b>一些全局/局部交错模式与相邻层共享的示例。来源：<a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai博客</a>。"%}

**量化：**推理通常对参数和KV的精度不太敏感。通过量化参数和KV缓存（例如到int8、int4、`fp8`等），我们可以节省两者的内存带宽，减少达到计算屋顶线所需的批量大小，并节省内存以在更大的批量大小下运行。量化的额外优势是，即使模型没有使用量化进行训练，它通常也可以在训练后应用。

**使用不规则HBM读取和分页注意力：**我们在上面的计算中为每个KV缓存分配了8k的上下文，但通常不需要从内存中读取整个KV缓存——请求具有广泛的长度分布范围，并且不使用模型的最大上下文，所以我们经常可以实现只读取KV缓存的非填充部分的内核（例如Flash Attention变体）。

分页注意力<d-cite key="paged"></d-cite>是对此的改进，它将KV缓存存储在操作系统风格的页表中，并且大部分避免了完全填充KV缓存。这增加了很多复杂性，但意味着每个批次只使用它需要的内存。这是一个运行时优化，所以它对架构是无关的。

{% include figure.liquid path="assets/img/paged-attention.png" class="img-fluid img-small" caption="<b>图：</b>在生成期间，单个token（第四个）关注多个KV缓存块/页。通过分页KV缓存，我们避免加载或存储超过我们需要的内存。取自<a href=\"https://arxiv.org/pdf/2309.06180\">PagedAttention论文</a>。" %}

<p markdown=1 class="takeaway">**大图景：**总的来说，与标准MHA Transformer相比，这些KV缓存优化可以将KV缓存大小减少一个数量级以上。这可以导致Transformer总成本的数量级改进。</p>

## 在多个加速器上分布推理

到目前为止，我们一直在回避如何扩展到单个芯片之外。按照[第5节](../training)，让我们探索可用的不同策略及其权衡。一如既往，我们将分别查看预填充和生成。

### 预填充

从屋顶线的角度来看，**预填充几乎与训练相同**，几乎所有相同的技术和权衡都适用——模型（Megatron）并行、序列分片（对于足够长的上下文）、流水线，甚至FSDP都是可行的！你只需要保留KV，以便稍后进行生成。与训练一样，增加芯片数量使我们可以访问更多的FLOPs/s（可能降低TTFT），但增加了通信开销（可能降低每芯片吞吐量）。

**分片预填充的一般规则：**这是预填充的一般规则集。我们假设我们只在单个序列上进行预填充（没有批次维度）：

1. *模型分片：*我们通常首先进行一定程度的模型并行，直到我们成为ICI受限。正如我们在[第5节](../training)中看到的，对于1轴，这大约是$F / 2550$（通常大约4-8路分片）。
2. *序列并行：*除此之外，我们进行序列并行（类似于数据并行，但在序列维度上分片）。虽然序列并行在注意力中引入了一些额外的通信，但在更长的上下文中通常相当小。与训练一样，我们可以重叠通信和计算（分别使用Megatron的集体矩阵乘法和环注意力）。

<p markdown=1 class="takeaway">**要点：**在预填充期间，几乎任何可以在训练期间工作的分片都可以正常工作。进行模型并行直到ICI界限，然后进行序列并行。</p>

### 生成

生成是比预填充更复杂的野兽。一方面，获得大批量大小更难，因为我们需要将许多请求批处理在一起。延迟目标更低。这些共同意味着我们通常更受内存限制并且对通信开销更敏感，这限制了我们的分片策略：

1. **FSDP是不可能的：**由于我们在将参数和KV缓存从HBM加载到MXU时受内存限制，我们不想通过ICI移动它们，ICI比HBM慢几个数量级。*我们希望移动激活而不是权重。*这意味着类似于FSDP的方法通常对生成完全不可行。<d-footnote>在训练后意外地将其保留是导致数量级回归的简单而常见的方法</d-footnote>

2. **没有理由进行数据并行：**纯数据并行是无用的，因为它复制了我们的参数，并不能帮助我们更快地加载参数。你最好启动模型的多个副本。<d-footnote>我们的意思是，在较小的批量大小下启动具有模型副本的多个服务器。模型级别的数据并行严格来说更糟。</d-footnote>

3. **没有序列=没有序列分片。**祝你序列分片好运。

_这主要给我们留下了用于密集模型生成的模型分片变体_。与预填充一样，我们可以做的最简单的事情是简单的模型并行（激活完全复制，权重在MLP的隐藏维度上完全分片），当我们成为ICI受限时，最多4-8路。然而，由于我们经常受内存带宽限制，我们实际上可以超越这个限制来改善延迟！

**生成的ICI界限注意事项：**在训练期间，我们希望受计算限制，所以我们的屋顶线查看我们的ICI通信何时比我们的FLOPs花费更长时间。然而，在生成期间，如果我们受参数加载的内存带宽限制，我们可以增加模型分片超过这一点，并以最小的吞吐量成本改善延迟。更多的模型分片使我们有更多的HBM来加载我们的权重，我们的FLOPs无关紧要。<d-footnote>从某种意义上说，FLOPs时间不会成为我们的瓶颈，所以我们需要担心的是ICI时间超过参数加载时间。</d-footnote>让我们看看在它成为瓶颈之前我们可以进行多少模型并行。

$$\begin{align*}T_\text{HBM通信} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{ICI通信} = \frac{2BD}{W_\text{ici}}\end{align*}$$

$$T_\text{ICI通信} > T_\text{HBM通信} \rightarrow \frac{W_\text{hbm}}{W_\text{ici}} > \frac{F}{Y \cdot B} \rightarrow Y > F / (B \cdot \beta)$$

其中$\beta = W_\text{hbm} / W_\text{ici}$。对于TPU v5e和TPU v6e，这个数字通常约为8。这意味着例如，如果$F$是16,384，$B$是32，我们理论上可以进行高达`16384 / (32 * 8) = 64`路的模型并行，而不会对吞吐量产生有意义的影响。这假设我们可以完全分片我们的KV缓存64路，这很困难：我们在下面讨论这个。

对于注意力层，我们还以Megatron风格在头上模型分片注意力$$W_Q$$和$$W_O$$。KV权重相当小，复制它们通常比超过$K$路分片更便宜。

<p markdown=1 class="takeaway">**要点：**我们在生成期间的唯一选择是模型并行的变体。我们的目标是移动激活而不是KV缓存或参数，它们更大。当我们的批量大小很大时，我们进行模型并行直到FLOPs-ICI界限（$F / \alpha$）。当我们的批量大小较小时，我们可以通过更多的模型分片来改善延迟（以适度的吞吐量成本）。当我们想要模型分片的方式多于我们拥有的KV头时，我们也可以沿批次维度分片我们的KV。</p>

### 分片KV缓存

**我们还有一个需要分片的额外数据结构——KV缓存。**同样，我们几乎总是更喜欢避免复制缓存，因为它是注意力延迟的主要来源。为此，我们首先沿头维度Megatron分片KV。这限于$K$路分片，所以对于头数较少的模型，我们尽可能多地分片头维度，然后沿批次维度分片，即$\text{KV}[2, B_Z, S, K_Y, H]$。这意味着KV缓存是完全分布的。

{% include figure.liquid path="assets/img/esta-figure.png" class="img-fluid" caption="<b>图：</b>（a）具有纯模型分片的多头注意力和（b）具有KV缓存批量分片的多查询注意力的注意力机制比较。注意我们需要两个额外的AllToAlls来将激活从模型分片转移到批量分片，以便它们可以作用于KV缓存。" %}

这样做的成本是每个注意力层两个AllToAlls——一个将Q激活转移到批量分片，以便我们可以使用批量分片计算注意力，一个将批量分片的注意力输出转移回纯模型分片。

{% details 这是完整的算法！ %}

这里我们将写出在$Y$和$Z$上都进行模型并行的完整注意力算法。我为同时使用$K$表示键张量和KV头维度而道歉。让$M=N/K$。

<div markdown=1 class="algorithm">

1. X[B, D] = ...（现有激活，来自上一层的未分片）
2. K[B<sub>Z</sub>, S, K<sub>Y</sub>, H], V[B<sub>Z</sub>, S, K, H] = ...（现有KV缓存，批量分片）
3. Q[B, N<sub>YZ</sub>, H] = X[B, D] \* W<sub>Q</sub>[D, N<sub>YZ</sub>, H]
4. Q[B<sub>Z</sub>, N<sub>Y</sub>, H] = **AllToAll**<sub>Z->B</sub>(Q[B, N<sub>YZ</sub>, H])
5. Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = **Reshape**(Q[B<sub>Z</sub>, N<sub>Y</sub>, H])
6. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] \*<sub>H</sub> K[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
7. O[B<sub>Z</sub>, S, K, M] = **Softmax**<sub>S</sub>(O[B<sub>Z</sub>, S, K<sub>Y</sub>])
8. O[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = O[B<sub>Z</sub>, S, K, M] \*<sub>S</sub> V[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
9. O[B, K<sub>Y</sub>, M<sub>Z</sub>, H] = **AllToAll**<sub>Z->M</sub>(O[B<sub>Z</sub>, K<sub>Y</sub>, M, H])
10. O[B, N<sub>YZ</sub>, H] = **Reshape**(O[B, K<sub>Y</sub>, M<sub>Z</sub>, H])
11. X[B, D] {U<sub>YZ</sub>} = W<sub>O</sub>[N<sub>YZ</sub>, H, D] \*<sub>N,H</sub> O[B, N<sub>YZ</sub>, H]
12. X[B, D] = **AllReduce**(X[B, D] { U<sub>YZ</sub>})

这相当复杂，但你可以大致看到它是如何工作的。新的通信适度昂贵，因为它们作用于我们的小激活，而作为回报，我们节省了大量加载KV（它们是静止的）的内存带宽。

</div>

{% enddetails %}

* **序列分片：**如果批量大小太小，或者上下文很长，我们可以序列分片KV缓存。同样，我们在这里累积跨分片的注意力时支付集体成本。首先我们需要AllGather Q激活，然后以类似于Flash Attention的方式累积KV。

## 设计有效的推理引擎

到目前为止，我们已经研究了如何有效地优化和分片单独的预填充和生成操作。要实际有效地使用它们，我们需要设计一个推理引擎，它可以在延迟/吞吐量帕累托前沿的我们选择的点上为这两个操作提供服务。

最简单的方法是简单地运行一批预填充，然后运行一批生成：

{% include figure.liquid path="assets/img/batched-prefill.png" class="img-fluid" caption="<b>图：</b>在最简单的设置中，请求被聚合，服务器在运行一批预填充和调用生成函数直到所有序列完成之间交替。" %}

这很容易实现，并且是大多数代码库中的第一个推理设置，但它有多个缺点：

1. **延迟很糟糕。**我们将预填充和生成批量大小耦合。在大预填充批量大小下，首个token时间（TTFT）很糟糕——你需要完成所有预填充，然后任何用户才能看到任何token。在小批量大小下，生成吞吐量很糟糕。
2. **我们在较长的生成上阻塞较短的生成。**许多序列将在其他序列之前完成，在生成期间留下空的批次槽，进一步损害生成吞吐量。随着批量大小和生成长度的增加，问题加剧。
3. **预填充被填充。**预填充被填充到最长的序列，我们浪费了很多计算。有解决方案，但历史上XLA使跳过这些FLOPs变得相当困难。同样，批量大小和预填充序列长度越大，这变得越糟。
4. **我们被迫在预填充和生成之间共享分片。**预填充和生成都在同一个片上，这意味着我们为两者使用相同的拓扑和分片（除非你保留两份权重副本），这通常对性能没有帮助，例如生成需要更多的模型分片。

因此，此方法仅建议用于边缘应用程序（通常只关心服务单个用户并使用具有较少FLOPs/字节的硬件）和Transformer代码库生命周期早期的快速迭代（由于其简单性）。

一个稍微更好的方法涉及以批量大小1执行预填充（其中它受计算限制但具有合理的延迟），但在生成期间将多个请求批处理在一起：

{% include figure.liquid path="assets/img/interleaving.png" class="img-fluid" %}

这将避免批量预填充造成的浪费TTFT，同时保持生成吞吐量高。我们称之为**交错**配置，因为我们"交错"预填充和生成步骤。这对于评估等批量生成应用程序非常强大，其中吞吐量是主要目标。编排器可以配置为在任何生成槽打开时立即优先预填充，即使对于非常大的生成批量大小也能确保高利用率。我们还可以避免将预填充填充到最大长度，因为它没有与另一个请求批处理。

主要缺点是当服务器执行预填充时，所有其他请求的生成都会暂停，因为预填充将消耗所有计算资源。用户A的响应正在忙于解码将被用户B的预填充阻塞。这意味着即使TTFT有所改善，平均而言token生成将是抖动的和缓慢的，这对于许多应用程序来说不是良好的用户体验——其他用户的预填充在请求的整体延迟的关键路径上。

为了解决这个问题，我们分离解码和预填充。虽然Transformer推理可以在一台服务器上完成，但从延迟的角度来看，在两组TPU/GPU上执行两个不同的任务通常更好。预填充服务器生成KV缓存，通过网络发送到生成服务器，生成服务器将多个缓存批处理在一起并为每个缓存生成token。我们称之为**"分解"**服务。

{% include figure.liquid path="assets/img/disaggregation.png" class="img-fluid" %}

这提供了几个优势：

1. **大规模低延迟**：用户的请求永远不会阻塞在另一个用户的请求上，除非预填充容量不足。请求应该立即预填充，然后发送到生成服务器，然后立即插入生成缓冲区。如果我们预期许多并发请求进入，我们可以独立于生成服务器的数量扩展预填充服务器的数量，这样用户就不会在预填充队列中停留很长时间。

2. **专业化：**通常，预填充和生成的延迟最优参数分片策略/硬件拓扑是完全不同的（例如，更多的模型并行对生成有用但对预填充没有用）。将两个操作限制为使用相同的分片会损害两者的性能，并且拥有两组权重会使用内存。此外，通过将预填充移动到自己的服务器上，它不需要保存任何KV缓存，除了它当前正在处理的那个。这意味着我们有更多的空闲内存用于历史缓存（见下一节）或优化预填充延迟。

一个缺点是KV缓存现在需要通过网络转移。这通常是可以接受的，但再次为减少KV缓存大小提供了动力。

<p markdown=1 class="takeaway">**要点：**对于延迟敏感、高吞吐量服务，我们通常必须将预填充和生成分离到单独的服务器中，预填充以批量1运行，生成将许多并发请求批处理在一起。</p>

### 连续批处理

上面的问题（2）激发了**连续批处理**的概念。我们优化和编译：

* 具有可变上下文长度的多个预填充函数，并将其插入某个KV缓冲区，某个最大批量大小和上下文长度/页数。
* 一个生成函数，它接收KV缓存，并为所有当前活动的请求执行生成步骤。

然后，我们将这些函数与编排器结合，编排器对传入的请求进行排队，根据可用的生成槽调用预填充和生成，处理历史缓存（见下一节）并流式传输token。

{% include figure.liquid path="assets/img/continuous-batching.gif" class="img-fluid" %}

### 前缀缓存

由于预填充昂贵且受计算限制（给我们更少的余地），减少其成本的最佳方法之一是少做。因为LLM是自回归的，查询["我"，"喜欢"，"狗"]和["我"，"喜欢"，"猫"]产生的KV缓存在前两个token中是相同的。这意味着，原则上，如果我们先计算"我喜欢狗"缓存，然后计算"我喜欢猫"缓存，我们只需要进行1/3的计算。我们可以通过重用缓存来节省大部分工作。这在几个特定情况下特别强大：

1. **聊天机器人**：大多数聊天机器人对话涉及严格追加到自身的来回对话。这意味着如果我们可以从每个对话回合保存KV缓存，我们可以跳过除最新token之外的所有计算。
2. **少样本提示**：如果我们有任何类型的少样本提示，这可以免费保存和重用。系统指令通常也具有这种形式。

这很难做的唯一原因是内存限制。正如我们所见，KV缓存很大（通常是许多GB），为了使缓存有用，我们需要保留它们直到后续查询到达。通常，预填充服务器上任何未使用的HBM都可以用于本地缓存系统。此外，加速器通常在其CPU主机上有很多内存（例如，8xTPUv5e服务器有128GiB的HBM，但大约450GiB的主机DRAM）。这个内存比HBM慢得多——通常太慢而无法进行生成步骤——但对于缓存读取来说足够快。在实践中：

* 因为KV缓存是处理初始请求的TPU集的本地缓存，我们需要某种形式的亲和路由来确保后续查询到达同一副本。这可能会导致负载平衡问题。
* 较小的KV缓存是有帮助的（再次）——它使我们能够在相同的空间中保存更多的KV缓存，并减少读取时间。
* KV缓存及其查找可以很自然地存储在树或字典树中。驱逐可以基于LRU进行。

{% include figure.liquid path="assets/img/prefix-caching-trie.png" class="img-fluid" caption="<b>图：</b>作为LRU字典树实现的KV前缀缓存。我们可以通过共享前缀来避免复制KV内存。来源：<a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai博客</a>。" %}

### 让我们看一个实现：JetStream

Google开源了一个实现此逻辑的库，称为[JetStream](https://github.com/google/JetStream)。服务器有一组"预填充引擎"和"生成引擎"，通常在不同的TPU片上，由单个控制器编排。预填充发生在"[预填充线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L499)"中，而生成发生在"[生成线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L629)"中。我们还有一个"[传输线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L592)"，编排将KV缓存从预填充复制到生成片。

引擎接口（在[这里](https://github.com/google/JetStream/blob/445f1aa8e857d0a09d72618e365daf80723bdf4c/jetstream/engine/engine_api.py#L138)实现）是任何LLM必须提供的通用接口。关键方法是：

* **prefill：**接收一组输入token并生成KV缓存。
* **insert：**接收KV缓存并将其插入生成正在生成的KV缓存批次中。
* **generate：**接收一组批量KV缓存并为每个批次条目生成一个token，为每个token将单个token的KV缓存追加到解码状态。

我们还有一个PyTorch版本的JetStream可在[这里](https://github.com/google/jetstream-pytorch)获得。

## 练习题

我将为本节发明一个基于LLaMA-2 13B的新模型。以下是详细信息：

| 超参数            | 值     |
| ----------------- | ------ |
| n\_layers (L)     | 64     |
| d\_model (D)      | 4,096  |
| d\_ff (F)         | 16,384 |
| n\_heads (N)      | 32     |
| n\_kv\_heads (K)  | 8      |
| d\_qkv (H)        | 256    |
| n\_embeddings (V) | 32,128 |

**问题1：**上述模型有多少参数？每个token的KV缓存有多大？*你可以假设我们共享输入和输出投影矩阵。*

{% details 点击这里查看答案。 %}

**参数数量：** 

* MLP参数数量：$L * D * F * 3$
* 注意力参数数量：$L * 2 * D * H * (N + K)$
* 词汇表参数：$D * V$（因为我们共享这些矩阵）

因此，我们的总参数数量是$L * D * (3F + 2H * (N + K)) + D * V$。代入上面的数字，我们有`64 * 4096 * (3*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 18.4e9`。因此，这个模型有大约184亿个参数。

{% enddetails %}

**问题2：**假设我们想在TPUv5e 4x4片上服务这个模型，并且可以在这个拓扑上完全分片我们的KV缓存。假设我们对所有内容都使用int8，我们可以容纳的最大批量大小是多少。如果我们将KV头的数量降至1会怎样？

**问题3：**假设我们完全受HBM带宽限制。将所有参数从HBM加载到MXU需要多长时间？*这是每步延迟的良好下界。*

**问题4：**假设我们想在TPUv5e 4x4片上服务这个模型。我们将如何分片它？*提示：也许先回答这些问题：*

1. 这个模型在ICI上的张量并行上界是什么？
2. 我们如何分片KV缓存？

对于这种分片，生成的大致每步延迟是多少？

**问题5：**假设上述模型实际上是MoE。MoE模型实际上是具有E个FFW块副本的密集模型。每个token通过k个FFW块，这些`k`被平均以产生输出。让我们使用`E=16`和`k=2`以及上述设置。

1. 它有多少参数？
2. 需要多大的批量大小才能成为FLOPs受限？
3. 每个token的KV缓存有多大（假设没有局部注意力）？
4. 具有T个token的前向传递涉及多少FLOPs？

**问题6：**使用MoE，我们可以进行"专家分片"，我们在网格的一个轴上分割我们的专家。在我们的标准符号中，我们的第一个FFW权重具有形状`[E, D, F]`，我们将其分片为[E<sub>Z</sub>, D<sub>X</sub>, F<sub>Y</sub>]，其中`X`仅在训练期间用作我们的FSDP维度。假设我们想在TPU v5e上进行推理：

1. 在Y=8，Z=16的TPU v5e 8x16片上，上述模型的HBM权重加载时间是多少？每个TPU有多少空闲HBM？
2. 我们可以容纳我们的模型的最小片是什么？

**问题7 [2D模型分片]：**这里我们将通过[ESTI论文](https://arxiv.org/pdf/2211.05102)称为2D权重静止分片的数学。我们在附录B中简要描述了这一点，但首先尝试做这个问题，看看你是否可以解决数学。2D权重静止分片的基本思想是沿$D$和$F$轴分片我们的权重，使每个块大致是正方形的。这减少了通信负载，并允许我们稍微扩展更远。

这是2D权重静止的算法：

<div markdown=1 class="algorithm">

1.  In[B, D<sub>X</sub>] = **AllGather**<sub>YZ</sub>(In[B, D<sub>XYZ</sub>])
2.  Tmp[B, F<sub>YZ</sub>] {U.X} = In[B, D<sub>X</sub>] \*<sub>D</sub> W<sub>in</sub>[D<sub>X</sub>, F<sub>YZ</sub>]
3.  Tmp[B, F<sub>YZ</sub>] = **AllReduce**<sub>X</sub>(Tmp[B, F<sub>YZ</sub>] {U.X})
4.  Out[B, D<sub>X</sub>] {U.YZ} = Tmp[B, F<sub>YZ</sub>] \*<sub>F</sub> W2[F<sub>YZ</sub>, D<sub>X</sub>]
5.  Out[B, D<sub>XYZ</sub>] = **ReduceScatter**<sub>YZ</sub>(Out[B, D<sub>X</sub>] {U.YZ})
</div>

你的目标是计算这个算法的$T_\text{math}$和$T_\text{comms}$，并找出它何时会优于传统的3D模型分片？

{% details 点击这里查看答案！ %}

让我们计算$T_\text{math}$和$T_\text{comms}$。我们所有的FLOPs都是完全分片的，所以如前所述，我们有$T_\text{math} = 4BDF / (N \cdot C)$，但我们的通信现在是

$$\begin{align*}
T_\text{2D通信} = \frac{2BD}{2X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}} + \frac{2BD}{2X \cdot W_\text{ici}} = \frac{2BD}{X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}}
\end{align*}$$

其中我们注意到AllReduce的成本是两倍，我们通过执行每个操作的轴数来缩放我们的通信。假设我们有选择拓扑的自由，并假设$F=4D$（如LLaMA-2），我们声称（通过一些基本微积分）$X$、$Y$和$Z$的最优值是$X = \sqrt{N / 8}$，$YZ = \sqrt{8N}$，所以总通信是

$$T_\text{2D通信} = \frac{2B}{W_\text{ici}} \left(\frac{D}{X} + \frac{8D}{YZ}\right) = \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \approx \frac{11.3 BD}{\sqrt{N} \cdot W_\text{ici}}$$

首先，从上面复制，正常的1D模型并行将有$T_\text{模型并行通信} = 4BD / (3 \cdot W_\text{ici})$，所以新通信何时更小？我们有

$$\begin{align*}
T_\text{模型并行通信} > T_\text{2D通信} \iff \frac{4BD}{3 \cdot W_\text{ici}} > \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \\
\iff N > 128 \cdot \left(\frac{3}{4}\right)^2 = 81
\end{align*}$$

对于一般的$F$，我们声称这个条件是

$$N > 32 \cdot \left(\frac{F}{D}\right) \cdot \left(\frac{3}{4}\right)^2$$

所以这告诉我们，如果我们有超过81个芯片，我们最好使用这个新方案。现在这是一个稍微奇怪的结果，因为我们历史上发现自己在大约20路张量并行时受ICI限制。但在这里，即使我们受通信限制，我们的总通信也会随着总芯片数的增加而继续减少！这告诉我们，我们可以继续增加我们的芯片，增加我们的批量大小，进行更多的参数缩放，并看到减少的延迟。

{% enddetails %}

<h3 markdown=1 class="next-section">第7部分到此结束！对于第8部分，了解我们如何在TPU上服务LLaMA 3，请点击[这里](../applied-inference)。</h3>

## 附录

### 附录A：批量大小> 240规则有多真实？

我们上面提供的简单规则，即我们的批量大小必须大于240个token才能受计算限制，大致是正确的，但忽略了TPU在其他操作不使用所有可用HBM时预取权重的一些能力，比如在进行设备间通信时。

这是一个经验图，显示了具有d<sub>model</sub> 8192、d<sub>ff</sub> 32768和每层只有2个矩阵乘法的小型Transformer的层时间（以微秒为单位）。这来自[这个Colab笔记本](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)。你会看到步骤时间增长非常缓慢，直到批量大约240，然后线性增加。

{% include figure.liquid path="assets/img/batch-scaling-latency.png" class="img-fluid img-small" %}

这是tokens / us的实际吞吐量。这使论点相当清楚。由于我们的层在这里分片4路大约600M参数，我们预计最小延迟大约为365us。

{% include figure.liquid path="assets/img/batch-scaling-throughput.png" class="img-fluid img-small" %}

所以至少在这个模型中，我们确实看到吞吐量增加直到每个数据并行分片大约BS240。

### 附录B：2D权重静止分片

随着拓扑的增长，如果我们可以访问更高维度的网格（如TPU的网格），可以通过"**2D权重分片**"进一步完善这一点。通过引入第二个分片轴。我们称之为"**2D权重静止**"，在[高效扩展Transformer推理论文](https://arxiv.org/abs/2211.05102)中有更详细的描述。

因为我们只在Megatron中分片隐藏的$$F$$维度，一旦芯片数量随着1D分片增长，它可能会变得比$$E$$（$$d_\text{model}$$维度）小得多。这意味着在更大的批量大小下，在应用MLP的第一层后在隐藏维度上执行部分集体可能更经济。

{% include figure.liquid path="assets/img/2d-weight-stationary.png" class="img-fluid img-small" %}

此图显示：

1. 1D权重静止分片，又名纯Megatron分片，其中激活在AllGather后完全复制，权重在隐藏F维度上完全分片。
2. 2D权重静止分片，其中权重在隐藏F和归约E维度上都分片，激活在E维度上分片。我们在第一层之前在（yz）轴上执行AllGather，然后在（x）轴上执行ReduceScatter。

对于注意力层，Megatron风格分片对于较少数量的芯片也相对简单。然而，Megatron发生在$$n_\text{heads}$$维度上，这对可能的分片数量设置了限制。修改2D分片（而不是分片隐藏，我们分片$$n_\text{heads}$$维度），我们获得了进一步扩展的能力。

### 附录C：延迟限制通信

作为回顾，在[第3节](../sharding)中，我们导出了在1D环链路上执行AllGather到每个TPU上大小为B的张量所需的时间量，全双工带宽为WICI，延迟为Tmin。

$$T_{total} = \max\left(\frac{T_{min} \cdot |X|}{2}, \frac{B}{W_{ICI}}\right)$$

对于大的B，墙钟时间保持相对恒定，因为当你向系统添加更多芯片时，你同时扩展了执行操作所需的数据移动量和可用的总带宽。

{% include figure.liquid path="assets/img/all-gather.gif" class="img-fluid" %}

由于延迟优化推理期间移动的数据量相对较少，激活上的集体通常受延迟项限制（特别是对于小批量大小）。可以通过计算完成之前需要完成的跳数来很容易地可视化延迟。

在TPU上，如果通信的张量大小相关部分每跳少于1微秒（跳是两个相邻设备之间的通信），我们可能会受到实际调度集体的固定开销的瓶颈。使用`4.5e10`单向ICI带宽，当$$(\text{字节} / n_\text{分片}) / 4.5e10 < 1e-6$$时，ICI通信变为延迟受限。对于8路Megatron分片，这是当`buffer_size < 360kB`时。**这在推理期间实际上并不小：**使用`BS=16`和`D=8192`的int8，我们的激活将使用`16*8192=131kB`，所以我们已经受延迟限制。

<p markdown=1 class="takeaway">**要点：**当$$\text{总字节} < W_{ICI} \times 1e-6$$时，我们的通信变为延迟受限。例如，对于$$Y$$上的模型并行，当$$Y > BD / 45,000$$时，我们在int8中受限。</p>

这里可以与计算屋顶线进行平行——我们正在承担一些小操作的固定成本（通信的延迟，矩阵乘法的内存带宽）。

### 附录D：推测采样

当我们*真正*关心端到端延迟时，我们可以使用一个额外的技巧，称为推测采样<d-cite key="spec1"></d-cite><d-cite key="spec2"></d-cite>。作为回顾，我们通常从大型Transformer一个接一个地生成token：

{% include figure.liquid path="assets/img/spec-sampling1.png" class="img-fluid" %}

使用推测采样，我们使用一个更小、更便宜的模型来生成token，然后用大模型检查结果。这在*贪婪解码*中最容易理解：

{% include figure.liquid path="assets/img/spec-sampling2.png" class="img-fluid" %}

1. 我们从一些更小、更便宜的模型贪婪地采样。理想情况下，我们使用训练来匹配较大模型的模型，例如通过蒸馏，但它可以简单到只使用n-grams或token匹配一小段文本语料库。
2. 在我们生成了K个token之后，我们使用大模型计算到目前为止我们生成的所有token的下一个token logits。
3. 由于我们正在贪婪地解码，我们可以检查较小模型生成的token是否具有所有可能token的最高概率。如果其中一个token错误，我们取最长的正确前缀并用正确的token替换第一个错误的token，然后回到（1）。如果所有token都正确，我们可以使用最后一个正确的logit在回到（1）之前采样一个额外的token。

**为什么这是延迟胜利？**这个方案仍然要求我们为每个token通过大模型进行相当于一次前向传递的FLOPs，但因为我们可以将一堆token批处理在一起，我们可以在一次前向传递中完成所有这些FLOPs，并利用我们*不*受*计算限制*的事实来免费评分更多token。

每个接受的token在FLOPs方面平均变得更昂贵（因为有些会被拒绝，我们必须调用草稿模型），但我们从硬件中榨取更多FLOPs，小模型很便宜，所以我们总体上获胜。由于一切都已被大模型检查，我们根本不会改变采样分布（尽管非贪婪的确切轨迹会有所不同）。

对于正常的自回归采样，token/s与步骤时间相同。我们仍然受制于根据这里的算术强度部分的理论最小步骤时间（事实上，推测采样步骤时间通常比正常自回归采样慢很多，但因为我们平均每步获得超过1个token，我们可以获得更好的tokens/s）。

{% include figure.liquid path="assets/img/spec-sampling3.png" class="img-fluid" caption="<b>图：</b>此图显示了Chinchilla（DeepMind的70B模型）与4B参数起草者（小模型）的每步延迟和推测成功率。对于XSum（自然语言数据集），理想的推测量大约是提前3-4个token，而HumanEval（编码数据集）更可预测，从更积极的推测中看到收益。" %}

**这如何用于非贪婪解码？**这有点复杂，但本质上归结为Metropolis-Hastings启发的算法，其中我们有从logits导出的$$P_{\text{草稿模型}}(\text{选择的token})$$和$$P_{\text{目标模型}}(\text{选择的token})$$，如果这些概率的比率小于某个阈值，则概率性地拒绝选择的token。

这[两篇](https://arxiv.org/abs/2211.17192)[论文](https://arxiv.org/abs/2302.01318)同时导出了这个，并有很好的实践示例。

<p markdown=1 class="takeaway">**要点：**推测采样是另一个强大的杠杆，用于交易吞吐量以获得更好的每token延迟。然而，在批量大小受限的情况下（例如小硬件占用空间或大KV缓存），它成为双赢。</p>