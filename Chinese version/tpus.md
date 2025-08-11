---
layout: distill
title: "如何理解TPU"
# permalink: /main/
description: "本节主要介绍TPU的工作原理、如何通过网络连接实现多芯片训练和推理，以及这如何影响我们喜爱的算法的性能。对GPU用户也有很多有用的内容！"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting

section_number: 2

previous_section_url: "roofline"
previous_section_name: "第1部分：Roofline分析"

next_section_url: sharding
next_section_name: "第3部分：分片"

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
  - name: 什么是TPU？
  - name: TPU网络
  - name: 关键要点
  - name: 练习题
  - name: 附录
  - subsections:
    - name: "附录A：更多TPU内部结构"
    - name: "附录B：关于GPU的一切"
    - name: "附录C：脉动阵列如何工作？"

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

## 什么是TPU？

**TPU基本上是一个专门进行矩阵乘法的计算核心（称为TensorCore），连接到一堆快速内存（称为高带宽内存或HBM）<d-cite key="tpu_paper"></d-cite>。** 这里是一个图表：

{% include figure.liquid path="assets/img/tpu-chip.png" class="img-fluid" caption="<b>图：</b>TPU芯片的基本组件。TensorCore是左侧的灰色框，包含矩阵乘法单元（MXU）、向量单元（VPU）和向量内存（VMEM）。" %}

您可以将TensorCore基本上看作是一个非常优秀的矩阵乘法机器，但它还有一些其他值得注意的功能。TensorCore有三个关键单元：

* **MXU**（矩阵乘法单元）是TensorCore的核心。对于大多数TPU代，它使用脉动阵列每8个周期执行一次 `bfloat16[8,128] @ bf16[128,128] -> f32[8,128]` 矩阵乘法<d-footnote>TPU v6e（Trillium）有256x256的MXU，而所有之前的代都使用128x128</d-footnote>（详见<a href="#appendix-c-how-does-a-systolic-array-work">附录B</a>）。
  * 在TPU v5e上以1.5GHz运行时，每个MXU约为 `5e13` bf16 FLOPs/s。大多数TensorCore有2或4个MXU，例如TPU v5e的总bf16 FLOPs/s为 `2e14`。
  * TPU还支持更低精度的矩阵乘法，具有更高的吞吐量（例如，每个TPU v5e芯片可以执行 `4e14` int8操作/秒）。

* **VPU**（向量处理单元）执行通用数学运算，如ReLU激活或向量之间的逐点加法或乘法。归约（求和）也在这里执行。<a href="#appendix-a-more-on-tpu-internals">附录A</a>提供更多细节。
* **VMEM**（向量内存）是位于TensorCore中的片上暂存器，靠近计算单元。它比HBM小得多（例如，TPU v5e上为128 MiB），但对MXU的带宽高得多。VMEM的运作有点像CPU上的L1/L2缓存，但更大并且由程序员控制。HBM中的数据需要先复制到VMEM中，TensorCore才能对其进行任何计算。

**TPU在矩阵乘法方面非常非常快**。这主要是它们所做的事情，而且做得很好。[TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture)是迄今为止最强大的TPU之一，可以执行 `2.5e14` bf16 FLOPs/秒/核心或 `5e14` bf16 FLOPs/秒/芯片。一个包含8960个芯片的单个pod可以执行4 exaflops/秒。这是*非常多的*。这是世界上最强大的超级计算机之一。而Google有很多这样的计算机。<d-footnote>TPU及其脉动阵列特别是如此强大的硬件加速器，因为矩阵乘法是少数几个使用$O(n^3)$计算量和$O(n^2)$字节的算法之一。这使得普通的ALU很容易被计算而不是内存带宽所瓶颈。</d-footnote>

上面的图表还包括其他一些组件，如SMEM和标量单元，用于控制流处理，在<a href="#appendix-a-more-on-tpu-internals">附录A</a>中简要讨论，但不是关键的理解。另一方面，HBM很重要且相当简单：

* **HBM**（高带宽内存）是一大块快速内存，存储供TensorCore使用的张量。HBM通常具有数十GB的容量（例如，[TPU v5e有16GiB的HBM](https://cloud.google.com/tpu/docs/v5e#system_architecture)）。

  * 当需要进行计算时，张量从HBM通过VMEM（见下文）流入MXU，结果从VMEM写回HBM。

  * HBM和TensorCore之间（通过VMEM）的带宽称为"HBM带宽"（通常约为1-2TB/秒），限制了在内存受限工作负载中计算的速度。

**通常，所有TPU操作都是流水线化和重叠的。** 要执行矩阵乘法 $X \cdot A \to Y$，TPU首先需要将矩阵 $A$ 和 $X$ 的块从HBM复制到VMEM，然后将它们加载到MXU中，MXU将8x128（对于$X$）和128x128（对于$A$）的块相乘，然后将结果逐块复制回HBM。为了高效地执行此操作，矩阵乘法被流水线化，使得与VMEM的复制与MXU工作重叠。这允许MXU继续工作而不是等待内存传输，使矩阵乘法保持计算受限，而不是内存受限。

这里是如何从HBM执行逐元素乘积的示例：

{% include figure.liquid path="assets/img/pointwise-product.gif" caption="<b>图：</b>展示在TPU上执行逐点乘积的动画，字节从HBM加载。注意字节如何分块从内存流出，部分结果如何在不等待完整数组实现的情况下被流水线传回。" %}

矩阵乘法看起来几乎相同，只是它会加载到MXU而不是VPU/向量单元，并且加载和存储会以不同的顺序发生，因为相同的权重块用于多个激活块。您可以看到数据块流入VMEM，然后进入VREG（向量寄存器），然后进入向量单元，然后返回VMEM和HBM。正如我们即将看到的，如果从HBM到VMEM的加载比向量单元（或MXU）中的FLOP慢，我们就会变成"带宽受限"，因为我们让VPU或MXU缺乏工作。

<p markdown=1 class="takeaway">**关键要点：** TPU非常简单。它们从HBM加载权重到VMEM，然后从VMEM加载到脉动阵列，该阵列每秒可以执行约200万亿次乘加运算。HBM $\leftrightarrow$ VMEM和VMEM $\leftrightarrow$ 脉动阵列带宽设定了TPU可以高效执行哪些计算的基本限制。</p>

**VMEM和算术强度：** VMEM比HBM小得多，但它对MXU的带宽高得多。正如我们在[第1节](../roofline)中看到的，这意味着如果算法可以将其所有输入/输出放入VMEM，它就不太可能遇到通信瓶颈。这在计算的算术强度较差时特别有用：VMEM带宽约为HBM带宽的22倍，这意味着从/向VMEM读写的MXU操作只需要10-20的算术强度即可达到峰值FLOP利用率。这意味着如果我们可以将权重放入VMEM而不是HBM，我们的矩阵乘法可以在更小的批量大小下实现FLOP受限。这也意味着基本上具有较低算术强度的算法仍然可以高效。VMEM只是太小了，这通常是一个挑战。<d-footnote>我们有时会谈论VMEM预取，这指的是提前在VMEM中加载权重，这样我们就可以掩盖为矩阵乘法加载的成本。例如，在正常的Transformer中，我们有时可以在注意力期间将我们的大型前馈权重加载到VMEM中，如果我们是内存带宽受限的，这可以隐藏权重加载的成本。这需要我们的权重足够小或足够分片，以便将单个层放入VMEM并留有空间。</d-footnote>

{% include figure.liquid path="assets/img/tpu-bandwidth.png" class="img-fluid" %}

**一个TPU芯片通常（但不总是）由两个共享内存的TPU核心组成，可以被视为一个具有两倍FLOP的大型加速器**（称为"megacore"配置）。自TPU v4以来一直如此。较旧的TPU芯片具有独立的内存，被视为两个独立的加速器（TPU v3及更早版本）。像TPU v5e这样的推理优化芯片每个芯片只有一个TPU核心。

{% include figure.liquid path="assets/img/cores.png" class="img-fluid img-small" %}

**芯片**以**4个为一组排列在"托盘"上**，通过**PCIe网络连接到CPU主机。** 这是大多数读者熟悉的格式，通过Colab或单个TPU-VM公开的4个芯片（8个核心，尽管通常被视为4个逻辑megacore）。对于像TPU v5e这样的推理芯片，我们每个主机有2个托盘，而不是1个，但每个芯片也只有1个核心，给我们8个芯片= 8个核心。<d-footnote>在Cloud TPU VM上，每个托盘作为单独VM的一部分公开，因此再次有4个核心可见。</d-footnote>

{% include figure.liquid path="assets/img/pcie.png" class="img-fluid" %}

**PCIe带宽有限：** 像HBM $\leftrightarrow$ VMEM链接一样，CPU $\leftrightarrow$ HBM PCIe连接具有特定的带宽，限制了您可以从主机内存加载到HBM或反之的速度。例如，TPU v4的PCIe带宽为每个方向16GB/秒，因此比HBM慢近100倍。我们*可以*将数据加载/卸载到主机（CPU）RAM，但速度不是很快。

## TPU网络

**芯片通过Pod中的ICI网络相互连接**。在较旧的代（TPU v2和TPU v3）、推理芯片（例如TPU v5e）和Trilium（TPU v6e）中，ICI（"芯片间互连"）连接4个最近的邻居（带有边缘链接以形成2D环面）。TPU v4和TPU v5p连接到最近的6个邻居（形成3D环面）。请注意，这些连接**不**通过它们的主机，它们是芯片之间的直接链接。

{% include figure.liquid path="assets/img/ici-wraparound.png" class="img-fluid img-small" %}

环面结构将任意两个节点之间的最大距离从$N$减少到$N / 2$，使通信更快。TPU还具有"扭曲环面"配置，以类似莫比乌斯带的拓扑包裹环面，以进一步减少节点之间的平均距离。

**TPU pod（通过ICI连接）可以变得非常大：** 最大的pod大小（称为**superpod**）对于TPU v4为`16x16x16`，对于TPU v5p为`16x20x28`。这些大型pod由通过[光学环绕链接](https://arxiv.org/pdf/2208.10041)<d-footnote>光学开关只是具有相同ICI带宽的可重新配置连接。它只是让我们在保留环绕链接的同时连接立方体。</d-footnote>连接的可重新配置的`4x4x4`芯片立方体组成，我们可以重新配置以连接非常大的拓扑。

{% include figure.liquid path="assets/img/tpu-rack.png" class="img-fluid" %}

也可以请求较小的拓扑（例如`2x2x1`、`2x2x2`），尽管没有环绕。这是一个重要的警告，因为它通常会使大多数通信的时间翻倍。任何完整立方体的倍数（例如`4x4x4`或`4x4x8`）将由光学开关提供环绕。<d-footnote>请注意，`2x2x4`不会有任何环绕，因为它们由光学开关提供，而光学开关仅在完整立方体上可用。然而，TPU v5e 8x16在较长的轴上_将_有环绕，因为它不使用可重新配置的光学网络。</d-footnote>

{% include figure.liquid path="assets/img/subslices.png" class="img-fluid" %}

TPU v5e和Trillium pod由单个`16x16` 2D环面组成，沿任何大小为16的轴都有环绕（意味着`8x16`在长轴上有环绕）。TPU v5e和v6e（Trillium）无法扩展到16x16环面之外，但pod仍然可以通过标准数据中心网络（DCN）相互通信，DCN将TPU主机相互连接。同样，可以请求较小的拓扑，在维度$<16$时没有环绕。

{% include figure.liquid path="assets/img/more-subslices.png" class="img-fluid" %}

**这种最近邻连接是TPU和GPU之间的关键区别**。GPU通过开关层次结构连接，近似于每个GPU之间的点对点连接，而不是像TPU那样使用本地连接。通常，节点内的GPU（H100为8个GPU或B200多达500个）直接连接，而较大的拓扑需要每个GPU之间O(log(N))跳。一方面，这意味着GPU可以在节点内以单个低延迟跳发送任意数据。另一方面，TPU显著更便宜（因为NVLink开关很昂贵）且更容易连接在一起，并且可以扩展到更大的拓扑，因为每个设备的链接数量和每个设备的带宽是恒定的。

**ICI相对于DCN非常快，但仍然比HBM带宽慢。** 例如，[TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture)具有：

* 每个芯片`2.5e12`字节/秒（2.5 TB/s）的HBM带宽。
* 每个轴`9e10`字节/秒（90<d-footnote>上面的页面列出了100 GB/s的带宽，与这里列出的略有不同。TPU ICI链接根据执行的操作具有略微不同的带宽。您通常可以毫无顾虑地使用本文档中的数字。</d-footnote> GB/s）的ICI带宽，每个芯片有3个轴。
* 每个主机`2.5e10`字节/秒（25 GB/s）的DCN（出口）带宽。由于我们通常每个主机有8个TPU，这实际上更接近每个芯片`3.1e9`字节/秒。

这意味着当我们将模型分割到多个芯片时，我们需要小心避免用较慢的跨设备通信瓶颈MXU。

**多切片训练：** 一组ICI连接的TPU称为**切片**。不同的切片可以使用DCN相互连接，例如链接不同pod上的切片。由于DCN是比ICI慢得多的连接，因此应该尽量限制我们的计算必须等待来自DCN的数据的程度。DCN是主机到主机的，因此要通过DCN从TPU传输缓冲区到TPU，我们首先需要通过PCIe传输到主机，然后通过网络出口，然后通过目标主机网络入口，然后通过PCIe进入HBM。

## 关键要点

* TPU很简单，在大多数情况下可以被视为连接到内存（超快）、通过ICI连接到其他芯片（相当快）以及通过DCN连接到数据中心其余部分（有点快）的矩阵乘法单元。

* 通信受到我们各种网络带宽的限制，按速度顺序：
  * HBM带宽：TensorCore与其相关HBM之间。
  * ICI带宽：TPU芯片与其最近的4或6个邻居之间。
  * PCIe带宽：CPU主机与其相关的芯片托盘之间。
  * DCN带宽：多个CPU主机之间，通常是未通过ICI连接的主机。

* **在切片内，TPU仅通过ICI连接到其最近的邻居。** 这意味着切片中远距离芯片之间通过ICI的通信需要首先跳过中间的芯片。

* **权重矩阵需要在两个维度上填充到至少128的大小**（TPU v6上为256）以填满MXU（实际上，较小的轴被填充到128）。

* **较低精度的矩阵乘法往往更快。** 对于支持它的代，TPU可以执行int8或int4 FLOP的速度大约是bfloat16 FLOP的2倍/4倍。VPU操作仍以fp32执行。

* 为了避免瓶颈TPU计算单元，我们需要**确保每个通道的通信量与其速度成比例**。

* **这里是我们芯片的一些具体数字：**

| 型号                                        | Pod大小  | 主机大小  | HBM容量/芯片 | HBM带宽/芯片（字节/秒） | FLOPs/秒/芯片（bf16） | FLOPs/秒/芯片（int8） |
| :----------------------------------------- | :------: | :-------: | :----------: | :-------------------: | :------------------: | :------------------: |
| <span class="nowrap-header">TPU v3</span>  |  32x32   |    4x2    |     32GB     |        9.0e11         |        1.4e14        |        1.4e14        |
| <span class="nowrap-header">TPU v4p</span> | 16x16x16 |   2x2x1   |     32GB     |        1.2e12         |       2.75e14        |       2.75e14        |
| <span class="nowrap-header">TPU v5p</span> | 16x20x28 |   2x2x1   |     96GB     |        2.8e12         |       4.59e14        |       9.18e14        |
| <span class="nowrap-header">TPU v5e</span> |  16x16   |    4x2    |     16GB     |        8.1e11         |       1.97e14        |       3.94e14        |
| <span class="nowrap-header">TPU v6e</span> |  16x16   |    4x2    |     32GB     |        1.6e12         |       9.20e14        |       1.84e15        |

主机大小指连接到单个主机的TPU拓扑（例如，TPU v5e有一个连接到4x2拓扑中8个TPU的单个CPU主机）。以下是互连数字：

| 型号        | ICI带宽/链接（单向，字节/秒） | ICI带宽/链接（双向，字节/秒） |
| :---------- | :-------------------------: | :-------------------------: |
| **TPU v3**  |            1e11             |            2e11             |
| **TPU v4p** |           4.5e10            |            9e10             |
| **TPU v5p** |            9e10             |           1.8e11            |
| **TPU v5e** |           4.5e10            |            9e10             |
| **TPU v6e** |            9e10             |           1.8e11            |

我们包括单向（单向）带宽和双向（双向）带宽，因为单向带宽更真实地反映硬件，但双向带宽在涉及完整环的方程中更常出现。<d-footnote>通过双向（双向）带宽，我们指的是可以沿单个链接在两个方向发送的总字节数，或者同样地，从单个TPU沿特定轴的总出站字节数，假设我们可以高效地使用两个链接。当我们有一个功能环时，这是真的，即当我们在特定轴上有环绕连接时。当我们有完整的16轴时，这在推理芯片上发生，或者在训练芯片（v*p）上，当我们有一个是4的倍数的轴时。我们更喜欢使用双向带宽，因为它经常出现在涉及双向通信的计算中。</d-footnote>

PCIe带宽通常约为每个芯片`1.5e10`字节/秒<d-footnote>Trillium（TPU v6e）有32GB/s，大约是v5的2倍。</d-footnote>，而DCN带宽通常约为每个主机`2.5e10`字节/秒。为了完整性，我们包括单向和双向带宽。当我们可以访问完整的环绕环时，双向带宽通常是更有用的数字，而单向带宽更真实地反映硬件。

## 练习题

这些数字有点枯燥，但它们让您可以对模型性能进行基本的roofline估计。让我们解决几个问题来解释为什么这很有用。您将在第3部分看到更多示例。

**问题1 [限定LLM延迟]：** 假设您想从一个200B参数的bf16模型中采样，该模型分布在32个TPU v4p上。将所有参数从HBM加载到脉动阵列需要多长时间？*提示：使用上面的数字。*

{% details 点击这里查看答案。 %}

**答案：** 我们在32个芯片上加载`sizeof(bf16) * 200e9 = 400e9`字节，意味着每个芯片12.5e9字节，每个芯片的HBM带宽为1.23e12。因此加载大约需要10ms。

这很酷，因为*这是从模型采样的延迟的合理下界*。每个采样步骤都需要从HBM加载所有参数，因此不能少于10毫秒。实际上，在小批量大小下，这几乎是可以实现的。

{% enddetails %}

**问题2 [TPU细节]：** 考虑一个完整的TPU v5e pod。总共有多少个CPU主机？多少个TPU TensorCore？整个pod的总FLOPs/s是多少？总HBM是多少？对TPU v5p pod进行相同的练习。

{% details 点击这里查看答案。 %}

**答案：** 对于TPU v5e，每个pod是`16x16`，每个主机是4x2切片，所以我们有`16*16 / 8 = 32`个主机。对于TPU v5e，每个TPU只有一个核心，所以我们有256个TensorCore。bfloat16的总FLOPs/s是`16*16*2e14 = 5.1e16`。每个芯片有16GB的HBM，所以总共是`256 * 16 = 4TB`的内存。

对于完整的TPU v5p pod，我们有`16x20x28`个芯片，每个主机是2x2x1，所以我们有`16*20*28 / 2*2 = 2,240`个主机。对于TPU v5p，每个TPU有两个TensorCore，所以我们有`8960 * 2 = 17,920`个核心。bfloat16的总FLOPs/s是`8960 * 4.5e14 = 4e18`。每个芯片有96GB的HBM，所以总共是`8960 * 96 = 860TB`的内存。

{% enddetails %}

**问题3 [PCIe操作强度]：** 想象我们被迫在主机DRAM中存储一个大权重矩阵$A$，类型为$\text{bfloat16}[D, F]$，以及一批激活$x$，类型为$\text{bfloat16}[B, D]$，并希望对它们进行矩阵乘法。这在单个主机上运行，我们使用连接到它的单个TPU v6e芯片。您可以假设$B \ll D$，并且$F = 4D$（我们将在未来的章节中看到为什么这些是合理的假设）。为了保持在PCIe上的FLOP受限，我们需要的最小批量大小$B$是多少？假设PCIe带宽为每秒1.5e10字节。

{% details 点击这里查看答案。 %}

**答案：** 我们必须执行$2BDF$个浮点运算，每个芯片可以执行`9.2e14`个浮点运算每秒。这需要$2BDF / 9.2e14$秒来执行。我们必须从DRAM加载$2DF + 2BD$字节，并将$2BF$字节写回。我们受到PCIe传输速度的瓶颈，因此我们需要$2 \cdot (BD + DF + BF) / 1.5e10$秒来传输数据到TPU和从TPU传输数据。由于我们希望计算时间比权重加载时间长，假设我们可以将所有权重加载与计算重叠，我们希望$2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.5e10$。我们可以使用我们的假设$B \ll D$和$F = 4D$来简化这个，得到

$$\frac{8BD^2}{9.2e14} > \frac{8D^2}{1.5e10}$$

或

$$B > \frac{9.2e14}{1.5e10} \simeq 61,000$$

{% enddetails %}

**问题4 [通用矩阵乘法延迟]：** 假设我们想将权重矩阵int8[16384, 4096]乘以大小为int8[B, 4096]的激活矩阵，其中B是某个未知的批量大小。假设我们开始时在1个TPUv5e上。

1. 这个乘法作为B的函数需要多长时间？*提示：计算从HBM加载数组需要多长时间以及乘法实际需要多长时间可能会有所帮助。哪个是瓶颈？*
2. 如果我们想从VMEM运行这个操作呢？作为B的函数需要多长时间？

{% details 点击这里查看答案。 %}

**答案：**（1）我们需要执行的浮点运算数量是$2 \cdot 4096 \cdot 16384 \cdot B = 1.3e8 \cdot B$。所以$T_{\text{math}} = (1.3e8 \cdot B) / 3.94e14$秒。我们需要从HBM加载$16384 \cdot 4096 + 4096 \cdot B$字节到VMEM，并从VMEM写回$16384 \cdot B$字节到HBM。这意味着$T_{\text{comms}} = (6.7e7 + 2e4\cdot B) / 8.1e11$秒。假设通信和计算尽可能多地重叠，整个乘法将大约需要

$$\max\{T_{\text{math}}, T_{\text{comms}}\} = \max\left\{\frac{6.7e7 + 2e4\cdot B}{8.1e11}, \frac{1.3e8 \cdot B}{3.94e14}\right\}$$

当$\frac{6.7e7 + 2e4\cdot B}{8.1e11} < \frac{1.3e8 \cdot B}{3.94e14}$时，我们将受FLOP限制，或者等效地，$B > 271$。这比我们下面推导的240数字略大，因为我们考虑了$$D$$和$$F$$的全部影响。

（2）如果我们从VMEM加载，让我们将VMEM带宽到MXU视为HBM $\leftrightarrow$ VMEM带宽的22倍。这将我们的数据加载分母从8.1e11变为1.78e13，我们得到$B > 11$。请注意，在实践中，我们不能将所有VMEM带宽专用于加载$W$，因此在实践中它将更接近20。

{% enddetails %}

**问题5 [ICI带宽]：** 假设我们有一个TPU v5e `4x4`切片。假设我们想从`TPU{0,0}`发送类型为`bfloat16[8, 128, 8192]`的数组到`TPU{3, 3}`。假设TPU v5e的每跳延迟为$1\mu s$。

1. 第一个字节多久会到达目的地？
2. 整个传输需要多长时间？

{% details 点击这里查看答案。 %}

**答案：** 在TPUv5e中，我们有2D连接。因为我们只有一个`4x4`切片（没有大小为16的轴），我们没有环绕连接。因此，我们的目标芯片可以从两个端口接收数据，同样，我们的源芯片可以从两个端口发送数据。我们必须传输的数据量是`2 * 8 * 128 * 8192 = 1.7e7`字节。我们可以同时从两个端口传输（即向右发送一半数组，向下发送一半），因此我们每秒传输`2 * 4.5e10 = 9e10`字节，这意味着传输整个数组大约需要`1.7e7 / 9e10 = 188us`（假设我们受带宽限制）。在`4x4`切片中，芯片$(0, 0)$和$(3, 3)$之间有六跳，因为对于少于16个芯片的轴没有环绕链接。由于每跳的延迟约为$1\mu s$，第一个字节将在约`6us`到达，整个传输将需要`188us`。

{% enddetails %}

**问题6 [综合运用，困难]：** 想象你有一个大矩阵**A**：`int8[128 * 1024, 128 * 1024]`均匀分片在TPU v5e 4x4切片上，但卸载到每个芯片上的主机DRAM。假设你想将整个数组复制到TPU{0, 0}并将其乘以向量`bf16[8, 128 * 1024]`。这需要多长时间？*提示：使用上面的数字。*

{% details 点击这里查看答案。 %}

**答案：** 让我们首先概述我们必须执行的操作。我们的数组约为16GB。从上表中，TPU v5e主机具有4x2拓扑，因此4x4有2个主机。因此，由于我们的数组是均匀分片的，每个主机实际上包含数组的1/2块，即8GB。我们需要将这些块全部复制到TPU{0,0}，这给了我们两个选择：

1. 我们可以通过DCN复制，然后通过PCIe将整个未分片数组加载到HBM中。
2. 我们可以将分片数组加载到它们相应的TPU上，然后通过ICI执行收集，然后在TPU{0,0}上执行矩阵乘法。

显然选项（2）更好。DCN与ICI相比很慢，我们更愿意通过许多PCIe链接而不是只有几个（主机0上的8个）加载大数组。这是系统一部分的图表。如上所述，请注意TPU通过ICI连接到它们的邻居（即使跨主机），所有TPU都通过PCIe连接到它们的主机CPU，主机通过DCN连接。

{% include figure.liquid path="assets/img/challenge-problem.png" class="img-fluid img-small" caption="每个芯片实际上都有自己的PCIe链接到其主机，尽管为了清晰起见，这里只显示了一个。" %}

现在让我们计算每个部分需要多长时间：

1. **PCIe加载**：我们通过16个PCIe链接加载16GB / 2 = 8GB的块，每个链接的带宽为`1.5e10`字节/秒。因此这大约需要33毫秒。

2. **ICI复制：** 每个TPU现在有我们数组的16GB / 16 = 1GB。我们的ICI带宽是每个链接*双向*9e10字节/秒，您会从上面的图表中注意到，TPU v5e上的4个ICI链接中只有2个在TPU{0,0}的此拓扑中使用。由于TPU{0,0}需要沿2个轴以`4.5e10`字节/秒/链接接收总共15GB，我们可以通过`15e9 / (4.5e10 * 2) = 167ms`来下限时间。实际上这可能无法实现，因为负载非常不均匀，但它可能在2倍以内。正如您将在第2节中看到的，执行完整的AllGather也将大约需要`16e9 / (4.5e10 * 2)`，所以这接近最优。

3. **HBM $\rightarrow$ MXU加载：** 要执行我们的最终矩阵乘法，我们需要将这16e9字节加上bf16[8, 128 \* 1024]数组（另外2MB，因此可以忽略不计）通过HBM带宽加载到MXU中，这将需要`16e9 / 8.1e11 = 19ms`。

4. **FLOPs：** 我们执行总共$$2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7e11$$ FLOPs，由于我们可以执行`1.97e14` bf16 FLOPs/s，我们得到1.3ms。

总时间的上界是所有这些时间的总和，但由于TPU通常可以重叠这些操作，我们可以将其视为由最慢部分瓶颈的流水线问题。假设这是真的，那么答案大约是150-200毫秒。

{% enddetails %}

<h3 markdown=1 class="next-section">第2部分到此结束！第3部分涵盖分区和跨TPU通信，[点击这里](../sharding)。</h3>

## 附录

### 附录A：更多TPU内部结构

在这里，我们将更深入地探讨TPU的内部操作。除非另有说明，我们将为TPU v5p提供规格。

### VPU

VPU是TPU的向量算术核心。VPU由执行逐元素算术运算（如vadd（向量加法）或vmax（逐元素最大值））的二维SIMD向量机（**VPU**）和一组称为**VREG**的向量寄存器组成，这些寄存器为VPU和MXU保存数据。

**VREG：** 每个TPU v5p核心有64个32位VREG（TPU v4中为32个），给我们每个核心总共约`64 * 8 * 128 * 4 = 256kB`的VREG内存（或整个芯片的2倍，因为我们有两个核心）。TPU v5p每个周期可以从VMEM加载3个寄存器，每个周期可以向VMEM写入1个寄存器。

**VPU：** VPU是形状为`(8, 128)`的2D向量算术单元，其中128维称为通道轴，8维称为子通道轴。v5上的每个（通道，子通道）对包含4个标准浮点ALU，它们彼此独立。VPU在其每个ALU中以一个周期执行大多数算术指令（如vadd或向量加法），延迟为2个周期，因此例如在v5中，您可以在每个周期中从VREG将4对f32值加在一起。典型的VPU指令可能看起来像`{v2 = vadd.8x128.f32 v0, v1}`，其中v0和v1是输入VREG，v2是输出VREG。

所有通道和子通道每个周期都以纯SIMD方式执行相同的程序，但每个ALU可以执行不同的操作。因此，我们可以例如在单个周期中处理1个vadd和1个vsub，每个操作都在两个完整的VREG上操作并将输出写入第三个。

**小测验 [计算VPU吞吐量]：** 使用上述信息，计算TPU v5p可以执行多少向量FLOPs/s。TPU v5p的时钟速度约为1.75GHz。

*答案*：每个周期，每个核心可以在`8 * 128`个ALU上执行4个向量指令。这给我们整个芯片每个周期`8 * 128 * 4 * 2` FLOPs，或`8 * 128 * 4 * 2 * 1.75e9 = 1.4e13` FLOPs/s。请注意这比MXU FLOPs/s约`2e14`小得多（大约10倍）。

**归约：** 通常，跨子通道维度的通信或归约比跨通道维度更容易。例如，VPU支持通道内洗牌操作，可以在约一个周期内沿大小为8的轴滚动。这可以用于沿子通道维度执行高效的归约（只需洗牌2、4和6，并执行3对逐元素求和）。

跨通道归约要困难得多，涉及一个称为XLU或"跨通道单元"的独立硬件单元，该单元速度慢且相当昂贵。

**与GPU的比较：** 对于熟悉NVIDIA GPU的人，VPU中的每个ALU类似于CUDA核心，单个VPU通道类似于"Warp调度器"，即执行SIMD算术的通常32个CUDA核心的集合。通道内的归约相当容易，但如果我们需要跨通道，我们至少需要传输VMEM/XLU/SMEM，这要慢得多。

### 标量核心

标量核心是TPU的控制单元。它获取并分派所有指令并执行从HBM到VMEM的传输，并且可以编程为执行标量元数据工作。由于标量核心是单线程的，这的一个副作用是TPU的每个核心每个周期只能创建一个DMA请求。

要将其置于上下文中，单个标量核心控制VPU（由4096个ALU组成）、4个MXU、2个XLU和多个DMA引擎。每单位计算的控制高度倾斜的性质是硬件效率的来源，但也限制了以任何有趣的方式进行数据相关矢量化的能力。

### 附录B：关于GPU的一切

自Volta代（V100）以来，TPU和GPU开始看起来非常相似：_它们都旨在非常快速地进行矩阵乘法_。它们都充当连接到CPU的加速器，许多组件大致类似（如果您不知道所有术语，请不要担心，我们稍后将介绍它们）：

|     TPU     |                    GPU                    |
| :---------: | :---------------------------------------: |
| Tensor Core |        SM（"流式多处理器"）         |
|     HBM     |                   DRAM                    |
|    VMEM     |      SMEM（通常用作L1缓存）       |
|     VPU     | Warp调度器（一组SIMD CUDA核心） |
|     MXU     |               Tensor Core                |
|     ICI     |              NVLink/NVSwitch              |

GPU的核心单元是SM或"流式多处理器"，大致类似于上述整个TPU Tensor Core。不过，与TPU相比，GPU有_更多_的SM（H100大约有144个）。每个SM都有自己的矩阵乘法单元，令人困惑地称为Tensor Core，它的作用类似于TPU MXU，以及一组称为Warp调度器的4个窄SIMD单元，它们的作用类似于TPU VPU（具有32个通道而不是1024个）。更多独立的SM使计算更灵活（因为每个都可以执行完全独立的工作），但也使硬件更昂贵且更难推理。

{% include figure.liquid path="assets/img/b100-sm-diagram.png" class="img-small" caption="<b>图：</b>Blackwell（B100）SM的基本组件。该图显示了4个SIMD计算单元（我们称之为warp调度器），每个都有一个用于矩阵乘法的Tensor Core。这还显示了每个warp调度器的寄存器、SM级L1缓存和TMEM或张量内存，这是Blackwell中的新增功能。" %}

每个SM还有一个O(256kB)的L1缓存（也称为SMEM），用于加速数据访问和寄存器溢出。用于L1缓存的内存部分也可以声明为共享内存，允许从线程块中的任何线程访问，并用于用户定义的缓存、并行归约和同步等（类似于TPU上的VMEM）。

GPU还有一个由所有SM共享的额外L2缓存。与VMEM不同，这是硬件管理的，优化缓存命中通常对性能很重要。

**网络：**

* 主要区别在于NVIDIA GPU通常通过开关（NVLink $\rightarrow$ NVSwitch）以8-256个GPU的"派系"形式存在，这允许该"派系"内任何GPU之间的点对点通信，但这意味着超过256个之间的通信明显较慢 - 这意味着在超过256个上进行训练通常需要流水线并行来扩展，这更复杂（相比之下，PaLM在两个各3072个TPU芯片的派系上进行训练）。
* 对于常见的神经网络操作（如AllReduce），全对全连接没有优势（因为无论如何都必须发生相同的通信模式），但它确实允许在更多GPU上存储MoE模型并更有效地传输专家。
* 每个GPU都需要一个成本与GPU本身相似的开关，使得像ICI这样的片上互连更便宜。
* [NVIDIA深度学习性能](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-arch)
* [NVSwitch](https://www.nvidia.com/en-au/data-center/nvlink/)
* 非常不同的张量并行/流水线并行转换点！

### 附录C：脉动阵列如何工作？

TPU MXU的核心是`128x128`脉动阵列（TPU v6e上为`256x256`）。当完全饱和时，脉动阵列可以每8个时钟周期执行一次`bfloat16[8,128] @ bf16[128x128] -> f32[8,128]`<d-footnote>如果您不熟悉这种表示法，它的意思是：将具有bfloat16元素的`8x128`矩阵乘以具有bfloat16元素的`128x128`矩阵，并将结果存储在具有float32元素的`8x128`矩阵中。</d-footnote>乘法。

* 在其核心，脉动阵列是一个2D `128x128`（`=16,384`）ALU网格，每个都能够执行乘法和加法操作。
* 权重（**W**，`128x128`输入）从上方传递下来（称为RHS），而输入（**X**，`8x128`输入）从左侧传入（称为LHS）。

这是一个简化的动画，展示了一组权重（蓝色）与一组激活（绿色）的乘法。您会注意到权重（RHS）首先部分加载，对角线地，然后激活被输入，也是对角线地。在下面的每一帧中，我们将所有重叠的绿色和蓝色单元相乘，将结果与从上方传入的任何残差相加，然后将结果依次向下传递一个单元。

{% include figure.liquid path="assets/img/systolic-array.gif" %}

这是这个动画的更通用版本，显示了从计算中流出的输出：

{% include figure.liquid path="assets/img/systolic-array2.gif" class="img-small" %}

这是一个图表，显示了如何在多个RHS和LHS数组之间进行流水线处理：

{% include figure.liquid path="assets/img/systolic-array-pipelining.png" class="img-fluid" %}

当权重（RHS）和激活（LHS）被加载时，有一个初始的流水线气泡。在初始气泡之后，可以在没有额外气泡的情况下加载新的输入和权重。

这是bf16[2, 3] x bf16[3, 3]矩阵乘法的一个不太好的动画，您可以将其想象为2x3权重矩阵与批次1和大小3的输入激活的矩阵乘法。与之前的幻灯片相比，这是旋转的，输入向右而不是向下流出，但您可以大致看到结构。

{% include figure.liquid path="assets/img/systolic-array-bad.gif" class="img-small" %}

我们可以有效地将其流水线化以乘以大矩阵，而不会有太大的流水线气泡。话虽如此，重要的是我们的矩阵的形状要大于MXU的侧面维度，通常为128x128。一些TPU（自TPU v3以来）有多个MXU，TPU v3为2个，TPU v4/5为4个，因此我们需要确保平铺维度大于128 * MXU数量。[这里](https://www.youtube.com/watch?v=sJltBQ4MOHA)有一个很好的动画。

Trillium（TPU v6e）有一个`256x256`脉动阵列，这意味着它每个周期可以执行4倍的FLOPs。这也意味着您的张量维度需要是两倍大才能完全利用MXU。

[这篇博客文章](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu)有另一个关于固定权重矩阵的脉动阵列乘法的出色动画。