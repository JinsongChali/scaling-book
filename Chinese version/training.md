---
layout: distill
title: "如何并行化Transformer进行训练"
# permalink: /main/
description: "这里我们讨论大语言模型训练期间使用的四种主要并行方案：数据并行、完全分片数据并行（FSDP）、张量并行和流水线并行。对于每种方案，我们计算在什么时候我们会被通信瓶颈。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 5

previous_section_url: "transformers"
previous_section_name: "第4部分：Transformer"

next_section_url: applied-training
next_section_name: "第6部分：训练LLaMA"

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
  - name: "扩展意味着什么？"
  - subsections:
    - name: "数据并行"
    - name: "完全分片数据并行（FSDP）"
    - name: "张量并行"
    - name: "混合FSDP和张量并行"
    - name: "流水线"
    - name: "Pod之间的扩展"
  - name: "TPU上大语言模型训练的要点"
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

## 扩展意味着什么？

"模型扩展"的目标是能够增加用于训练或推理的芯片数量，同时实现成比例的线性吞吐量增长（我们称之为*强扩展*）。虽然单个芯片上的性能取决于内存带宽和FLOPs之间的权衡，但集群级别的性能取决于通过将芯片间通信与有用的FLOPS重叠来隐藏它。这并非易事，因为增加芯片数量会增加通信负载，同时减少我们可以用来隐藏它的每设备计算量。正如我们在[第3节](../sharding)中看到的，分片矩阵乘法通常需要昂贵的AllGather或ReduceScatter，这可能会阻止TPU执行有用的工作。本节的目标是找出这些何时变得*太昂贵*。

在本节中，我们将讨论四种常见的并行方案：（纯）**数据并行、完全分片数据并行**（FSDP / ZeRO分片）、**张量并行**（也称为模型并行）和（简要地）**流水线并行**。对于每种方案，我们将展示我们产生的通信成本以及该成本在什么时候开始瓶颈我们的计算成本。<d-footnote>我们将专注于通信界限——因为虽然内存容量约束很重要，但在预训练期间使用重新物化（激活检查点）和大量芯片时，它们通常不会限制我们。我们也不在这里讨论MoE的专家并行——这大大扩展了设计空间，只讨论密集Transformer的基本情况。</d-footnote> 对于本节，您可以仅关注芯片间通信成本，因为只要我们有足够大的单芯片批量大小，从HBM到MXU的数据传输就已经与计算重叠了。

我们将在本节中使用以下符号来简化计算。

| 符号 | 含义（模型参数）                                           |
| :--- | :-------------------------------------------------------- |
| D    | **d**<sub>model</sub>（隐藏维度/残差流维度）              |
| F    | **d**<sub>ff</sub>（前馈维度）                            |
| B    | 批次维度（批次中的token数量；总计，不是每设备）            |
| T    | 序列长度                                                   |
| L    | 模型中的层数                                               |

| 符号 | 含义（硬件特性）                                                                                       |
| :--- | :---------------------------------------------------------------------------------------------------- |
| C    | 每芯片FLOPS/s                                                                                        |
| W    | 网络带宽（双向，通常下标为例如$W_{\text{ici}}$或$W_{\text{dcn}}$                                      |
| X    | 沿网格轴X的芯片数量                                                                                   |
| Y    | 沿备用网格轴的芯片数量，标记为Y                                                                       |
| Z    | 沿第三个网格轴的芯片数量，标记为Z                                                                     |

为简单起见，**我们将Transformer近似为MLP块的堆栈**——正如我们在[第4节](../transformers)中看到的，对于较大的模型，注意力是FLOPs的相对较小部分。我们还将忽略门控矩阵乘法，给我们留下每层的以下简单结构：

{% include figure.liquid path="assets/img/simple-transformer.png" class="img-fluid" caption="<b>图：</b>简化的Transformer层。我们将每个FFW块视为两个矩阵的堆栈<b>W<sub>in</sub></b>：<code>bf16[D, F]</code>（上投影）和<b>W<sub>out</sub></b>：<code>bf16[F, D]</code>（下投影），输入为<b>In</b>：<code>bf16[B, D]</code>。" %}

以下是我们将讨论的4种并行方案。每种方案都可以被认为是由上图中**In**、**W<sub>in</sub>、W<sub>out</sub>和Out**的分片唯一定义的。

**1. 数据并行：** *激活沿批次分片，参数和优化器状态在每个设备上复制。通信仅在反向传递期间发生。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

**2. 完全分片数据并行（FSDP或ZeRO-3）：** *激活沿批次分片（如纯数据并行），参数沿同一网格轴分片并在前向传递中使用前及时AllGather。优化器状态也沿批次分片。减少重复内存。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

**3. 张量并行（也称为Megatron分片或模型并行）：** *激活沿D（$d_\text{model}$）分片，参数沿F（$d_{ff}$）分片。在每个块之前和之后AllGather和ReduceScatter激活。与FSDP兼容。*

$$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$

**4. 流水线并行：** *权重沿层维度分片，激活微批处理并沿层维度滚动。流水线阶段之间的通信最小（只是在单跳上移动激活）。滥用符号：*

$$\text{In}[L_Z, B, D][i] \cdot_D W_\text{in}[L_Z, D, F][i] \cdot_F W_\text{out}[L_Z, F, D][i] \rightarrow \text{Out}[L_Z, B, D_Y][i]$$

### 数据并行

**语法：** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

**当您的模型能够以即使是很小的批量大小（>240个token，以便受计算限制）装入单个芯片时，您应该始终使用简单的数据并行。** 纯数据并行将我们的激活分割到任意数量的TPU上，只要TPU的数量小于我们的批量大小。前向传递不涉及通信，但在每个步骤结束时，每个都对其梯度执行**AllReduce以便在更新参数之前同步它们。**

{% include figure.liquid path="assets/img/data-parallelism.png" class="img-fluid" caption="<b>图：</b>纯数据并行图（前向传递）。我们的激活（左）沿批次维度完全分片，我们的权重完全复制，因此每个TPU都有权重的相同副本。这意味着我们权重的总内存增加了N倍，但前向传递不需要通信。" %}

{% details 这里是前向和反向传递的完整算法。我们滥用符号将dL/dOut写为dOut，纯粹是为了紧凑。 %}

<div markdown=1 class="algorithm">

**纯数据并行算法：**

**前向传递：** 需要计算Loss[B<sub>X</sub>]

1.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B<sub>X</sub>] = ...

**反向传递：** 需要计算dW<sub>out</sub>[F, D]、dW<sub>in</sub>[D, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D] = **AllReduce**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*不在关键路径上，可以异步完成*)
4.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D]
5.  dW<sub>in</sub>[D, F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
6.  dW<sub>in</sub>[D, F] = **AllReduce**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) (*不在关键路径上，可以异步完成*)
7.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*前面层需要*)

</div>

我们忽略损失函数的细节并缩写$\text{Tmp} = W_\text{in} \cdot \text{In}$。请注意，尽管我们的最终损失是平均**AllReduce**(Loss[B<sub>X</sub>])，我们只需要在平均权重梯度时在反向传递中计算AllReduce。

{% enddetails %}

请注意，前向传递没有通信——**都在反向传递中**！反向传递还有一个很好的属性，即AllReduce不在"关键路径"中，这意味着每个AllReduce可以在方便时执行，不会阻止您执行后续操作。我们将看到模型/张量并行没有这个属性。

**为什么这样做？** 纯数据并行通过在批次维度上分割我们的激活来减少激活内存压力，允许我们几乎任意增加批量大小，只要我们有更多芯片来分割批次维度。特别是在训练期间，当我们的激活通常主导我们的内存使用时，这非常有帮助。

**为什么不这样做？** 纯数据并行对减少模型参数或优化器状态的内存压力没有任何作用，这意味着纯数据并行对于规模上有趣的模型很少有用，因为我们的参数+优化器状态不适合单个TPU。为了给出规模感，如果我们使用Adam训练bf16参数和fp32优化器状态<d-footnote>Adam存储参数、一阶和二阶累加器。由于参数是bfloat16，优化器状态是float32，这给我们每个参数`2 + 8 = 10`字节。</d-footnote>，我们可以容纳的最大模型有$$\text{TPU内存} / 10$$个参数，例如在具有96GB HBM和纯数据并行的TPUv5p pod上，这大约是9B参数。

<p markdown=1 class="takeaway">**要点**：我们可以使用Adam和纯数据并行训练的最大模型有$$\text{num_params} = \text{每设备HBM} / 10$$。对于TPU v5p，这大约是9B参数。<d-footnote>请注意，这不包括梯度检查点，所以这实际上不会有用。这是批次为1个token的绝对下限。</d-footnote></p>

*为了使这对训练期间的真实模型有用，我们至少需要部分分片模型参数或优化器。*

**我们什么时候被通信瓶颈？** 如上所示，我们每层有两个AllReduce，每个大小为$$2DF$$（对于bf16权重）。数据并行何时使我们受通信限制？

如上表所示，设$C$ = 每芯片FLOPs，$W_{\text{ici}}$ = **双向**网络带宽，$X$ = 批次分区的分片数<d-footnote>我们假设这种分区是在ICI网格上完成的，因此相关的网络带宽是$W_\text{ici}$</d-footnote>。让我们计算执行相关矩阵乘法所需的时间$$T_\text{math}$$和所需的通信时间$$T_\text{comms}$$。由于这种并行方案在前向传递中不需要通信，我们只需要为反向传递计算这些量。

*通信时间：* 从前面的部分我们知道，在1D网格中执行AllReduce所需的时间仅取决于被AllReduce的数组的总字节数和ICI带宽$W_\text{ici}$；具体来说，AllReduce时间是$2 \cdot \text{总字节数} / W_\text{ici}$。由于我们需要为$W_\text{in}$和$W_\text{out}$进行AllReduce，我们每层有2个AllReduce。每个AllReduce用于权重矩阵，即$DF$参数的数组，或$2DF$字节。将所有这些放在一起，单层中AllReduce的总时间是

$$\begin{align}
T_\text{comms} &= \frac{2 \cdot 2 \cdot 2 \cdot D \cdot F}{W_\text{ici}}. \\
\end{align}$$

*矩阵乘法时间：* 每层在前向传递中包含两个矩阵乘法，或在反向传递中包含四个矩阵乘法，每个需要$2(B/X)DF$ FLOPs。因此，对于反向传递中的单层，我们有

$$\begin{align}
T_\text{math} &= \frac{2 \cdot 2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
\end{align}$$

由于我们重叠，每层的总时间是这两个量的最大值：

$$\begin{aligned}
T &\approx \max(\frac{8 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{8 \cdot D \cdot F}{W_\text{ici}}) \\
T &\approx 8 \cdot D \cdot F \cdot \max(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}})
\end{aligned}$$

当$$T_\text{math}/T_\text{comms} > 1$$时我们变为计算受限，或者当

$$\begin{align}
\frac{B}{X} > \frac{C}{W_\text{ici}}.
\end{align}$$

结果是，要使用数据并行保持计算受限，我们需要每设备批量大小$$B / X$$超过ICI操作强度$C / W_\text{ici}$。这最终是计算时间随每设备批量大小缩放，而通信时间独立于此量（因为我们正在传输模型权重）的结果。请注意$B > C/W_\text{ici}$条件与单设备计算受限规则$B > 240$的相似性；在这种情况下，规则也来自计算时间随批量大小缩放，而数据传输大小（在$B \ll F, D$状态下）独立于批量大小的事实。

让我们输入一些实际数字来了解规模。对于TPUv5p，`C=4.6e14`和`W=2 * 9e10`用于ICI上的1D数据并行，所以**我们每个芯片的批量大小必须至少为2,550以避免受通信限制**。由于我们可以在多个轴上进行数据并行，如果我们将TPUv5p pod的所有三个轴都专用于纯数据并行，我们将带宽$W_\text{ici}$增加3倍，可以缩小到每个TPU只有BS=850或每个pod（8960个芯片）每批760万个token！**这告诉我们，很难被纯数据并行瓶颈！**

<p markdown=1 class="takeaway">**关于上下文并行的说明：** 在本节中，我们使用$B$来指代以token为单位的总批量大小。然而，很明显，我们的批次由$K$个序列组成，每个序列有$T$个token，那么我们如何做到这一点？就MLP而言，*token就是token*！它们属于同一批次还是两个不同批次并不重要。因此，我们或多或少可以自由地在批次和序列维度上进行数据并行：我们称之为上下文并行或序列并行，但您可以将其视为另一种数据并行。注意力比MLP更棘手，因为我们进行一些跨序列计算，但这可以通过在注意力期间收集KV或Q并仔细重叠FLOPs和通信（通常使用称为"环注意力"的东西）来处理。在本节中，我们将完全忽略我们的序列维度，并假设一定量的批次或序列并行。</p>


### 完全分片数据并行（FSDP）

**语法：** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

完全分片数据并行（通常称为FSDP或ZeRO分片<d-cite key="zero"></d-cite>）将模型优化器状态和权重分割到数据并行分片上，并根据需要有效地收集和分散它们。**与纯数据并行相比，FSDP大大减少了每设备内存使用，并在反向传递FLOPs上节省，开销非常小。**

{% include figure.liquid path="assets/img/fsdp.png" class="img-fluid" caption="<b>图：</b>FSDP沿数据维度分片Win的收缩维度和Wout的输出维度。这减少了内存，但（从第3节）要求我们在执行矩阵乘法之前为W收集权重。请注意，激活（左）<it>没有沿收缩维度分片</it>，这就是迫使我们收集的原因。<b>请注意，我们的权重优化器状态同样沿收缩维度分片。</b>" %}

您会记得（从[第3节](../sharding)）AllReduce可以分解为AllGather和ReduceScatter。这意味着，我们可以在芯片上分片权重和优化器状态，在前向传递期间在每层AllGather它们，在反向传递期间跨权重ReduceScatter，而不是为标准数据并行执行完整的梯度AllReduce，无需额外成本。

{% details 这里是FSDP的完整算法。 %}

<div markdown=1 class="algorithm">

**完全分片数据并行（FSDP）：**

**前向传递：** 需要计算Loss[B<sub>X</sub>]

1.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*不在关键路径上，可以在前一层期间执行*)
2.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F] (*现在可以丢弃W<sub>in</sub>[D, F]*)
3.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*不在关键路径上，可以在前一层期间执行*)
4.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
5.  Loss[B<sub>X</sub>] = ...

**反向传递：** 需要计算dW<sub>out</sub>[F, D<sub>X</sub>]、dW<sub>in</sub>[D<sub>X</sub>, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D<sub>X</sub>] = **ReduceScatter**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*不在关键路径上，可以异步完成*)
4.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*可以提前完成*)
5.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D] *(可以在这里丢弃W<sub>out</sub>[F, D])*
6.  dW<sub>in</sub>[D,F] {U<sub>X</sub>} = dTmp[B<sub>X</sub>, F] \*<sub>B</sub> In[B<sub>X</sub>, D]
7.  dW<sub>in</sub>[D<sub>X</sub>, F] = **ReduceScatter**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) *(不在关键路径上，可以异步完成)*
8.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*可以提前完成*)
9.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*前面层需要) (可以在这里丢弃W<sub>in</sub>[D, F]*)

</div>

{% enddetails %}

这也称为"ZeRO分片"，来自"零开销分片"，因为我们不执行任何不必要的计算或存储任何不必要的状态。ZeRO-{1,2,3}用于分别指以这种方式分片优化器状态、梯度和权重。由于所有都有相同的通信成本<d-footnote>技术上，FSDP在前向传递中添加了纯DP没有的通信，但这与反向传递的比例相同，因此对通信roofline应该没有影响。这里的关键是ZeRO-3将反向传递AllReduce转换为AllGather和ReduceScatter，它们具有相同的总通信量。</d-footnote>，我们基本上总是可以进行ZeRO-3分片，它将参数、梯度和优化器状态分片到一组设备上。

**我们为什么要这样做？** 标准数据并行涉及大量重复工作。每个TPU AllReduce完整梯度，然后更新完整优化器状态（所有TPU上的相同工作），然后更新参数（再次，完全重复）。对于ZeRO分片（分片梯度/优化器状态），您可以ReduceScatter梯度，而不是AllReduce，仅更新优化器状态的分片，更新参数的分片，然后根据前向传递的需要AllGather参数。

**我们什么时候被通信瓶颈？** 我们的相对FLOPs和通信成本与纯数据并行完全相同，因为反向传递中的每个AllReduce都变成了AllGather + ReduceScatter。回想一下，AllReduce实现为AllGather和ReduceScatter，每个都有一半的成本。这里我们对前向传递进行建模，因为它具有与反向传递相同的FLOPs与通信比率：

$$\begin{aligned}
T_{math} &= \frac{2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
T_{comm} &= \frac{2 \cdot 2 \cdot D \cdot F}{W_\text{ici}} \\
T &\approx \max\left(\frac{4 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{4 \cdot D \cdot F}{W_\text{ici}}\right) \\
T &\approx 4 \cdot D \cdot F \cdot \max\left(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}}\right)
\end{aligned}$$

因此，与纯数据并行一样，当$$B / X > C / W_\text{ici}$$时我们受计算限制，即当每设备批量大小$B/X$超过"ICI操作强度"$C/W_\text{ici}$（对于v5p为`4.59e14 / 1.8e11 = 2550`）。这对我们来说很好，因为这意味着如果我们的每设备批量大小足够大以受纯数据并行的计算限制，我们可以——不用担心离开计算受限状态——简单地升级到FSDP，为自己节省大量参数和优化器状态内存！虽然我们确实必须向前向传递添加通信，但这个成本是无关紧要的，因为它只是与前向传递FLOPs重叠。

<p markdown=1 class="takeaway">**要点：** FSDP和纯数据并行在TPUv5上当每设备批量大小小于$2550 / n_\text{axes}$时都变为带宽受限。</p>

例如，DeepSeek-V2（最近发布训练批量大小信息的少数强大模型之一）使用了约40M token的批量大小。**这将允许我们扩展到大约47,000个芯片，或大约5个TPUv5 pod，然后达到带宽限制。**

对于LLaMA-3 70B，它训练了大约`6.3e24 (15e12 * 70e9 * 6)` FLOPs，我们可以将16M token的批次分割到大约`16e6 / (2550 / 3) = 18,823`个芯片（大约2个8960芯片的pod），每个具有`4.59e14` FLOPs，以50%的峰值FLOPs利用率（通常称为MFU）运行，**在大约17天内训练它**。不错！但让我们探索如何做得更好。

<p markdown=1 class="takeaway">**关于关键批量大小的说明**：有点反直觉的是，随着我们的总批量大小减少（固定芯片数），我们变得更受通信瓶颈。数据并行和FSDP让我们扩展到任意多的芯片，只要我们能继续增加批量大小！然而，在实践中，随着批量大小的增加，我们往往会看到训练的收益递减，因为我们的梯度变得几乎无噪声。我们有时也会看到训练不稳定。因此，在"无限计算状态"中找到最佳分片方案的游戏通常从固定的批量大小（由缩放定律确定）和已知的（大）芯片数量开始，然后旨在找到允许我们在这么多芯片上容纳该小批量大小的分区。</p>

### 张量并行

**语法：** $$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$（我们使用$$Y$$最终与FSDP结合）

在完全分片的数据并行AllReduce中，我们跨芯片移动权重。我们还可以分片模型的前馈维度并在层期间移动激活——这称为"1D模型并行"或Megatron分片<d-cite key="megatron"></d-cite>。这可以解锁每个pod更小的有效批量大小。下图显示了以这种方式分片的单个矩阵的示例：

{% include figure.liquid path="assets/img/model-parallelism.png" class="img-fluid" caption="<b>图：</b>基本张量并行的示例。由于我们只在Y上分片我们的激活（与在X上分片的FSDP不同），我们在X上复制我们的激活。使用我们的标准语法，这是<b>A</b>[B, D<sub>Y</sub>] * <b>B</b>[D, F<sub>Y</sub>] -> <b>C</b>[B, F<sub>Y</sub>]。因为我们只在收缩维度之一上分片，我们通常在矩阵乘法之前AllGather激活<b>A</b>。" %}

如前所述，**In\[B, D<sub>Y</sub>\] \*<sub>D</sub> W<sub>in</sub>\[D, F<sub>Y</sub>\] \*<sub>F</sub> W<sub>out</sub>\[F<sub>Y</sub>, D\] \-\> Out\[B, D<sub>Y</sub>\]意味着我们必须在第一个矩阵乘法之前收集我们的激活。当激活小于权重时，这比ZeRO分片便宜。** 这通常只有在添加一些ZeRO分片（减少收集的大小）时才是真的。这是我们倾向于混合ZeRO分片和模型并行的原因之一。

{% details 这里是张量并行的算法！ %}

<div markdown=1 class="algorithm">

**张量并行：**

**前向传递：** 需要计算Loss[B]

1.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *(在关键路径上)*
2.  Tmp[B, F<sub>Y</sub>] = In[B, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(未沿收缩分片，因此无通信)*
3.  Out[B, D] {U<sub>Y</sub>} = Tmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
4.  Out[B, D<sub>Y</sub>] = **ReduceScatter**(Out[B, D] {U<sub>Y</sub>}) *(在关键路径上)*
5.  Loss[B] = ...

**反向传递：** 需要计算dW<sub>out</sub>[F<sub>Y</sub>, D]、dW<sub>in</sub>[D, F<sub>Y</sub>]

1.  dOut[B, D<sub>Y</sub>] = ...
2.  dOut[B, D] = **AllGather**(dOut[B, D<sub>Y</sub>]) *(在关键路径上)*
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] = Tmp[B, F<sub>Y</sub>] \*<sub>B</sub> dOut[B, D]
4.  dTmp[B, F<sub>Y</sub>] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *(可以在这里丢弃dOut[B, D])*
5.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *(这可以通过与前向传递的(1)共享来跳过)*
6.  dW<sub>in</sub>[D, F<sub>Y</sub>] = dTmp[B, F<sub>Y</sub>] \*<sub>B</sub> In[B, D]
7.  dIn[B, D] {U.Y} = dTmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(前面层需要)*
8.  dIn[B, D<sub>Y</sub>] = **ReduceScatter**(dIn[B, D] {U.Y}) *(在关键路径上)*

</div>

{% enddetails %}

张量并行的一个好处是它与我们的Transformer前向传递中的两个矩阵很好地交互。天真地，我们会在两个矩阵中的每一个之后执行AllReduce。但这里我们首先执行**In[B, D<sub>Y</sub>] \* W<sub>in</sub>[D, F<sub>Y</sub>] -> Tmp[B, F<sub>Y</sub>]**，然后**Tmp[B, F<sub>Y</sub>] \* W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B, D<sub>Y</sub>]**。这意味着我们在开始时AllGather **In**，在结束时ReduceScatter **Out**，而不是执行AllReduce。

**这有多昂贵？** 让我们只对前向传递建模——反向传递只是这里每个操作的转置。在1D模型并行中，我们在第一个矩阵乘法之前AllGather激活，在第二个之后ReduceScatter它们，一次发送两个字节（bf16）。让我们弄清楚我们什么时候被通信瓶颈。

$$\begin{align}
T_{math} & = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} \\
T_{comms} & =
\frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\\
\textnormal{T} & \approx \max \left(\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C}, \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\right)
\end{align}$$

注意到我们希望计算成本大于通信成本，我们得到：

$$\begin{align}
\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} > \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}
\end{align}$$

$$\begin{align}
\frac{F}{Y \cdot C} > \frac{1}{W_\text{ici}}
\end{align}$$

$$\begin{align}
F > Y \cdot \frac{C}{W_\text{ici}}
\end{align}$$

因此，例如，对于TPUv5p，$$C / W_{ici} = 2550$$在bf16中，所以我们只能进行张量并行，直到$$Y < F / 2550$$。当我们有多个ICI轴时，我们的$$T_\text{comms}$$减少了$n_\text{axes}$倍，所以我们得到$$Y < n_\text{axes} * F / 2550$$。

<p markdown=1 class="takeaway">**要点**：当$$Y > n_\text{axes} * F / 2550$$时，模型并行变为通信受限。对于大多数模型，这在8到16路模型并行之间。</p>

**请注意，这不依赖于计算的精度**，因为例如对于int8，在TPUv5p上，$$C_\text{int8} / W_{ici}$$是$$5100$$而不是$$2550$$，但通信量也减半，所以两个二的因子抵消了。

**让我们考虑一些例子：**

* 在TPUv4p上使用LLaMA 3-70B，$$D = 8192,$$ $$F \approx 30,000$$，我们可以舒适地进行8路模型并行，但在16路模型并行时将受通信限制。模型8路分片所需的F是20k。

* 对于Gemma 7B，$$F \approx 50k$$，所以我们在19路模型并行时变为通信受限。这意味着我们可能可以进行16路并仍然看到良好的性能。

### 混合FSDP和张量并行

**语法：** $$\text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]$$

FSDP和张量并行的好处是它们可以结合。通过沿两个轴分片**W<sub>in</sub>**和**W<sub>out</sub>**，我们既节省内存又节省计算。因为我们沿X分片B，我们减少了模型并行AllGather的大小，因为我们沿Y分片F，我们减少了FSDP的通信开销。这意味着两者的组合可以让我们达到比上面看到的更低的有效批量大小。

{% include figure.liquid path="assets/img/mixed-fsdp-model-parallelism.png" class="img-fluid" caption="<b>图：</b>结合FSDP和张量并行的图表。与其他情况不同，没有模型参数的重复。" %}

{% details 这里是混合FSDP + 张量并行的完整算法。虽然我们有很多通信，但我们所有的AllGather和ReduceScatter都更小，因为我们批量分片了我们的激活并张量分片了我们的权重！ %}

<div markdown=1 class="algorithm">

**前向传递：** 需要计算Loss[B]

1.  In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *(在关键路径上)*
2.  W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *(可以提前完成)*
3.  Tmp[B<sub>X</sub>, F<sub>Y</sub>] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>]
4.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *(可以提前完成)*
5.  Out[B<sub>X</sub>, D] {U.Y} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
6.  Out[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(Out[B<sub>X</sub>, D] {U.Y}) *(在关键路径上)*
7.  Loss[B<sub>X</sub>] = ...

**反向传递：** 需要计算dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]、dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]

1.  dOut[B<sub>X</sub>, D<sub>Y</sub>] = ...
2.  dOut[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(dOut[B<sub>X</sub>, D<sub>Y</sub>]) *(在关键路径上)*
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] {U.X} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
4.  dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>out</sub>[F<sub>Y</sub>, D] {U.X})
5.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *(可以提前完成)*
6.  dTmp[B<sub>X</sub>, F<sub>Y</sub>] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *(可以在这里丢弃dOut[B, D])*
7. In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *(不在关键路径上 + 这可以与前一层的(2)共享)*
8.  dW<sub>in</sub>[D, F<sub>Y</sub>] {U.X} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> In[B<sub>X</sub>, D]
9.  dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>in</sub>[D, F<sub>Y</sub>] {U.X})
10. W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *(可以提前完成)*
11. dIn[B<sub>X</sub>, D] {U.Y} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(前面层需要)*
12. dIn[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(dIn[B<sub>X</sub>, D] {U.Y}) *(在关键路径上)*

</div>

{% enddetails %}

**FSDP和MP的正确组合是什么？** 一个简单但关键的格言是FSDP移动权重，模型并行移动激活。这意味着随着我们的批量大小缩小（特别是当我们进行更多数据并行时），模型并行变得更便宜，因为我们每个分片的激活更小。

* 模型并行执行$$\mathbf{AllGather}_Y([B_X, D_Y])$$，随着$$X$$增长而缩小。
* FSDP执行$$\mathbf{AllGather}_X([D_X, F_Y])$$，随着$$Y$$增长而缩小。

因此，通过结合两者，我们可以将每个副本的最小批量大小降得更低。我们可以以与上面相同的方式计算FSDP和MP的最佳量：

设$$X$$为专用于FSDP的芯片数量，$$Y$$为专用于张量并行的芯片数量。设$$N$$为我们切片中的芯片总数，$$N=XY$$。设$$M_X$$和$$M_Y$$为我们分别进行FSDP和MP的网格轴数（这些应该大致加起来为3）。我们将纯粹对前向传递建模，因为它每个FLOP的通信最多。然后将上面算法中的通信加起来，我们有

$$T_\text{FSDP comms}(B, X, Y) = \frac{2\cdot 2\cdot D \cdot F}{Y \cdot W_\text{ici} \cdot M_X}$$

$$T_\text{MP comms}(B, X, Y) = \frac{2 \cdot 2 \cdot B \cdot D}{X \cdot W_\text{ici} \cdot M_Y}$$

同样，我们的总FLOPs时间是

$$T_\text{math} = \frac{2\cdot 2 \cdot B \cdot D \cdot F}{N \cdot C}.$$

为了简化分析，我们做了两个简化：首先，我们允许$X$和$Y$取非整数值（只要它们是正数并满足$XY=N$）；其次，我们假设我们不在$X$和$Y$轴上重叠通信。在第二个假设下，总通信时间是

$$T_\text{comms} = T_\text{FSDP comms} + T_\text{MP comms}.$$


在我们询问在什么条件下我们将受计算限制之前，让我们找到$X$和$Y$的最佳值以最小化我们的总通信。由于我们的FLOPs独立于$X$和$Y$，最佳设置是那些简单地最小化通信的设置。为此，让我们用$X$和$N$（保持固定，因为它是我们系统中的芯片数）而不是$X$和$Y$来写上面的$T_\text{comms}$：

$$T_\text{comms} (X) = \frac{F \cdot X}{N \cdot M_X} + \frac{B}{X \cdot M_Y}$$

对这个表达式关于$X$求导并将导数设为零，得到最佳值$X_{opt}$：

$$\begin{align*}
\frac{d}{dX} T_\text{comms} (X_{opt}) = \frac{F}{N \cdot M_X} - \frac{B}{X_{opt}^2 \cdot M_Y} \rightarrow \\
X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}
\end{align*}$$

这非常有用！这告诉我们，对于给定的$B$、$F$和$N$，什么量的FSDP是最佳的。让我们了解一下规模。插入现实值，即$N = 64$（对应于4x4x4芯片阵列）、$B=48,000$、$F=32,768$，大约给出$X\approx 13.9$。所以我们会选择$X$为16，$Y$为4，接近我们计算的最佳值。

<p markdown=1 class="takeaway">**要点：** 一般来说，在训练期间，FSDP的最佳量是$$X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}$$。</p>

现在让我们回到我们对所有并行策略一直在问的问题：**在什么条件下我们将受计算限制？** 由于我们可以重叠FLOPs和通信，当

$$T_\text{FSDP comms} + T_\text{MP comms} < T_\text{math}$$

时我们受计算限制，这给我们

$$\frac{2\cdot 2\cdot D \cdot F}{Y \cdot W_\text{ici} \cdot M_X} + \frac{2 \cdot 2 \cdot B \cdot D}{X \cdot W_\text{ici} \cdot M_Y} < \frac{2\cdot 2 \cdot B \cdot D \cdot F}{N \cdot C}$$

设$\alpha \equiv C / W_\text{ici}$，ICI算术强度，我们可以简化：

$$\frac{F}{Y \cdot M_X} + \frac{B}{X \cdot M_Y} < \frac{B \cdot F}{N \cdot \alpha}$$

将我们计算的$X_{opt}$插入上面的方程（并注意$Y_{opt} = N/X_{opt}$）会导致批量大小$B$的以下条件：

$$ \sqrt{\frac{4 \cdot B\cdot F}{M_X \cdot M_Y \cdot N}} < \frac{B \cdot F}{N \cdot \alpha},$$

其中左侧与通信时间成比例，右侧与计算时间成比例。请注意，虽然计算时间与批量大小线性缩放（无论并行性如何），但通信时间与批量大小的平方根缩放。因此，计算与通信时间的比率也按批量大小的平方缩放：

$$ \frac{T_\text{math}}{T_\text{comms}} = \frac{\sqrt{BF}\sqrt{M_X M_Y}}{2\alpha \sqrt{N}}. $$

为了确保这个比率大于1，以便我们受计算限制，我们需要

$$ \frac{B}{N} > \frac{4\alpha^2}{M_X M_Y F}$$

有关此关系的替代推导，请参见附录C。要获得近似数字，再次插入$F=32,768$、$\alpha=2550$和$M_X M_Y=2$（对于3D网格必须如此）。这大约给出$B/N > 400$。与纯数据并行（或FSDP）情况相比，这大约赢得了两倍的因子，假设3D网格，我们计算$B/N$必须超过约$850$才能受计算限制。

<p markdown=1 class="takeaway">**要点：** 将张量并行与FSDP结合允许我们将$B/N$降至$$2 \cdot 2550^2 / F$$。这让我们处理每个芯片少至400的批次，这大约比我们仅使用FSDP可以实现的小两倍。</p>

下面我们绘制了混合FSDP + MP的FLOPs与通信时间的比率，将其与仅模型并行和仅数据并行（FSDP）进行比较，在代表性的4x4x4芯片阵列上。虽然纯FSDP并行在非常大的批量大小下占主导地位，但在批量大小超过芯片数量在大约400到850之间的状态下，需要混合FSDP + MP策略才能受计算限制。

{% include figure.liquid path="assets/img/mixed-fsdp-comms-2.png" class="img-fluid" caption="<b>图：</b>在TPUv5p 4x4x4切片上，F=30k时最佳混合FSDP/MP的FLOPs与通信时间的比率。如预期，模型并行与批量大小有固定的比率；理想的混合FSDP + MP按$\sqrt{B}$缩放，FSDP按$B$缩放。然而，在中等批量大小状态下，只有FSDP + MP实现了大于1的比率。"%}

这是TPU v5p 16x16x16的另一个示例，显示了不同分片方案的批量大小函数的FLOPs和通信时间。

{% include figure.liquid path="assets/img/comms-flops-time.png" class="img-fluid" caption="<b>图：</b>不同并行方案的通信时间。黑色虚线是矩阵乘法FLOPs所需的时间，因此任何高于此线的曲线都受通信限制。我们注意到所有策略在批量大小1.5e6以下都变为通信受限，这与我们预期的4096 * 2 * 2550^2 / (8192 * 4) = 1.6e6一致。" %}

黑色曲线是花在模型FLOPs上的时间，这意味着任何批量大小，如果这低于所有通信成本，则严格受通信限制。您会注意到黑色曲线在约`1.6e10`处与绿色曲线相交，如预测。

放大，我们可以看到，将两个轴专用于FSDP，并使用光学开关重新配置拓扑以具有8长轴用于模型分片，将在1M和6M批量大小之间给我们最低的通信量，而纯FSDP组合在6M和100M之间最好。这与我们上面的计算一致！

{% include figure.liquid path="assets/img/comms-flops-time-zoom.png" class="img-fluid" %}

这是一个互动动画来玩这个，显示不同批量大小的总计算时间和通信时间：

<div class="l-page">
  <iframe src="{{ 'assets/plotly/training-roofline.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

您会注意到这通常与上面一致（最小值在FSDP=256，MP=16附近），加上或减去一些摆动因子，以适应每个轴数的一些轻微差异。

### 流水线

您可能会注意到我们在前面的部分中完全避免了谈论流水线。流水线是GPU并行的主导策略，在TPU上不太重要。简而言之，流水线训练涉及将模型的层分割到多个设备上，并在前向和反向传递期间在流水线阶段之间传递激活。算法类似于：

1. 在TPU 0上初始化您的数据，您的权重沿层维度分片（$W_\text{in}[L_Z, D_X, F_Y]$用于具有FSDP和张量并行的流水线）。
2. 在TPU 0上执行第一层，然后将生成的激活复制到TPU 1，并重复直到到达最后一个TPU。
3. 计算损失函数及其导数$\partial L / \partial x_L$。
4. 对于最后一个流水线阶段，计算导数$\partial L / \partial W_L$和$\partial L / \partial x_{L-1}$，然后将$\partial L / \partial x_{L-1}$复制到前一个流水线阶段并重复直到到达TPU 0。

{% details 这里有一些（工作的）Python伪代码 %}

这个伪代码应该在Cloud TPU VM上运行。虽然它不是很有效或现实，但它让您了解数据如何在设备之间传播。

```python
batch_size = 32
d_model = 128
d_ff = 4 * d_model

num_layers = len(jax.devices())

key = jax.random.PRNGKey(0)

# 假设每层只是一个矩阵乘法。
x = jax.random.normal(key, (batch_size, d_model))
weights = jax.random.normal(key, (num_layers, d_model, d_model)) 

def layer_fn(x, weight):
  return x @ weight

# 假设我们有num_layers == num_pipeline_stages
intermediates = [x]
for i in range(num_layers):
  x = layer_fn(x, weights[i])
  intermediates.append(x)

  if i != num_layers - 1:
    x = jax.device_put(x, jax.devices()[i+1])

def loss_fn(batch):
  return jnp.mean(batch ** 2)  # 编造一些假损失函数

loss, dx = jax.value_and_grad(loss_fn)(x)

for i in range(0, num_layers, -1):
  _, f_vjp = jax.vjp(layer_fn, intermediates[i + 1], weights[i])
  dx, dw = f_vjp(dx)  # 计算jvp dx @ J(L)(x[i], W[i])
  weights[i] = weights[i] - 0.01 * dw  # 更新我们的权重

  if i != 0:
    dx = jax.device_put(dx, jax.devices()[i-1])
```

{% enddetails %}

**为什么这是个好主意？** 流水线很棒有很多原因：它在流水线阶段之间具有低通信成本，这意味着您即使使用低带宽互连也可以训练非常大的模型。这在GPU上通常非常有用，因为它们不像TPU那样通过ICI密集连接。

**为什么这很困难/烦人？** 您可能已经注意到上面的伪代码中TPU 0几乎总是空闲的！它只在流水线的第一步和最后一步上工作。空闲期称为流水线气泡，处理起来非常烦人。通常，我们首先尝试通过微批处理来缓解这个问题，这会通过流水线发送多个小批次，使TPU 0至少在总步骤时间的更大部分内被利用。

第二种方法是仔细重叠前向矩阵乘法$W_i @ x_i$、反向$dx$矩阵乘法$W_i @ \partial L / \partial x_{i+1}$和$dW$矩阵乘法$\partial L / \partial x_{i+1} @ x_i$。由于每个都需要一些FLOPs，我们可以重叠它们以完全隐藏气泡。这是最近DeepSeek v3论文<d-cite key="DeepSeek3"></d-cite>的图，显示了他们的"无气泡"流水线调度：

{% include figure.liquid path="assets/img/deepseek-pipeline.png" class="img-fluid" caption="<b>图：</b>DeepSeek v3流水线调度（来自他们的<a href=\"https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf\">最近论文</a>）。橙色是前向矩阵乘法，绿色是dL/dx矩阵乘法，蓝色是dL/dW矩阵乘法。通过优先考虑向后dL/dx乘法，我们可以避免"搁浅"FLOPs。" %}

因为它对TPU不太关键（TPU有更大的互连pod），我们不会深入探讨这一点，但理解关键的流水线瓶颈是一个很好的练习。

### Pod之间的扩展

让我们退一步看看一个具体的例子，比如在TPU v5p上训练LLaMA-3 70B。LLaMA-3 70B有$$F\approx 30,000$$。从上面的部分，我们知道以下内容：

* 当我们进行模型并行大于$$Y > n_\text{axes} * F / 2550 \approxeq n_\text{axes} * 11$$时，我们将受ICI限制。
* 当我们有$$\text{批量大小} < 2550 / n_\text{axes}$$时，纯FSDP变为ICI受限。这里这意味着如果我们想用BS=2M训练，我们最多只能使用$\approx 2400$个芯片，这大约是TPU v5p pod的四分之一。
* 当我们有$$\text{批量大小} < 2 \cdot 2550^2 / 30,000 = 432$$时，混合FSDP + 模型并行变为ICI受限，所以这让我们扩展到大约9k个芯片！然而，TPU v5p pod的最大大小是8k个芯片，超过这个我们必须扩展到较低带宽的数据中心网络（DCN）。

所以这给了我们一个很好的配方，以适应BS=3.5M的单个pod。我们会使用上面的方程，它给出大约X（FSDP）= 1024和Y（MP）= 8。如果模型更大，将有空间将模型分片扩展到16。我们有一点空间将批量大小降低到该pod上的BS=1.5M，并且仍然受计算限制，但我们接近那里的下限。

**要扩展到一个pod以上，我们需要通过DCN扩展。** 因为DCN的带宽较低，通常太慢而无法进行太多有用的FSDP。相反，我们在DCN轴上进行纯数据并行，在pod内进行FSDP。让我们计算数据中心网络（DCN）是否支撑得住。

使用纯数据并行通过DCN，我们需要在每个步骤期间同步权重和优化器状态（当模型完成其反向传递时，我们需要完成AllReduce）。我们实际上可以借用上面纯数据并行部分的数学，它告诉我们当$\text{每pod批量大小} < C_\text{pod} / W_\text{dcn}$时我们变为通信受限，其中这里的RHS是整个pod的总计算和总带宽。

* 我们的总DCN入口+出口带宽是每个主机2.5e10，每个主机有4个芯片。这给了我们切片中约2000个主机，总共`5e13`字节的带宽。
* $$C_\text{pod}$$这里是pod大小乘以每芯片计算，即`8k * 4.5e14 = 3.8e18` FLOPs。

像以前一样，当$T_\text{math} < T_\text{comms}$时我们变为瓶颈，这发生在我们的$\text{每pod批量大小} < C / W_\text{DCN} = 3.8e18 / 5e13 = 76,000$（我们的pod级DCN操作强度）时。对于LLaMA-3，这不会是一个问题，因为我们的每pod批量大小远高于此，但如果我们在较小的切片上训练（例如v5e），这可能会成为一个问题。

<p markdown=1 class="takeaway">**要点：** 这意味着我们可以跨pod相当任意地扩展，例如，使用10个8960芯片的pod，我们可以在89,600个芯片上执行约40M token的全局批量大小，在大约2天内训练LLaMA-3 70B。</p>

## TPU上大语言模型训练的要点

* 增加并行性或减少批量大小都倾向于使我们更受通信限制，因为它们减少了每个芯片执行的计算量。

* 直到合理的上下文长度（~32k），我们可以将Transformer建模为MLP块的堆栈，并通过它们如何分片每层的两个/三个主要矩阵乘法来定义几种并行方案中的每一种。

* 在训练期间，我们考虑4种主要的并行方案，每种都有自己的带宽和计算要求（数据并行、FSDP、模型并行）。

| **策略**                                  | **描述**                                                                                                                                                                                |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **数据并行**                              | 激活是批量分片的，其他所有内容都是完全复制的，我们在反向传递期间全归约梯度。                                                                                                           |
| **FSDP**                                  | 激活、权重和优化器是批量分片的，权重在使用前被收集，梯度被归约分散。                                                                                                                   |
| **模型并行（又名Megatron、张量）**        | 激活沿$$d_\text{model}$$分片，权重沿$$d_{ff}$$分片，激活在W<sub>in</sub>之前被收集，结果在W<sub>out</sub>之后归约分散。                                                               |
| **混合FSDP + 模型并行**                   | 以上两者，其中FSDP收集模型分片的权重。                                                                                                                                                 |

这里是每种方法的"公式"：

$$\small
\begin{array}{cc}
\text{策略} & \text{公式}\\
\hline
\text{DP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D] \\
\text{FSDP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D] \\
\text{MP} & \text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y] \\
\text{MP + FSDP}  & \text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y] \\
\hline
\end{array}$$

* 这些策略中的每一个都有一个限制，在该限制下它变为网络/通信受限，基于它们的每设备计算和通信。这是每层的计算和通信，假设$$X$$是FSDP，$$Y$$是模型并行。

$$
\small
\begin{array}{ccc}
\text{策略} & \text{每层计算} & \text{每层通信} \\
& \text{（忽略门控einsum）} & \text{（字节，前向 + 反向传递）}\\
\hline
\text{DP} & 4BDF/X + 8BDF/X & 0 + 8DF \\
\text{FSDP} & 4BDF/X + 8BDF/X & 4DF + 8DF \\
\text{MP} & 4BDF/Y + 8BDF/Y & 4BD + 4BD \\
\text{FSDP + MP} & 4BDF/(XY) + 8BDF/(XY) & (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) \\
\hline
\end{array}$$

* 纯数据并行很少有用，因为模型及其优化器状态使用字节 = 10x参数计数。这意味着我们很少能在内存中容纳超过几十亿个参数。

* 当$$\text{每分片批量大小} < C / W$$时，数据并行和FSDP变为通信受限，这是网络的算术强度。对于ICI，这是2,550，对于DCN，这是75,000。这可以通过更多并行轴增加。

* 当$$\lvert Y\rvert > F / 2550$$时，模型并行变为通信受限。**对于大多数模型，这大约是8-16路。** 这与批量大小无关。

* 混合FSDP + 模型并行允许我们将批量大小降低到$$2 \cdot 2550^2 / F \approx 400$$。这相当接近我们无论如何都变为HBM带宽受限的点（~200）。

* 跨pod的数据并行需要每个pod的最小批量大小约为75,000，然后变为DCN受限。

* 基本上，如果您的批量大小很大或您的模型很小，事情很简单。您可以在DCN上进行数据并行或FSDP \+ 数据并行。中间部分是事情变得有趣的地方。

## 练习题

让我们使用LLaMA-2 13B作为本节的基本模型。以下是一些细节：

| 超参数                  | 值     |
| ----------------------- | ------ |
| n\_layers (L)           | 40     |
| d\_model (D)            | 5,120  |
| ffw\_multiplier (F / D) | 2.7    |
| n\_heads (N)            | 40     |
| n\_kv\_heads (K)        | 40     |
| d\_qkv (H)              | 128    |
| n\_embeddings (V)       | 32,000 |

**问题1：** LLaMA-2 13B有多少参数（我知道这很愚蠢，但请计算）？*请注意，如[Transformer数学](../transformers)中所述，LLaMA-3有3个大的FFW矩阵，两个上投影和一个下投影。我们在本节中忽略了两个"门控"einsum矩阵，但它们的行为与本节中的W<sub>in</sub>相同。*

{% details 点击这里查看答案。 %}

* FFW参数：$$3LDF$$ = `8.5e9`
* 注意力参数：$$4DNHL$$ = `4.2e9`
* 词汇参数：$$2VD$$ = `0.3e9`
* 总计：`8.5e9 + 4.2e9 + 0.39e9 = 13.1e9`，如预期！

{% enddetails %}

**问题2：** 假设我们使用BS=16M token进行训练并使用Adam。暂时忽略并行性，模型的参数、优化器状态和激活使用了多少总内存？*假设我们以bf16存储参数，以fp32存储优化器状态，并每层检查点激活三次（在三个大矩阵乘法之后）。*

{% details 点击这里查看答案。 %}

参数（bf16）和两个优化器状态（fp32，一阶和二阶矩累加器）使用的总内存为`(2 + 4 + 4) * 13e9 ~ 130GB`。前两个矩阵乘法后的激活形状为$BF$，最后一个后为$BD$（根据上面的Transformer图），因此bf16的总内存为$2 \cdot L \cdot (BD + 2 * BF) = 2LB \cdot (D + 2F)$或`2 * 40 * 16e6 * 5,120 * (1 + 2 * 2.7) ~ 4.2e13 = 42TB`，因为`B=16e16`。所有其他激活或多或少可以忽略不计。

{% enddetails %}

**问题3：** 假设我们想在TPUv5p 16x16x16切片上使用32k序列长度和3M token的总批量大小进行训练。假设我们想使用bfloat16权重和float32优化器，如上所述。

1. 我们可以使用纯数据并行吗？为什么或为什么不？
2. 我们可以使用纯FSDP吗？为什么或为什么不？使用纯FSDP，每个设备将使用多少内存（假设我们仅在3个大FFW矩阵之后进行梯度检查点）。
3. 我们可以使用混合FSDP + 模型并行吗？为什么或为什么不？如果可以，$X$和$Y$应该是什么？每个设备将存储多少内存？仅使用roofline FLOPs估计并忽略注意力，每个训练步骤需要多长时间？

{% details 点击这里查看答案。 %}

首先，让我们写下一些数字。使用32k序列长度和3M批量大小，我们有96的序列批量大小。在TPU v5p 16x16x16切片上，我们有`393TB`的HBM。

1. 我们不能使用纯数据并行，因为它在每个芯片上复制参数和优化器状态，这些已经约130GB（来自Q2），这比我们每个芯片的HBM（96GB）多。

2. 让我们首先纯粹从内存角度看。将Q2中的BS=16M替换为3M，我们得到`~7.86e12`总检查点激活，加上1.3e11优化器状态，这使我们几乎正好达到8e12 = 8TB。TPUv5p切片总共有`393TB`的HBM，所以我们安全地在HBM限制之下。接下来让我们看看我们是否会受通信或计算限制。使用4096个芯片和3个并行轴，我们可以执行`850 * 4096 = 3.48M` token的最小批量大小。这略高于我们的3M批量大小。所以我们实际上受通信限制，这很糟糕。所以一般答案是**不，我们不能单独进行FSDP**。

3. 现在我们知道我们的主要关注点是受通信限制，所以让我们插入一些数字。首先，从上面的判别式，我们知道使用混合FSDP + 模型并行的每芯片批量大小需要高于$2 \cdot 2550^2 / F = 940$，这实际上比纯FSDP稍差。显然，这在某种程度上是我们所做的一些近似的产物，但这表明混合FSDP + 模型并行实际上并没有好多少。部分原因是$F$太小，我们不能进行完整轴值的模型并行。一种解决方法是进行4个芯片的张量并行的小子环，并将第一个轴的剩余带宽专用于FSDP。我们不会计算数学，但检查我们可能可以在不受通信限制的情况下做到这一点是很好的。

{% enddetails %}

**问题4：** 如果我们想降到批量大小1M怎么办？这如何影响问题3的答案？批量大小10M呢？

<h3 markdown=1 class="next-section">第5部分到此结束！第6部分将这些内容应用于真实的LLaMA模型，[点击这里](../applied-training)！</h3>

## 附录

### 附录A - 更多关于FSDP的内容

这是一个很好的额外图表，显示了FSDP如何分片参数/梯度。行依次是纯数据并行、ZeRO-1/2/3。没有太多理由不进行ZeRO-3，因为它实际上具有相同的通信负载。

{% include figure.liquid path="assets/img/fsdp-figure.png" class="img-fluid" %}

**图：**分别显示纯数据并行、ZeRO-1/2/3的参数、梯度和优化器状态内存的图表。[来源](https://arxiv.org/abs/1910.02054)。

### 附录B - 推导反向传递所需的通信

上面，我们将Transformer层前向传递简化为Out\[B, D\] \= In\[B, D\] \*D W<sub>in</sub>\[D, F\] \*<sub>F</sub> W<sub>out</sub>\[F, D\]。我们如何推导反向传递所需的通信？

这自然地遵循前一节中单个矩阵乘法**Y = X \* A**的规则：

$$\frac{dL}{dA} = \frac{dL}{dY}\frac{dY}{dA} = X^T \left(\frac{dL}{dY}\right)$$

$$\frac{dL}{dX} = \frac{dL}{dY}\frac{dY}{dX} = \left(\frac{dL}{dY}\right) A^T$$

使用这个，我们得到以下公式（让Tmp\[B, F\]代表In\[B, D\] \* W<sub>in</sub>\[D, F\]）：

<div markdown=1 class="algorithm">

1. dW<sub>out</sub>[F, D] = Tmp[B, F] \*<sub>B</sub> dOut[B, D] 
2. dTmp[B, F] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F, D] 
3. dW<sub>in</sub> = dTmp[B, F] \*<sub>B</sub> Tmp[B, F] 
4. dIn[B, D] = dTmp[B, F] \*<sub>F</sub> W<sub>in</sub>[D, F]

</div>

请注意，这些公式是数学陈述，没有提及分片。反向传递的工作是计算这四个量。因此，要弄清楚必要的通信，我们只需采用上面四个方程中要进行矩阵乘法的所有量的分片（Tmp、dOut、W<sub>out</sub>、W<sub>in</sub>），这些由我们的并行化方案指定，并使用分片矩阵乘法的规则来弄清楚我们必须进行什么通信。请注意，dOut的分片方式与Out相同。

### 附录C - 混合FSDP + 模型并行的批量大小约束的替代推导

上面我们推导出，当使用FSDP + 模型并行的组合时，我们可以在

$$ \frac{B}{N} > \frac{4\alpha^2}{M_X M_Y F} $$

时受计算限制。这里我们提供这个事实的替代推导。我们首先将通信时间设置为等于计算时间，并寻找使这个等式不可能的条件。

$$\frac{F}{Y \cdot M_X} + \frac{B}{X \cdot M_Y} = \frac{B \cdot F}{N \cdot \alpha}$$

由于$XY=N$，我们可以用$X$重写：

$$\frac{FX}{N \cdot M_X} + \frac{B}{X \cdot M_Y} = \frac{B \cdot F}{N \cdot \alpha}$$，或

$$X^2 \frac{F}{N \cdot M_X} + \frac{B}{M_Y} - X \frac{B \cdot F}{N \cdot \alpha} = 0.$$

由于这是$X$的二次方程，我们没有解的点是判别式变为零的点。这发生在

$$B^2\cdot F^2 \cdot M_X^2 \cdot M_Y^2 - 4\cdot \alpha^2 \cdot F \cdot B \cdot N \cdot M_Y \cdot M_X = 0$$

或通过简化

$$B\cdot F \cdot M_X \cdot M_Y - 4\cdot \alpha^2 \cdot N = 0$$

这给我们

$$B = \frac{4 \cdot \alpha^2 \cdot N}{F \cdot M_X \cdot M_Y}$$

所以我们的总批量大小除以芯片总数不能低于

$$\frac{4 \alpha^2}{F \cdot M_X \cdot M_Y},$$

正如我们上面推导的。