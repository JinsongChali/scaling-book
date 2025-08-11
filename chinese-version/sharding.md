---
layout: distill
title: "分片矩阵及其乘法"
# permalink: /main/
description: '这里我们将解释最大的机器学习模型如何在多个加速器之间分割（或"分片"）。由于大语言模型主要由矩阵乘法组成，理解这一点归结为理解当矩阵分布在设备间时如何进行矩阵乘法。我们基于TPU通信原语的成本开发了一个简单的分片矩阵乘法理论。'
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 3

previous_section_url: "chinese-version/tpus"
previous_section_name: "第2部分：TPU"

next_section_url: chinese-version/transformers
next_section_name: "第4部分：Transformer数学"

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
  - name: "分区记法和集合操作"
  - subsections:
    - name: "分片的统一记法"
    - name: "题外话：如何在代码中描述这一点？"
  - name: "分片数组的计算"
  - subsections:
    - name: "情况1：两个乘数都没有分片的收缩维度"
    - name: "情况2：一个乘数有分片的收缩维度"
    - name: "情况3：两个乘数都有分片的收缩维度"
    - name: "情况4：两个乘数在相同轴上有分片的非收缩维度"
  - name: "深入了解TPU通信原语"
  - subsections:
    - name: "我们最后的通信原语：AllToAll"
    - name: "更多关于ReduceScatter"
  - name: "我们学到了什么？"
  - name: "练习题"

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

## 分区记法和集合操作

当我们在一万个TPU上训练大语言模型时，我们在抽象层面上仍在执行与在单个TPU上训练时相同的计算。区别在于**我们的数组无法装入单个TPU的HBM**，所以我们必须将它们分割开。<d-footnote>值得注意的是，我们也可能为了速度而选择并行化。即使我们可以在较少数量的芯片上容纳，扩展到更多芯片也会给我们更多的FLOPs/s。例如，在推理过程中，我们有时可以在较小的拓扑上容纳，但选择扩展到较大的拓扑以减少延迟。同样，在训练期间，我们经常扩展到更多芯片以减少步骤时间。</d-footnote> 我们称之为"*分片（sharding）*"或"*分区（partitioning）*"我们的数组。

这是一个在4个TPU上分片的2D数组**A**的示例：

{% include figure.liquid path="assets/img/sharding-example.png" class="img-fluid" caption="<b>图：</b>一个形状为<b>A</b>[I, J]的示例数组被分片到4个设备上。两个维度都均匀地分片到2个设备上，分片为<b>A</b>[I<sub>X</sub>, J<sub>Y</sub>]。每个TPU持有总内存的1/4。" %}

请注意，分片数组仍然具有与未分片数组相同的*全局*或*逻辑形状*，比如`(4, 128)`，但它也有一个*设备本地形状*，如`(2, 64)`，它给出了每个TPU持有的实际字节大小（在上图中，每个TPU持有总数组的¼）。现在我们将把这推广到任意数组。

### 分片的统一记法

我们使用*命名轴记法*的变体来描述张量*如何*在设备间以块的形式分片：我们假设存在一个称为**设备网格**的2D或3D设备网格，其中每个轴都被赋予了**网格轴名称**，**例如X**、**Y和Z**。然后，我们可以通过描述数组的每个命名维度如何在物理网格轴上分区来指定矩阵数据如何在设备网格上布局。我们称这种分配为**分片**。

**示例（上图）**：对于上图，我们有：
* **分片：** $A[I_X, J_Y]$，它告诉我们沿网格轴$X$分片第一个轴$I$，沿网格轴$Y$分片第二个轴$J$。这个分片告诉我们每个分片持有数组的$1 / (\lvert X\rvert \cdot \lvert Y\rvert)$。
* **网格：** 上面的设备网格`Mesh(devices=((0, 1), (2, 3)), axis_names=('X', 'Y'))`，它告诉我们有4个TPU组成2x2网格，轴名为$X$和$Y$。

综合起来，我们知道数组的本地形状（单个设备持有的分片大小）是$(\lvert I\rvert / 2, \lvert J\rvert / 2)$，其中$$\lvert I\rvert$$是A的第一个维度的大小，$$\lvert J\rvert$$是A的第二个维度的大小。

**示例（沿1个轴的2D分片）**：$A[I_{XY}, J]$沿X和Y硬件轴分片第一个维度（I）。每个设备的字节数与之前的分片相同，但本地形状不同。现在是$(\lvert I\rvert /(\lvert X\rvert \cdot \lvert Y\rvert), \lvert J\rvert)$。

**可视化这些分片：** 让我们尝试通过查看分布在4个设备上的2D数据数组来可视化这些分片：

{% include figure.liquid path="assets/img/sharding-colored1.png" class="img-fluid img-small" %}

我们将矩阵的*完全复制*形式简单地写为$A[I, J]$，没有分片分配。这意味着*每个*设备都包含整个矩阵的完整副本。

{% include figure.liquid path="assets/img/sharding-colored2.png" class="img-fluid img-small" %}

当我们希望表示其中一个维度已在网格轴上分区时，我们使用网格轴下标来表示。例如，$A[I_X, J]$意味着**I**逻辑轴已在**X**网格维度上分区，但**J**维度*未*分区，块在**Y**网格轴上保持*部分复制*。

{% include figure.liquid path="assets/img/sharding-colored3.png" class="img-fluid img-small" %}

$A[I_X, J_Y]$意味着**I**逻辑轴已在**X**网格轴上分区，**J**维度已在**Y**网格轴上分区。

{% include figure.liquid path="assets/img/sharding-colored4.png" class="img-fluid img-small" %}

我们在下图中说明了其他可能性：

{% include figure.liquid path="assets/img/sharding-colored5.png" class="img-fluid" %}

这里$A[I_{XY}, J]$意味着我们将**X**和**Y**网格轴视为一个更大的扁平化维度，并在所有设备上分区**I**命名轴。多个网格轴下标的顺序很重要，因为它指定了跨网格分区的遍历顺序。

{% include figure.liquid path="assets/img/sharding-colored6.png" class="img-fluid img-small" %}

最后，请注意我们*不能*有多个命名轴沿*相同*网格维度分片。例如，$A[I_X, J_X]$是无意义的、禁止的分片。一旦网格维度被用于分片数组的一个维度，它在某种意义上就被"用完"了。

<b markdown=1 style="color: #57cf57;">小测验：</b> 设**A**是一个形状为`int8[128, 2048]`的数组，分片$A[I_{XY}, J]$，网格`Mesh({'X': 2, 'Y': 8, 'Z': 2})`（总共32个设备）。**A**每个设备使用多少内存？**A**在所有设备上总共使用多少内存？

{% details 点击这里查看答案。 %}

**答案：** 我们的数组**A**在X和Y上分片，在Z上复制，所以每个设备上它的形状为`int8[128 / (2 * 8), 2048] = int8[8, 2048]`，大小为`8 * 2048 = 16,384`字节。因为它在Z上复制，虽然在Z平面内它在X和Y上完全分片，但每个Z平面有一个副本，有2个这样的平面，所以总大小（跨所有设备）是`128 * 2048 * 2 = 512kiB`。

{% enddetails %}

### 题外话：如何在代码中描述这一点？

JAX使用的命名分片语法与我们上面描述的抽象语法非常匹配。我们将在[第10节](../jax-stuff)中更多地讨论这一点，但这里有一个快速预览。您可以在Google Colab中玩这个[这里](https://colab.research.google.com/drive/15cxw66eABwZPG-V4QFmbLfiykPFf_gaP?usp=sharing)并分析结果以查看JAX如何处理不同的分片。这个片段做了3件事：

1. 创建一个**jax.Mesh**，将我们的8个TPU映射到4x2网格，名称'X'和'Y'分配给两个轴。
2. 创建矩阵A和B，其中A沿其两个维度分片，B沿输出维度分片。
3. 编译并执行返回分片数组的简单矩阵乘法。

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

# 创建我们的网格！我们在TPU v2-8 4x2切片上运行，名称为'X'和'Y'。
assert len(jax.devices()) == 8
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# 一个小工具函数来帮助定义我们的分片。PartitionSpec是我们的
# 分片（从轴到名称的映射）。
def P(*args):
  return shd.NamedSharding(mesh, shd.PartitionSpec(*args))

# 我们在非收缩维度上分片A和B，在收缩维度上分片A。
A = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))
B = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P(None, 'Y'))

# 我们可以在这些分片数组上执行矩阵乘法！out_shardings告诉我们想要
# 输出如何分片。JAX/XLA为我们处理其余的分片。
compiled = jax.jit(lambda A, B: jnp.einsum('BD,DF->BF', A, B), out_shardings=P('X', 'Y')).lower(A, B).compile()
y = compiled(A, B)
```

JAX的酷之处在于这些数组的行为就像它们没有分片一样！`B.shape`会告诉我们全局或逻辑形状（2048, 8192）。我们必须实际查看`B.addressable_shards`以查看它是如何在本地分片的。我们可以在这些数组上执行操作，JAX将尝试找出如何广播或重塑它们以执行操作。例如，在上面的示例中，**A**的本地形状是`[2, 1024]`，**B**是`[2048, 4096]`。JAX/XLA将根据需要自动在这些数组之间添加通信以执行最终的乘法。

## 分片数组的计算

如果您有一个分布在许多设备上的数据数组，并希望对其执行数学运算，与分片数据和计算相关的开销是什么？

显然，这取决于所涉及的计算。

* 对于*逐元素*操作，操作分布式数组**没有开销**。
* 当我们希望对驻留在许多设备上的元素执行操作时，事情变得复杂。幸运的是，对于大多数机器学习，几乎所有计算都以矩阵乘法的形式进行，它们相对容易分析。

本节的其余部分将处理如何乘以分片矩阵。大致上，这涉及移动矩阵的块，以便您可以完全乘以或求和每个块。**每个分片将涉及不同的通信。** 例如，$A[I_X, J] \cdot B[J, K_Y] \to C[I_X, K_Y]$可以在没有任何通信的情况下相乘，因为*收缩维度*（J，我们实际求和的维度）未分片。但是，如果我们希望输出未分片（即$A[I_X, J] \cdot B[J, K_Y] \to C[I, K]$），我们需要将$A$或$C$复制到每个设备。这两种选择有不同的通信成本，所以我们需要计算这个成本并选择最低的。

{% details 您可以从"块矩阵乘法"的角度来考虑这一点。 %}

首先让我们回顾一下"块矩阵"的概念，或矩阵的嵌套矩阵：

$$\begin{equation}
\begin{pmatrix}
a_{00} & a_{01} & a_{02} & a_{03} \\
a_{10} & a_{11} & a_{12} & a_{13} \\
a_{20} & a_{21} & a_{22} & a_{23} \\
a_{30} & a_{31} & a_{32} & a_{33}
\end{pmatrix}
=
\left(
\begin{matrix}
\begin{bmatrix}
a_{00} & a_{01} \\
a_{10} & a_{11}
\end{bmatrix} \\
\begin{bmatrix}
a_{20} & a_{21} \\
a_{30} & a_{31}
\end{bmatrix}
\end{matrix}
\begin{matrix}
\begin{bmatrix}
a_{02} & a_{03} \\
a_{12} & a_{13}
\end{bmatrix} \\
\begin{bmatrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{bmatrix}
\end{matrix}
\right)
=
\begin{pmatrix}
\mathbf{A_{00}} & \mathbf{A_{01}} \\
\mathbf{A_{10}} & \mathbf{A_{11}}
\end{pmatrix}
\end{equation}$$

矩阵乘法具有一个很好的性质，即当矩阵乘数以块的形式写出时，乘积可以按照标准规则以块矩阵乘法的形式写出：

$$\begin{equation}
\begin{pmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{pmatrix}
\cdot
\begin{pmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11}
\end{pmatrix}
=
\begin{pmatrix}
A_{00}B_{00} + A_{01}B_{10} & A_{00}B_{01} + A_{01}B_{11} \\
A_{10}B_{00} + A_{11}B_{10} & A_{10}B_{01} + A_{11}B_{11}
\end{pmatrix}
\end{equation}$$

这意味着实现分布式矩阵乘法归结为在网络上移动这些分片块，对块执行*本地*矩阵乘法，并对其结果求和。**然后问题是添加什么通信，以及它有多昂贵。**

{% enddetails %}

方便的是，我们可以将所有可能的分片归结为大约4种我们需要考虑的情况，每种情况都有我们需要添加的通信规则
1. **[情况1](#case-1-neither-multiplicand-has-a-sharded-contracting-dimension)：** 两个输入都没有沿收缩维度分片。_我们可以在没有任何通信的情况下乘以本地分片。_
2. **[情况2](#case-2-one-multiplicand-has-a-sharded-contracting-dimension)：** 一个输入有分片的收缩维度。_我们通常沿收缩维度"AllGather"分片输入。_
3. **[情况3](#case-3-both-multiplicands-have-sharded-contracting-dimensions)：** 两个输入都沿收缩维度分片。_我们可以乘以本地分片，然后"AllReduce"结果。_
4. **[情况4](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis)：** 两个输入在同一轴上有分片的非收缩维度。我们不能在不先AllGather两个输入之一的情况下继续。

您可以将这些视为简单需要遵循的规则，但理解为什么这些规则成立以及它们有多昂贵也很有价值。我们现在将详细介绍其中的每一个。

### 情况1：两个乘数都没有分片的收缩维度

**引理：** 当乘以分区张量时，计算是有效的，输出遵循输入的分片，*除非*收缩维度被分片或两个张量在同一轴上有分片的非收缩维度。例如，这工作得很好

$$\begin{equation*}
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow \mathbf{C}[I_X, K_Y]
\end{equation*}$$

没有任何通信，并导致张量在X和Y硬件维度上分片。试着想想为什么会这样。基本上，计算*独立于*分片，因为每个批处理条目都有一些被收缩的轴的本地块，它可以乘以和减少。任何这些情况都可以正常工作并遵循此规则：

$$\begin{align*}
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I, K] \\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I_X, K]\\
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I, K_Y]\\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I_X, K_Y]
\end{align*}$$

因为**A**和**B**都没有分片的收缩维度**J**，我们可以简单地执行输入的本地块矩阵乘法，结果将*已经*根据所需的输出分片进行分片。当两个乘数在同一轴上有分片的非收缩维度时，这不再成立（详见[无效分片](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis)部分）。

### 情况2：一个乘数有分片的收缩维度

让我们考虑**A**在收缩**J**维度中分片与完全复制的**B**的分布式矩阵乘法的简单情况：

$$\mathbf{A}[I, J_X] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

我们不能简单地执行本地**A**、**B**块相互之间的本地矩阵乘法，因为我们缺少**A**收缩轴的完整数据。通常，我们首先在本地"**AllGather**"**A**的分片，然后才与**B**相乘：

$$\textbf{AllGather}_X[I, J_X] \rightarrow \mathbf{A}[I, J]$$

$$\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

AllGather沿轴*移除分片*并将分布在设备上的分片重新组装到该轴上的*每个*设备上。使用上述符号，AllGather从一组轴中移除下标，例如

$$\textbf{AllGather}_{XY}(A[I_{XY}, J]) \rightarrow A[I, J]$$

我们也不必移除给定维度的所有下标，例如$$A[I_{XY}, J] \rightarrow A[I_Y, J]$$也是AllGather，只是在单个轴上。

请注意，我们也可能希望使用AllGather来移除*非收缩*维度分片，例如矩阵乘法：

$$A[I_X, J] \cdot B[J, K] \rightarrow C[I, K]$$

我们将类似地沿**X**进行AllGather以移除输出分片，但在这种情况下，我们可以自由选择在矩阵乘法之前或之后进行，不像AllGather收缩维度的情况，我们被迫在执行矩阵乘法之前进行。

**AllGather实际上是如何执行的？** 要沿单个轴执行AllGather，我们需要沿轴传递所有分片，直到每个设备都有一个副本。图1显示了一个示例。8个设备中的每一个都以数组的1/8开始，最后都有所有副本。一种有效的方法是让每个设备沿分片维度环传递其分片，要么在一个方向，要么在两个方向。如果我们做一个方向，它需要$$N - 1$$跳，每个链接的大小为$$\text{总大小} / N$$，否则我们有$\lceil \frac{N}{2} \rceil$跳，每个链接的大小为$$2 \cdot \text{总大小} / N$$。

{% include figure.liquid path="assets/img/all-gather.gif" %}

**这需要多长时间？** 让我们采用双向AllGather并计算它需要多长时间。设$$V$$为数组中的字节数，$$\lvert X\rvert$$为收缩维度上的分片数。然后从上图中，每跳在每个方向发送$V / \lvert X\rvert$字节，所以每跳需要

$$T_{hop} = \frac{2 \cdot V}{|X| \cdot W_\text{ici}}$$

其中$$W_\text{ici}$$是**双向**ICI带宽。<d-footnote>分子中的因子2来自我们使用双向带宽的事实。我们在每个方向发送$V / |X|$，或总共$2V / |X|$。</d-footnote> 我们需要发送总共$\lvert X\rvert / 2$跳以到达每个TPU<d-footnote>技术上，$\lceil | X | / 2 \rceil$</d-footnote>，所以总的归约需要

$$T_{total} = \frac{2 \cdot V \cdot |X|}{2 \cdot |X| \cdot W_\text{ici}}$$

$$T_{total} = \frac{V}{W_\text{ici}}$$

请注意，这**不依赖于$$\lvert X\rvert$$！** 这有点引人注目，因为这意味着即使我们的TPU只是本地连接的，连接的局部性也不重要。我们只是被每个链接的速度所瓶颈。

<p markdown=1 class="takeaway">**要点：** 在吞吐量受限的情况下执行AllGather（或ReduceScatter或AllReduce）时，实际通信时间仅取决于数组的大小和可用带宽，而不是我们的数组分片的设备数量！</p>

**关于ICI延迟的说明：** 通过ICI链接的每一跳都有一些固有的开销，无论数据量如何。这通常约为1us。这意味着当我们的数组$$A$$非常小且每跳需要少于1us时，我们可以进入"延迟受限"的状态，其中计算_确实_依赖于$$\lvert X \rvert$$。

{% details 有关完整详细信息，请单击此处。 %}

设$$T_\text{min}$$为单跳的最短时间。然后

$$T_{hop} = \max \left[ T_{min}, \frac{2 \cdot V}{|X| \cdot W_\text{ici}} \right]$$

$$T_{total} = \max \left[ \frac{T_{min} \cdot |X|}{2}, \frac{V}{W_\text{ici}} \right]$$

因为我们执行$$\lvert X \rvert / 2$$跳。对于大型归约或收集，我们完全受带宽限制。我们发送的数据太多，以至于每跳的开销基本上可以忽略不计。但对于小数组（例如，从模型采样时），这并非微不足道，ICI带宽不相关。我们纯粹受延迟限制。另一种说法是，给定特定的TPU，例如具有`4.5e10`单向ICI带宽的TPU v5e，发送任何低于`4.5e10 * 1e-6 = 45kB`的缓冲区都将受延迟限制。

{% enddetails %}

这是TPU v5e 8x16切片上AllGather带宽的经验测量。数组沿16轴分片，因此它有一个完整的双向环。

{% include figure.liquid path="assets/img/all-gather-bandwidth.png" class="img-small" caption="<b>图：</b>AllGather期间TPU v5e的经验带宽和估计链接带宽。橙色的BW是每秒AllGather的实际字节数，而蓝色曲线显示根据集合的已知成本计算的经验单向链接带宽。" %}

请注意，我们只达到了声称的峰值带宽（`4.5e10`）的约95%，并且我们只在约10MB时达到此峰值，当16路分片时，每个设备约500kB。

**当我们在多个轴上AllGather时会发生什么？** 当我们在多个轴上收集时，我们有多个ICI维度来执行收集。例如，AllGather<sub>XY</sub>([B, D<sub>XY</sub>])在两个硬件网格轴上操作。这将可用带宽增加了$$n_\text{axes}$$倍。

{% details 有关完整详细信息，请单击此处。 %}

一般来说，我们有

$$T_{total} = \max \left[ \frac{T_{min} \cdot \sum_{i} |X_i|}{2}, \frac{V}{W_\text{ici} \cdot n_\text{axes}} \right]$$

其中$$\sum_i \lvert X_i \rvert / 2$$是TPU网格中最长路径的长度。

{% enddetails %}

<b markdown=1 style="color:rgb(144, 92, 255);">小测验2 [AllGather时间]：</b> 使用[第2部分](../tpus)的数字，在具有2D网格`{'X': 8, 'Y': 4}`的TPUv5e上执行AllGather<sub>Y</sub>([E<sub>Y</sub>, F]) → [E, F]需要多长时间，$$E = 2048$$，$$F = 8192$$在bfloat16中？$$E=256, F=256$$呢？

{% details 点击这里查看答案。 %}

**答案：** 让我们首先计算一些基本量：

1) TPU v5e对其2个轴中的每一个都有4.5e10字节/秒的单向ICI带宽。
2) 在bfloat16中对于(a)，我们有$A[E_Y, F]$，所以每个设备持有形状为bfloat16[512, 8192]的数组，具有512 * 8192 * 2 = 8.4MB。总数组大小为2048 * 8192 * 2 = 34MB。

*对于第(1)部分*，我们可以使用上面的公式。由于我们在一个轴上执行AllGather，我们有$T_{\text{comms}} = 34e6 / 9e10 = 377\mu s$。要检查我们是否不受延迟限制，我们知道在大小为4的轴上，我们最多有3跳，所以我们的延迟界限类似于3us，所以我们不接近。但是，TPU v5e只有在一个轴大小为16时才有环绕连接，所以这里*我们实际上不能做完全双向的AllGather*。我们必须为数据从边缘到达另一边缘进行3跳，所以理论上我们更像$T_{\text{comms}} = 3 * 8.4e6 / 4.5e10 = 560\mu s$。[**这里**](https://imgur.com/a/RkvpRGQ)**是来自[这个Colab](https://colab.research.google.com/drive/15tDZMfNqm2vJjvSzw5VC9qtSwc5td-oV?usp=sharing)的实际配置文件**，显示$680 \mu s$，这是合理的，因为我们可能没有获得100%的理论带宽！*对于第(2)部分*每个分片的大小为`64 * 256 * 2 = 32kB。32e3 / 4.5e10 = 0.7us`，所以我们受延迟限制。由于我们有3跳，这将大约需要3 * 1us = 3us。[实际上，它更接近8us。](https://imgur.com/a/HZLQmYs)

{% enddetails %}

### 情况3：两个乘数都有分片的收缩维度

第三个基本情况是当两个乘数都在其收缩维度上分片，沿同一网格轴：

$$\textbf{A}[I, J_X] \cdot \textbf{B}[J_X, K] \rightarrow C[I, K]$$

在这种情况下，*本地*分片块矩阵乘法至少*可能*执行，因为它们将共享相同的收缩索引集。但每个乘积只代表完整所需乘积的*部分和*，沿**X**维度的每个设备将留下此最终所需乘积的不同*部分和*。这太常见了，我们扩展我们的符号以明确标记此条件：

$$\textbf{A}[I, J_X] \cdot_\text{LOCAL} \textbf{B}[J_X, K] \rightarrow C[I, K] \{\ U_X \}$$

符号**{ U<sub>X</sub> }**读作"沿X网格轴**未归约**"，指的是操作在某种意义上"不完整"的状态，因为它只会在最终求和待定时完成。$\cdot_\text{LOCAL}$语法意味着我们执行本地求和但保留结果未归约。

这可以看作是关于矩阵乘法和外积的以下结果：

$$A \cdot B = \sum_{i=1}^{P} \underbrace{A_{:,i} \otimes B_{i,:}}_{\in \mathbb{R}^{n \times m}}$$

其中⊗是外积。因此，如果轴**X**上的TPU **i**有**A**的第**i**列和**B**的第**i**行，我们可以进行本地矩阵乘法以获得$$A_{:,i} \otimes B_{i,:} \in \mathbb{R}_{n\times m}$$。该矩阵在每个条目中都有**A • B**在该条目处的第**i**项求和。我们仍然需要在**P**上执行该求和，我们在网格轴**X**上分片，以获得完整的**A • B**。如果我们按块（即分片）写**A**和**B**，然后对结果的每个分片求和，这工作方式相同。

我们可以使用跨**X**轴的完整**AllReduce**执行此求和以纠正此问题：

$$\begin{align*}
A[I, J_X] \cdot_\text{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{AllReduce}_X C[I, K] \{ U_X \} \rightarrow &\ C[I, K]
\end{align*}$$

AllReduce删除部分和，导致沿轴的*每个*设备都有相同的完全求和值。AllReduce是我们将在本节中讨论的几个关键通信中的第二个，第一个是AllGather，其他是ReduceScatter和AllToAll。AllReduce采用具有未归约（部分求和）轴的数组，并通过沿未归约轴传递这些分片并累积结果来执行求和。签名是

$$\textbf{AllReduce}_Y A[I_X, J] \{U_Y\} \rightarrow A[I_X, J]$$

这意味着它只是删除$\\{U_Y\\}$后缀，但其他方面保持结果不变。

**AllReduce有多昂贵？** AllReduce如何执行的一个心理模型是每个设备将其分片发送给其邻居，并对其接收的所有分片求和。显然，这比AllGather更昂贵，因为每个"分片"都具有与完整数组相同的形状。通常，**AllReduce的成本是AllGather的两倍。** 看到这一点的一种方法是注意**AllReduce**可以表示为另外两个原语的组合：**ReduceScatter**和**AllGather**。像AllReduce一样，ReduceScatter解析数组上的部分和，但导致输出沿给定维度"分散"或分区。AllGather收集所有这些片段并沿该物理轴"取消分区/取消分片/复制"逻辑轴。

$$\begin{align*}
\textbf{ReduceScatter}_{Y,J} : A[I_X,J] \{U_Y\} \rightarrow &\ A[I_X, J_Y] \\
\textbf{AllGather}_Y : A[I_X, J_Y] \rightarrow &\ A[I_X, J]
\end{align*}$$

**ReduceScatter呢？** 正如AllReduce删除下标（上面的$F_Y \to F$），ReduceScatter对未归约/部分求和的数组求和，然后沿同一网格轴分散（分片）不同的逻辑轴。$[F]\\{U_Y\\} \to [F_Y]$。动画显示了如何完成此操作：请注意，它与AllGather非常相似，但不是保留每个分片，而是将它们求和。因此，其延迟大致相同，不包括执行归约所需的时间。

{% include figure.liquid path="assets/img/reduce-scatter.gif" class="img-fluid" %}

每跳的通信时间只是每分片字节$V / Y$除以带宽$W_\text{ici}$，就像AllGather一样，所以我们有

$$T_{\text{每个AllGather或ReduceScatter的通信}} = \frac{V}{W_\text{ici}}$$

$$T_{\text{每个AllReduce的通信}} = 2 \cdot \frac{V}{W_\text{ici}}$$

其中$$W_\text{ici}$$是双向带宽，只要我们有一个完整的环来归约。

### 情况4：两个乘数在相同轴上有分片的非收缩维度

每个网格维度在分片张量时最多可以出现一次。执行上述规则有时会导致违反此规则的情况，例如：

$$A[I_X, J] \cdot B[J, K_X] \rightarrow C[I_X, K_X]$$

这是无效的，因为沿维度**X**的给定分片，比如**i**，将具有**C**的**(i, i)**分片，即对角线条目。然后，在所有分片中没有足够的信息来恢复除结果的对角线条目之外的任何内容，因此我们不能允许这种分片。

解决此问题的方法是AllGather某些维度。这里我们有两个选择：

$$\begin{align*}
\textbf{AllGather}_X A[I_X, J] \rightarrow &\ A[I, J] \\
A[I, J] \cdot B[J, K_X] \rightarrow &\ C[I, K_X]
\end{align*}$$

或

$$\begin{align*}
\textbf{AllGather}_X B[J, K_X] \rightarrow &\ B[J, K] \\
A[I_X, J] \cdot B[J, K] \rightarrow &\ C[I_X, K]
\end{align*}$$

在任何一种情况下，结果在其形状中只会提到**X**一次。我们选择哪一个将基于以下操作需要什么分片。

## 深入了解TPU通信原语

前面的4个案例介绍了用于执行分片矩阵乘法的几个"核心通信原语"：

1. **AllGather：** 从分片中删除下标，收集分片。
2. **ReduceScatter：** 通过对该轴上的分片求和，从数组中删除"未归约"后缀，使数组在第二个轴上分片。
3. **AllReduce：** 删除"未归约"后缀，使数组沿该轴未分片。

还有一个需要提及的核心通信原语，它出现在专家混合（MoE）模型和其他计算中：**AllToAll**。

### 我们最后的通信原语：AllToAll

在考虑分片矩阵乘法时不会自然出现的最后一个基本集合，但在实践中经常出现的是**AllToAll**集合，或更准确地说是*分片转置*或重新分片操作的特殊情况。例如

$$\textbf{AllToAll}_{X, J} A[I_X, J] \rightarrow A[I, J_X]$$

AllToAll通常需要在分片计算的不同区域之间重新排列分片布局，这些区域没有兼容的布局方案。在考虑分片专家混合模型时，它们自然出现。*您可以将AllToAll视为将下标从一个轴移动到另一个轴*。因为AllToAll不需要跨环复制每个分片的所有数据，所以它实际上比AllGather*便宜*（因子为¼）<d-footnote>对于偶数大小的双向环，每个设备将向右发送$(N/2 + (N/2-1) + … + 1)$块，向左发送$((N/2-1) + … + 1)$块$= 0.5 \cdot (N / 2) \cdot (N/2 + 1) + 0.5 \cdot (N / 2) \cdot (N/2 - 1) = N^2/4$。每个块的大小（即分片的分片）是$\text{字节} / N^2$，因此每个设备的成本是$(\text{字节} / N^2) \cdot N^2 / 4 = \text{字节} / 4$。此结果跨所有设备扩展，因为总带宽随设备数量扩展。</d-footnote>。

{% include figure.liquid path="assets/img/all-to-all.gif" class="img-fluid" %}

如上所述，$V$字节数组的总成本为

$$T_\text{每个AllToAll的通信} = \frac{V}{4 \cdot W_\text{ici}}$$

其中像往常一样$W_\text{ici}$是双向ICI带宽。这是AllGather成本的1/4和AllReduce的1/8。

### 更多关于ReduceScatter

ReduceScatter是比最初出现的更基本的操作，因为它实际上是AllGather的导数，反之亦然。即，如果在前向传递中我们有：

$$\textbf{AllGather}_X A[I_X] \rightarrow A[I]$$

然后我们ReduceScatter反向模式导数**A'**（通常在每个分片上不同）以导出分片的**A'**：

$$\textbf{ReduceScatter}_X A'[I] \{ U_X \} \rightarrow A'[I_X]$$

同样，前向传递中的$$\text{ReduceScatter}_X(A[I] \{U_X\}) \to A[I_X])$$意味着后向传递中的$$\text{AllGather}_{X}(A'[I_X]) \to A'[I]$$。

将AllReduce转换为AllGather和ReduceScatter还具有方便的属性，即我们可以将最终的AllGather推迟到稍后的某个时刻。很常见的是，我们宁愿不支付在设备间复制重新组装完整矩阵乘积的成本。相反，即使在组合具有分片收缩维度的两个乘数的情况下，我们也希望保留分片状态：

$$A[I, J_X] \cdot B[J_X, K] \rightarrow C[I, K_X]$$

在这种情况下，我们也可以执行ReduceScatter而不是AllReduce，然后可选地在稍后的某个时间执行AllGather，即

$$\begin{align*}
A[I, J_X] \cdot_{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{ReduceScatter}_{X,K} C[I, K] \{ U_X \} \rightarrow &\ C[I, K_X]
\end{align*}$$

请注意，ReduceScatter*引入*分片维度，因此在这种情况下具有沿**I**或**K**命名维度分片的自然自由度。使用ReduceScatter时，我们通常需要选择*哪个*命名维度来引入新的分片（尽管选择通常由更大的建模上下文强制）。这就是为什么我们使用语法**ReduceScatter<sub>X,K</sub>**来指定要分片的轴。

## 我们学到了什么？

* 数组的分片由命名我们的TPU网格的物理硬件轴的**Mesh**和将网格轴名称分配给数组的逻辑轴的**Sharding**指定。
  * 例如，**A**[I<sub>XY</sub>, J]描述了一个抽象数组**A**，其第一个维度沿两个网格轴X和Y分片。结合Mesh(mesh_shape=(4, 8), axis_names=('X', 'Y'))或缩写的Mesh({'X': 4, 'Y': 8})，这告诉我们我们的数组沿第一个维度32路分片。

* **分片数组的算术与未分片数组的算术完全相同，除非您沿分片轴执行收缩**。在这种情况下，我们必须引入一些通信。我们考虑四种情况：

  1. *两个数组都没有沿收缩维度分片*：不需要通信。
  2. *一个数组沿收缩维度分片*（或收缩维度沿不同轴分片）：我们在执行操作之前AllGather其中一个输入。
  3. *两个数组沿收缩维度相同地分片：*我们在本地乘以分片，然后执行AllReduce或ReduceScatter。
  4. *两个数组沿非收缩维度的同一网格轴分片：*我们首先AllGather其中一个输入。

* TPU使用大约**4个核心通信原语**：
  1. AllGather：$[A_X, B] \to [A, B]$
  2. ReduceScatter：$[A, B] \\{U_X\\} \to [A, B_X]$
  3. AllToAll：$[A, B_X] \to [A_X, B]$
  4. AllReduce：$[A_X, B]\\{U_Y\\} \to [A_X, B]$（技术上不是原语，因为它结合了ReduceScatter + AllGather）

{% include figure.liquid path="assets/img/all-collectives.png" class="img-fluid" %}

* 这些操作中每一个的成本和延迟**不依赖于轴的大小（只要它们受带宽限制）**，而只依赖于输入数组的大小和链接的带宽。对于单向AllGather/ReduceScatter：

$$T_{\text{每个AllGather或ReduceScatter的通信}} = \frac{\text{数据量}}{\text{带宽}} \cdot \frac{\text{轴} - 1}{\text{轴}}
\longrightarrow \frac{\text{数据量}}{\text{带宽（双向）}}$$

* AllReduce由ReduceScatter后跟AllGather组成，因此具有上述成本的2倍。AllToAll只需要在环周围部分传递分片，因此是AllGather成本的¼。这是一个总结：

| 操作              | 描述                                                                                | 语法                             | 运行时间                                          |
| :---------------- | :---------------------------------------------------------------------------------- | :------------------------------- | :----------------------------------------------- |
| **AllGather**     | 沿轴收集分片数组的所有分片，删除下标。                                               | $[A_X, B] \to [A, B]$            | 字节 / (双向ICI带宽 * num_axes)                  |
| **ReduceScatter** | 沿轴对部分求和的数组求和，并沿另一个轴分片（添加下标）。                             | $[A, B] \\{U_X\\} \to [A_X, B]$  | 与AllGather相同                                  |
| **AllReduce**     | 沿轴对部分求和的数组求和。删除{ U<sub>x</sub> }。结合AllGather和ReduceScatter。    | $[A_X, B]\\{U_Y\\} \to [A_X, B]$ | 2 * AllGather                                    |
| **AllToAll**      | 收集（复制）一个轴并沿同一轴分片不同的维度。                                        | $[A, B_X] \to [A_X, B]$          | 双向环的AllGather / 4                            |

## 练习题

*这里有一些基于本节内容的指导性问题。我们目前不会包括所有答案，但我们会尽可能写出更多答案。*

**问题1 [复制分片]**：数组分片为$A[I_X, J, K, \ldots]$（即，仅跨$X$分片），网格为`Mesh({'X': 4, 'Y': 8, 'Z': 2})`。$A$在所有芯片上占用的总字节数与数组的一个副本的大小之比是多少？

{% details 点击这里查看答案。 %}

我们的数组仅沿X分片，X的大小为4，因此每个分片的大小实际上为$[I / 4, J, K, \ldots] = \text{sizeof}(A) / 4$。由于我们的数组在Y和Z上复制，总大小为$Y \cdot Z \cdot \text{sizeof}(A)$，因此总大小与单芯片大小的比率为$Y \cdot Z \cdot \text{sizeof}(A) / \text{sizeof}(A) = 16$。

{% enddetails %}

**问题2 [AllGather延迟]**：在具有网格`Mesh({'X': 4, 'Y': 4, 'Z': 4})`的TPUv4p 4x4x4切片上，$\text{AllGather}_X([B_X, D_Y])$应该需要多长时间，如果$B=1024$和$D=4096$在bfloat16中？$$\text{AllGather}_{XY}([B_X, D_Y])$$呢？$$\text{AllReduce}_Z([B_X, D_Y] \{U_Z \})$$呢？

{% details 点击这里查看答案。 %}

我们在所有轴上都有环绕链接，因为我们有一个完整的`4x4x4`立方体，所以我们有9e10双向带宽可以使用。

1. 因为我们只是在一个轴上收集而另一个是分片的，我们实际上在1个轴上收集$2BD / Y$字节。由于我们的TPU v4p的ICI带宽是每秒9e10字节双向，这将需要$2BD / (9e10 \cdot Y) = 2 \cdot 1024 \cdot 4096 / (9e10 \cdot 4) = 23 \mu s$。

2. 我们的带宽是以前的两倍，但我们正在AllGather完整数组，所以`T = 2BD / (2 * W) = 2*1024*4096 / (2 * 9e10) = 46us`。这远离4us的延迟界限（每跳1us），所以我们没问题。

3. AllReduce的成本是AllGather的两倍。每个分片的大小为$2BD / (X * Y)$，因此成本约为$4BD / (X * Y * W)$，或大约`4 * 1024 * 4096 / (16 * 9e10) = 11.6us`。

{% enddetails %}

**问题3 [延迟受限的AllGather]**：假设我们正在执行$\text{AllGather}_X([B_X])$，但$B$非常小（比如128）。在具有网格`Mesh({'X': 4, 'Y': 4, 'Z': 4})`的TPUv4p 4x4x4切片上，这应该需要多长时间在bfloat16中？*提示：您可能受延迟限制。*

{% details 点击这里查看答案。 %}

我们的数组在bfloat16中仅使用256字节总计，每个设备仅64字节。由于我们在TPU v4p上有大小为4的轴，我们有环绕链接，所以我们可以在两个方向发送数组。使用`4.5e10`的单向带宽，每跳大约需要`64 / 4.5e10 ~ 0`，所以我们肯定受延迟限制。计算跳数，我们可以在仅2跳中完成完整收集，因此大约2us是一个很好的估计。

{% enddetails %}

**问题4 [矩阵乘法策略]**：要执行$X[B, D] \cdot_D Y[D_X, F] \to Z[B, F]$，在本节中我们告诉您执行$\text{AllGather}_X(Y[D_X, F])$并乘以完全复制的矩阵（情况2，*策略1*）。相反，您可以像$X[B, D_X] \cdot_D Y[D_X, F] \to Z[B, F] \\{U_X\\}$（情况4，*策略2*）那样乘以本地分片，然后$\text{AllReduce}_X(Z[B, F] \\{ U_X\\})$。这些中的每一个执行多少FLOP和通信？哪个更好，为什么？

{% details 点击这里查看答案。 %}

让我们从我们的基线（*策略1*）开始。如我们所示，AllGather的成本是$2DF / W_\text{ici}$。一旦我们有了完全复制的数组，总计算时间是$2BDF / C$（其中$C$是我们的加速器FLOPs/s，因为每个TPU执行相同的FLOPs）。所以我们有

$$T_\text{总计（策略1）} = \max\left(\frac{2BDF}{C}, \frac{2DF}{W_\text{ici}}\right)$$

相比之下，新策略（策略2）对$2BF$字节执行AllReduce，成本为$4BF / W_\text{ici}$，但执行$1 / X$更少的FLOPs（因为计算是分片的）。这意味着我们执行$2\cdot B\cdot D\cdot F / X$ FLOPs，结果AllReduce在bfloat16中传达$$2 \cdot 2 \cdot B \cdot F$$字节。因此，我们的*策略2*的总时间（没有AllGather，只是稍后的AllReduce）大约是

$$T_\text{总计} = \max\left(\frac{2BDF}{X \cdot C}, \frac{4BF}{W_\text{ici}}\right)$$

问题是：*这些中哪个更大？*当$D / (X \cdot C) > 2 / W_\text{ici}$时，策略(2)受计算限制，或者当$D / 2X > C / W_\text{ici} \approx 2550 \rightarrow X < D / (2 * 2550)$时。我们可能合理地期望$D \approx 8k$，所以这意味着大约$X < 2$，这是不太可能的——因此我们基本上总是使用策略2受通信限制。使用基线（策略1），当$$B < C / W_\text{ici} = 2550$$时我们受通信限制，这经常但并非总是如此。

所以如果$B < 2550$，我们在两种情况下都受通信限制，我们有

$$T_\text{策略2的通信} < T_\text{策略1的通信} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2DF}{W_\text{ici}}$$

当$D > 2B$其中$2B < 5100$时，这是真的。这通常是真的，所以如果我们的批次很小，策略2有时会更好。当我们的批次很大（$B > 2550$）时，我们有

$$T_\text{策略2的通信} < T_\text{策略1的数学} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2BDF}{C}$$

当$2 / W_\text{ici} < D / C$时，这是真的，或者当$D > 2 * 2550 = 5100$时，这对于大型模型通常是真的。因此，这种替代策略通常对大型模型更好，除非$D$很小。

*为什么我们不总是这样做？*嗯，实际上我们有时可能会这样做，但通常很少有一个输入的收缩维度沿着另一个输入未分片的轴分片的情况。例如，如果我们正在执行FSDP（在[第5节](../training)中解释），我们将在数据维度上分片我们的参数，但我们的激活_也将沿数据分片_。所以从这个意义上说，这并不经常出现。

{% enddetails %}

**问题5 [最小延迟]**：假设我想在TPUv5p 4x4x4上以最低可能的延迟执行矩阵乘法$A[B, D] \cdot_D B[D, F] \to C[B, F]$。我的输入应该如何分片？总的FLOPs和通信时间是多少？

**问题6：**假设我们想在TPUv5e 4x4上执行$A[I_X, J_Y] \cdot_J B[J_Y, K] \to C[I_X, K]$。我们执行什么通信？在通信与计算上花费了多少时间？

* $A[I_X, J] \cdot_J B[J_X, K_Y] \to C[I_X, K_Y]$呢？这是训练的最标准设置，我们结合数据、张量和零分片。
* $A[I_X, J] \cdot_J B[J, K_Y] \to C[I_X, K_Y]$呢？这是推理的标准，我们进行纯张量并行（+数据）。

**问题7：**典型的Transformer块有两个矩阵$B[D, F]$和$C[F, D]$，其中$F \gg D$。使用批量大小B，整个块是$$C \cdot B \cdot x$$，其中$$x[B, D]$$。让我们选择$$D=8192$$，$$F=32768$$和$$B=128$$，并假设一切都在bfloat16中。假设我们在TPUv5e 2x2切片上运行，但假设每个TPU只有300MB的空闲内存。**B、C和输出应该如何分片以保持在内存限制以下，同时最小化总时间？在通信和FLOPs上花费了多少时间？**

**问题8 [挑战]**：使用上面的短代码片段作为模板，分配分片数组并使用pmap或shard_map对4个主要通信原语（AllGather、AllReduce、ReduceScatter和AllToAll）中的每一个进行基准测试。您将要使用`jax.lax.all_gather`、`jax.lax.psum`、`jax.lax.psum_scatter`和`jax.lax.all_to_all`。您了解这些函数的语义吗？它们需要多长时间？

**问题9 [分片矩阵乘法的另一种策略？]**：[上面](#case-2-one-multiplicand-has-a-sharded-contracting-dimension)我们声称，当只有一个矩阵乘法的输入沿其收缩维度分片时，我们应该AllGather分片矩阵并在本地执行结果收缩。您可能想到的另一种策略是执行分片矩阵乘法，然后AllReduce结果（就像两个输入都沿收缩维度分片一样），即$A[I, J_X] *_J B[J, K] \to C[I, K]$通过

1. $C[I, K] \\{ U_X \\} = A[I, J_X] \cdot B[J_X, K]$
2. $C[I, K] = \text{AllReduce}(C[I, K] \\{ U_X\\})$

回答以下问题：

1. 为矩阵$A[N, M]$和$B[M, K]$明确写出此算法，使用索引来准确显示在哪个设备上执行什么计算。假设$A$在ND设备上分片为$A[I, J_X]$，并且您希望您的输出在所有设备上复制。
2. 现在假设您可以接受最终结果不在每个设备上复制，而是分片（跨N或K维度）。上面的算法将如何改变？
3. 纯粹从上述策略的通信成本（在第(b)部分，而不是(a)）来看，此通信成本与我们首先AllGather A然后执行矩阵乘法的算法的通信成本相比如何？

{% details 点击这里查看答案。 %}


1. 首先计算外积，将结果存储在$$O[N, K]: o_{kj} = \sum_i a_{ki} b_{ij}$$中。请注意，重复的索引不是被收缩的索引，因为我们正在执行外积。这里的求和范围跨越存储在我们使用的特定设备上的i值集。因此，例如，如果我们有大小为16的收缩轴和4个设备，那么在设备0上，i的范围将是{0, 1, 2, 3}；在设备1上，i的范围将是{4, 5, 6, 7}；在设备2上，i的范围将是{8, 9, 10, 11}；在设备3上，i的范围将是{12, 13, 14, 15}。然后AllReduce驻留在每个设备上的$O[N, K]$的部分和，以形成完整的$O[N, K]$。
2. 我们可以在步骤2中执行更便宜的ReduceScatter，而不是执行AllReduce，沿任一轴：$[N, K] \\{ U_X \\} \to [N_X, K]$或$[N, K] \\{ U_X \\} \to [N, K_X]$。
3. 如上面主文本中所述，执行AllGather的成本（当我们受吞吐量限制时）与ReduceScatter的成本相同；它只是由我们正在处理的完整矩阵的大小给出。因此，在gather-then-matmul算法中，这按$NM$缩放（因为我们正在$\text{AllGather}$-ing $A$）；在matmul-then-reduce-scatter算法中，这按NK缩放（因为我们正在reduce-scattering $O$）。因此，两种算法的通信成本比率是`M/K`。

{% enddetails %}

**问题10：AllToAll的乐趣：**在上表中，注意到执行AllToAll的时间比执行AllGather或ReduceScatter的时间低4倍（在我们受吞吐量限制的情况下）。在这个问题中，我们将看到这个因子4来自哪里，以及如果我们只有单向ICI链接而不是双向ICI链接，这个因子将如何变化。

1. 让我们首先从单向情况开始。想象我们在环拓扑中有*D*个设备，如果我们正在执行AllGather或ReduceScatter，在N x N矩阵*A*上，它被分片为$A[I_X, J]$（为简单起见，假设$D$整除$N$）。描述这两个集合中涉及的通信，并计算在此算法的整个过程中通过**单个**ICI链接传输的标量（浮点数或整数）的总数。
2. 现在让我们考虑AllToAll，仍然在单向ICI情况下。在这种情况下，算法与all-gather情况有何不同？计算在此算法中通过单个ICI链接传输的标量数。
3. 您应该发现第(a)部分和第(b)部分的答案之间的比率是一个不错的数字。用简单的术语解释这个因子来自哪里。
4. 现在让我们添加双向通信。这如何影响all-gather情况下所需的总时间？
5. 添加双向通信如何影响AllToAll情况下所需的总时间？
6. 现在简单地解释双向环中AllGather时间和AllToAll时间之间的比率。

{% details 点击这里查看答案。 %}

(1) **解决方案：**过程很简单：在算法的每个步骤中，每个设备将向其最近的邻居发送矩阵的单个分片"条"（总大小为$$\frac{N}{D} \times N$$个元素）。这发生$$D-1$$次，因为每个分片需要传达给除了它开始的设备之外的所有设备。因此，总共$$\frac{N^2(D-1)}{D}$$个标量由每个设备传输，即流过单个ICI链接。

**答案：** $$N^2 (1-\frac{1}{D})$$，或者当$$D >> 1$$时简单地$$N^2$$。

(2) **解决方案：**从通信的角度来看，AllToAll和AllGather之间的关键区别在于，在AllToAll中，驻留在特定设备上的分片的全部内容不需要传达给每个其他设备。想象驻留在特定设备（称其为设备0）上的分片是$$[A, B, C, D]$$（这里A、B、C、D是矩阵，我们想象一个有4个设备的环用于说明）。现在矩阵$$A$$不需要在任何地方传达，矩阵$$B$$需要最终在设备1上；矩阵$$C$$最终在设备2上；矩阵$$D$$最终在设备3上。因此，在算法的第一步中，我们将$$B$$、$$C$$和$$D$$发送到设备1；在下一步中，设备1将$$C$$和$$D$$发送到设备2；在最后一步中，设备2只将$$D$$发送到设备3。在这种情况下传输的参数总数是$$(\text{A/B/C/D的大小}) * (3 + 2 + 1)$$。A/B/C/D的大小（在一般情况下）是$$\frac{N^2}{D^2}$$，同样在一般情况下，$$(3 + 2 + 1)$$项变为$$((D-1) + (D-2) + … + 1)$$，或$$\frac{(D)(D-1)}{2}$$。因此，通过单个ICI链接传输的总字节数是$$\frac{N^2(D-1)}{D \times 2}$$。

**答案：** $$\frac{N^2}{2}(1-\frac{1}{D})$$，或者当$$D >> 1$$时简单地$$\frac{N^2}{2}$$。

(3) **解决方案：**因子简单地是$$\frac{1}{2}$$，即AllToAll在单向环拓扑上的成本是all-gather/ReduceScatter的一半。查看上面的推导，这最终来自于在all-gather情况下，我们传输相同大小的块每个$$(D-1)$$次，即我们正在执行求和$$\text{小块大小} * (D + D + D + … + D)$$，而在AllToAll情况下，我们正在执行求和$$\text{小块大小} * (D + D-1 + D-2 + … + 1)$$。因此，二的因子基本上来自于$$1 + 2 + \ldots + n = n(n+1)/2$$的事实。

(4) **解决方案：**任何一个链接必须携带的标量总数现在减少了2倍，因为在双向环中，每个"分片条"可以同时以两种方式发送。

(5) **解决方案：**在这种情况下，与单向情况相比，我们赢得了4倍。通过考虑单个分片条中每个大小为(N2/D2)的块的命运，这最容易看到，比如源自设备0的块。现在，我们不是（如在单向情况下）发送这些块中的一个距离D-1，另一个块距离D - 2等一直到1，而是将条分成向右或向左移动的块，最大移动距离为ceil(D/2)。因此，相应的和现在变为$$D/2 + D/2 - 1 + D/2 - 2 + … = D/2 \cdot (D/2+1)/2$$，或在大$$D$$的极限中$$D^2/8$$。与单向情况下的$$D^2/2$$相比，我们看到我们赢得了4倍。

(6) **解决方案：**在单向环中，我们看到AllToAll时间已经是all-gather时间的两倍；这来自于我们不需要将我们的完整条发送到每个设备的事实。然后，当我们添加双向性时，我们看到AllToAll获得了4倍的胜利，而all-gather只有2倍的胜利。将这些比率放在一起，我们得到了我们寻求的因子4。

{% enddetails %}

<h3 markdown=1 class="next-section">第3部分到此结束！第4部分（关于Transformer数学），点击[这里](../transformers)！</h3>