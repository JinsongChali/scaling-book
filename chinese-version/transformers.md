---
layout: distill
title: "你需要知道的所有Transformer数学"
# permalink: /main/
description: "这里我们将快速回顾Transformer架构，特别是如何计算FLOP、字节和其他感兴趣的量。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 4

previous_section_url: "../sharding"
previous_section_name: "第3部分：分片"

next_section_url: "../training"
next_section_name: "第5部分：训练"

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

bibliography: main.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "点运算计数"
  - subsections:
    - name: "前向和反向FLOP"
  - name: "Transformer计算"
  - name: "全局FLOP和参数计算"
  - name: "其他数学"
  - subsections:
    - name: "稀疏性和专家混合"
    - name: "梯度检查点"
    - name: "键值(KV)缓存"
  - name: "这一节你应该记住什么？"
  - name: "几个练习题"
  - name: "附录"
  - subsections:
    - name: "附录A：Flash Attention如何工作？"

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

## 点运算计数

让我们从向量$$x$$、$$y$$和矩阵$$A$$、$$B$$开始，它们具有以下形状：

$$
\def \red#1{\textcolor{red}{#1}}
\def \green#1{\textcolor{green}{#1}}
\def \blue#1{\textcolor{blue}{#1}}
\def \purple#1{\textcolor{purple}{#1}}
\def \orange#1{\textcolor{orange}{#1}}
\def \gray#1{\textcolor{gray}{#1}}

\begin{array}{cc}
\textrm{数组}  & \textrm{形状} \\ \hline
x               & \textrm{[P]}   \\
y               & \textrm{[P]}   \\
A               & \textrm{[N P]} \\
B               & \textrm{[P M]} \\
\hline
\end {array}
$$

- $$x \cdot y$$的点积需要$$P$$次_加法_和_乘法_，或总共$$2P$$次浮点运算。
- 矩阵向量乘积$$Ax$$沿$$A$$的行进行$$N$$次点积，共$$2NP$$次FLOP。
- 矩阵矩阵乘积$$AB$$对$$B$$的每列进行$$M$$次矩阵向量乘积，总共$$2NPM$$次FLOP。
- 一般地，如果我们有两个高维数组$$C$$和$$D$$，其中一些维度是<span style="color:red">收缩</span>的，一些是<span style="color:blue">批处理</span>的。（例如$$C[\blue{GH}IJ\red{KL}], D[\blue{GH}MN\red{KL}]$$）那么这个收缩的FLOP成本是所有$$C$$和$$D$$维度乘积的两倍，其中批处理和收缩维度只计算一次，（例如$$2\blue{GH}IJMN\red{KL}$$）。注意只有在两个乘数中都出现的维度才是批处理维度。（还要注意，如果没有收缩维度且这只是元素级乘积，则不会应用因子2。）

$$
\begin{array}{ccc}
\textrm{操作} & \textrm{FLOP} & \textrm{数据} \\
\hline
x \cdot y  & 2P   & 2P      \\
A x        & 2NP  & NP + P  \\
AB         & 2NPM & NP + PM \\
[c_0,...,c_N] \cdot [d_0,...,d_N] &
2 \prod c_i \times \prod_{\substack{d_j \notin \blue{BATCH} \\ d_j \notin \red{CONTRACT}}} d_j
&
  \prod c_i + \prod d_j \\
\hline
\end {array}
$$

请注意，对于矩阵矩阵乘法，*计算*按立方缩放$$O(N^3)$$，而数据传输仅按二次缩放$$O(N^2)$$ \- 这意味着当我们扩大matmul大小时，达到计算饱和限制变得*更容易*。这是极不寻常的，很大程度上解释了为什么我们使用以矩阵乘法为主的架构\-它们适合缩放！

{% include figure.liquid path="assets/img/matmul-flops.gif" class="img-fluid" %}

### 前向和反向FLOP

在训练期间，我们并不特别关心给定矩阵乘法的结果；我们真正关心的是它的导数。这意味着我们在反向传播期间进行了显著更多的FLOP。

如果我们想象**B**只是更大网络中的一个矩阵，**A**是我们的输入激活，**C = A B**，损失**L**相对于**B**的导数由链式法则给出：

$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial B} = A^T \left(\frac{\partial L}{\partial C}\right)$$

这是一个外积，需要$2NPM$次FLOP来计算（因为它在$N$维度上收缩）。类似地，损失相对于**A**的导数是

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial A} = \left(\frac{\partial L}{\partial C}\right) B^T$$

再次是$2NPM$次FLOP，因为**dL/dC**是大小为$$[N, M]$$的（余）向量。虽然这个量不是相对于参数的导数，但它用于计算网络前一层的导数（例如，就像上面使用dL/dC计算dL/dB一样）。

将这些加起来，我们看到**在训练期间，我们总共有6NPM次FLOP**，而推理期间是2NPM：前向传递中2NPM，后向传递中4NPM。由于PM是矩阵中的参数数量，这是著名的$$6 * \text{参数数量} * \text{token数量}$$训练期间Transformer FLOP近似的最简单形式：每个token需要$$6 * \text{参数数量}$$次FLOP。我们将在下面展示更正确的推导。

## Transformer计算

Transformer是未来。嗯，它们至少是现在。也许几年前，它们是许多架构中的一种。但今天，值得了解架构的几乎每个细节。我们不会重新介绍架构，但[这篇博客](https://jalammar.github.io/illustrated-transformer/)和[原始Transformer论文](https://arxiv.org/abs/1706.03762)可能是有用的参考。

这是Transformer解码器架构的基本图：

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图：</b> 这个图显示了标准Transformer的一层，从上到下流动。我们使用单字母约定来描述Transformer中数组的形状和布局，再次以红色显示收缩维度，以蓝色显示批处理维度。在给定操作中，输入形状在左上方给出，参数形状在右上方给出，结果形状在下方，例如BTD是门控einsum的输入形状，DF是权重形状。" %}

**注意[门控einsum]**：上图使用"[门控einsums](https://arxiv.org/abs/2002.05202)"<d-cite key="glu"></d-cite>，其中我们将up-projection矩阵分成两个矩阵（上面的$W_\text{In1}$和$W_\text{In2}$），它们的输出进行元素级乘法作为一种"门控函数"。不是所有的LLM都使用这个，所以你有时会看到单个$W_\text{In}$矩阵和总MLP参数数量为2DF而不是3DF。通常在这种情况下，D和F会被放大以保持参数数量与3矩阵情况相同。话虽如此，某种形式的门控einsum被LLAMA、DeepSeek和许多其他模型使用。

**注意2[MHA注意力]**：对于自注意力，T和S是相同的，但对于交叉注意力它们可能不同。对于普通的多头注意力（MHA），N和K相同，而对于[多查询注意力](https://arxiv.org/abs/1911.02150)（MQA）<d-cite key="mqa"></d-cite> K=1，对于[分组MQA](https://arxiv.org/abs/2305.13245)（GMQA）<d-cite key="gmqa"></d-cite> K只需要整除N。

## 全局FLOP和参数计算

对于下面的内容，我们将计算每层的FLOP以避免在所有地方都粘贴**L**因子。

### MLP

Transformer的MLP通常由2个输入matmul组成，它们进行元素级组合，以及一个输出matmul：

$$
\begin{array}{ccc}
\textrm{操作} & \textrm{训练FLOP} & \textrm{参数} \\
\hline \\
A[B,T,\red{D}] \cdot W_{in1}[\red{D}, F] & 6BTDF & DF \\[10pt]
A[B,T,\red{D}] \cdot W_{in2}[\red{D}, F] & 6BTDF & DF \\[10pt]
\sigma\left(A_{in1}\right)[B,T, F] * A_{in2}[B,T, F] & \gray{O(BTF)} \\[10pt]
A[B,T,\red{F}] \cdot W_{out}[\red{F}, D] & 6BTDF & DF \\[10pt]
\hline \\
& \approx 18BTDF & 3DF
\end{array}
$$

### 注意力

对于具有不同**Q**和**KV**头数的通用分组查询注意力情况，让我们假设**Q**、**K**、**V**投影的头维度H相等，并估算**QKVO** matmul的成本：

$$
\begin{array}{ccc}
\textrm{操作} & \textrm{训练FLOP} & \textrm{参数} \\
\hline \\
A[B,T,\red{D}] \cdot W_{Q}[\red{D}, N, H] & 6BTDNH & DNH \\[10pt]
A[B,T,\red{D}] \cdot W_{K}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{D}] \cdot W_{V}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{N}, \red{H}] \cdot W_{O}[\red{N}, \red{H}, D] & 6BTDNH & DNH \\[10pt]
\hline \\ & 12BTD(N+K)H & 2D(N+K)H
\end{array}
$$

点积注意力操作更微妙，实际上是一个$$TH \cdot HS$$ matmul，在$$B$$、$$K$$维度上批处理，一个softmax，以及一个$$TS \cdot SH$$ matmul，再次在$$B$$、$$K$$维度上批处理。我们用蓝色突出显示批处理维度：

$$
\begin{array}{cc}
\textrm{操作} & \textrm{训练FLOP} \\
\hline \\[3pt]
Q[\blue{B}, T, \blue{K}, G, \red{H}] \cdot K[\blue{B}, S, \blue{K}, \red{H}]
& 6BTSKGH = 6BTSNH  \\[3pt]
\textrm{softmax}_S \;\; L[B, T, S, K, G] & \gray{O(BTSKG) = O(BTSN)} \\[3pt]
S[\blue{B}, T, \red{S}, \blue{K}, G] \cdot V[\blue{B}, \red{S}, \blue{K}, H] 
& 6BTSKGH = 6BTSNH \\[3pt]
\hline \\
& \approx 12BTSNH = 12BT^2NH \\
\end{array}
$$

### 其他操作

Transformer中还有其他几个操作。Layernorm相对便宜，可以在一阶成本估算中忽略。还有最终巨大的（虽然不是每层的）反嵌入矩阵乘法。

$$
\begin{array}{ccc}
\textsf{操作} & \textsf{训练FLOP} & \textsf{参数} \\
\hline \\
\textrm{layernorm}_D \;\; A[B,T,\red{D}] & \gray{O\left(BTD\right)} & \gray{D} \\[10pt]
A[B,T,\red{D}] \cdot W_{unembed}[\red{D}, V] & 6BTDV & DV \\
\end{array}
$$

### Transformer FLOP的一般经验法则

如果我们忽略较短上下文训练中点积注意力的成本，那么所有层的总FLOP是

$$
\begin{align*}
(18BTDF + 12BTD(N+K)H)L = 6 *BT * (3DF + 2D(N+K)H)L \\ = 6 * \textrm{token数量} * \textrm{参数数量}
\end{align*}
$$

这导致了估算密集Transformer FLOP数量的著名经验法则，忽略了注意力FLOP。（反嵌入是另一个简单的matmul，有$6BSDV$次FLOP和$DV$个参数，遵循相同的经验法则。）

### 注意力随上下文长度的分数成本

如果我们确实考虑上面的点积注意力，并假设$$F=4D$$、$$D=NH$$（如通常情况）和$$N=K$$：

$$\small{\frac{\textrm{注意力FLOP}}{\textrm{matmul FLOP}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{12BT^2D}{4*18 BTD^2 + 24 BTD^2} = \frac{12BT^2D}{96 BTD^2} = \frac{T}{8D}}$$

所以要点是**点积注意力FLOP只有在训练期间T>8D时才占主导地位**。对于D ~ 8k，这将是~64K个token。这是有道理的，因为它意味着随着MLP大小增加，注意力FLOP变得不那么关键。对于大模型，注意力的二次成本实际上并不是更长上下文训练的巨大障碍。然而，对于较小的模型，甚至例如Gemma-27B，D=4608，这意味着注意力在大约32k序列长度时变得占主导地位。Flash Attention也有助于减轻长上下文的成本，我们在[附录A](#appendix-a-how-does-flash-attention-work)中简要讨论。

## 其他数学

### 稀疏性和专家混合

我们不得不简要讨论专家混合（MoE）模型<d-cite key="moe"></d-cite>，它用一组可以动态路由的独立MLP替换标准Transformer中的单个密集MLP块。作为一阶近似，**MoE就是每层有E个MLP块的普通密集模型**，而不只是一个。每个token激活这些专家中的$k$个，通常$k=2$。这将参数数量增加$O(E)$，同时将每个token的激活参数总数与密集版本相比乘以$k$。

{% include figure.liquid path="assets/img/moe.png" class="img-fluid img-small" caption="<b>图：</b> 一个有$n$个专家的MoE层示例。门控专家将每个token路由到其中的$k$个，那$k$个MLP的输出被求和。我们的参数数量是每个专家大小的$n$倍，但每个token只使用$k$个。<a href=\"https://deepgram.com/learn/mixture-of-experts-ml-model-guide\">来源</a>。" %}

与密集模型相比，MoE引入了新的通信，主要是两个AllToAll（MoE块之前和之后各一个），将token路由到正确的专家并将它们带回到其主设备。<d-footnote>严格来说，这只有在我们沿与专家相同的轴进行数据或序列分片时才会发生。</d-footnote> 然而，正如我们在前一节中看到的，每个AllToAll的成本只是沿单轴可比AllGather的1/4（对于双向环）。

### 梯度检查点

反向传播作为一种算法用内存换取计算。反向传递不需要$$O(n_\text{layers}^2)$$次FLOP，**它需要$$O(n_\text{layers})$$内存**，保存前向传递期间生成的所有中间激活。虽然这比二次计算更好，但在内存方面极其昂贵：一个有$$B * T=4M$$（每批总共4M个token）、L=64和D=8192的模型，如果避免所有不必要的反向传递计算，将必须在bfloat16中保存大约$$2 * 20 * B * T * D * L = 84TB$$的激活。20来自（粗略地）计算上面Transformer图中的每个中间节点，因为例如

$$f(x) = \exp(g(x))$$

$$\frac{df}{dx} = \exp(g(x)) \cdot \frac{dg}{dx}$$

所以为了避免重新计算，我们需要从前向传递保存$$g(x)$$和$$\exp(g(x))$$。为了避免保存这么多内存，我们可以选择只保存一部分中间激活。这里有几种我们使用的策略。

* **块重计算**：只保存每层的输入。这是我们使用的最激进的方法，每层只保存1个检查点，意味着在上面的例子中我们只会保存4.2TB。这迫使我们在反向传递中重复基本上所有的前向传递FLOP，意味着我们将FLOP从$$6ND$$增加到大约$$8ND$$。
* **仅大矩阵乘法：** 另一个简单的策略是只保存大矩阵乘法的输出。这让我们避免在反向传递期间重新计算任何大矩阵乘法，但仍然让我们重新计算其他激活函数和注意力的部分。这将每层的20减少到接近7。

这绝不是全面的。使用JAX时，这些通常由`jax.remat`/`jax.checkpoint`控制（你可以在[这里](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)阅读更多）。

### 键值(KV)缓存

正如我们将在[第7节](../inference)中看到的，LLM推理有两个关键部分，预填充和生成。

* **预填充**处理长提示并将其注意力激活保存在键值缓存（KV缓存）中以供生成使用，特别是注意力块中的键值投影。
* **生成**将其中几个KV缓存批处理在一起并从每个采样token。

每个KV缓存实际上是一个大小为$[2, S, L, K, H]$的数组，其中2表示键和值。这相当大！int8中键值缓存的总大小是$2SLKH$。对于一个中等大小的模型，有8k上下文长度，64层，以及$KH = NH = D = 8192$，这是$2 \cdot 8192 \cdot 64 \cdot 8192 = 8\text{GiB}$。你可以看到为什么我们希望使用$K \ll N$的GMQA。

## 这一节你应该记住什么？

* Transformer的总体参数和FLOP相当容易计算，在这里总结，假设MHA（批大小B，词汇大小V，长度为T的序列，D=d<sub>model</sub>，F=d<sub>ff</sub>）：


<!-- $$
\begin{array}{ccc}
\textrm{组件} & \textrm{每层参数} & \textrm{每层训练FLOP} \\
\hline \\
\textbf{MLP} & 3DF & 18BTDF \\[10pt]
\textbf{注意力} & 4DNH & 24BTDNH + 12BT^2NH \\[10pt]
\textbf{其他} & D & BTD \\[10pt]
\textbf{词汇} & DB \text{ (总共，不是每层)} & 12BTDV \\[10pt]
\end{array}
$$ -->


| 组件         | 每层参数                    | 每层训练FLOP                    |
| :---------- | :------------------------ | :---------------------------- |
| **MLP**     | 3DF                       | 18BTDF                        |
| **注意力**   | 4DNH                      | 24BTDNH \+ 12BT<sup>2</sup>NH |
| **其他**     | D                         | BTD                           |
| **词汇**     | DV (总共，不是每层)          | 12BTDV                        |

* MLP块的参数数量主导总参数数量，只要序列长度$T < 8D$，MLP块也主导FLOP预算。
* 训练期间的总FLOP预算在合理的上下文长度下很好地近似为$$6 \cdot \text{参数数量} \cdot \text{token数量}$$。
* 在推理期间，我们的KV缓存每个缓存大约是$$2 \cdot S \cdot L \cdot N \cdot H$$，尽管架构修改通常可以减少这个。

## 几个练习题

**问题1：** 一个有$D=4096$、$F=4 \cdot D$、$V=32,000$和$L=64$的模型有多少参数？其中注意力参数占多少比例？每个token的KV缓存有多大？*你可以假设$N\cdot H=D$和多头注意力，int8 KV。*

{% details 点击这里查看答案。 %}

1. 总参数大约是$$L \cdot (3DF + 4DNH + D) + 2DV$$。对于给定的数字，这是$$64 \cdot (3 \cdot 4e3 \cdot 16e3 + 4 \cdot 4e3 \cdot 4e3 + 4e3) + 2 \cdot 4e3 \cdot 32e3 = 16e9$$，或16B参数。
2. 注意力参数与总参数的比例一般是$$4DNH / (4DNH + 3DF) = 4D^2 / (4D^2 + 12D^2) = 1/4$$。这给我们大约1/4的参数用于注意力。
3. 每个token，我们的KV缓存是int8中的$$2 \cdot L \cdot N \cdot H = 2 \cdot 64 \cdot 4096$$，即`512kB / token`。

{% enddetails %}

**问题2：** 在`{'X': 4, 'Y': 8, 'Z': 4}`上执行A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]需要多少总FLOP？每个TPU执行多少FLOP？

{% details 点击这里查看答案。 %}

操作的总"理论"FLOP是$$2 \cdot B \cdot D \cdot F$$。然而，因为计算没有沿Z维度分片，我们实际上在做Z次额外FLOP，意味着$$2 \cdot B \cdot D \cdot F \cdot Z$$总FLOP。由于计算沿其他维度分片，每个设备的总计大约是$$2 \cdot B \cdot D \cdot F / (X \cdot  Y)$$。

{% enddetails %}

**问题3：** 执行$A[I,J,K,L] * B[I,J,M,N,O] \rightarrow C[K,L,M,N,O]$涉及多少FLOP？

{% details 点击这里查看答案。 %}

根据上面的规则，我们有I和J作为收缩维度，K、L、M、N和O作为非收缩维度。我们没有"批处理维度"，所以这只是$$2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O$$，所有轴的和。如果我们有共享轴，它只会被计算一次。

{% enddetails %}

**问题4：** 自注意力（忽略Q/K/V/O投影）的算术强度是多少？*将答案给出为Q和KV长度T和S的函数。* 在什么上下文长度下注意力是FLOP受限的？给定我们TPU的HBM带宽，绘制注意力相对于FFW块的有效相对成本随上下文长度增长的图。

{% details 点击这里查看答案。 %}

自注意力需要加载$$Q$$、$$K$$和$$V$$激活，然后计算$$\text{softmax}(Q \cdot K) \cdot V$$，然后将结果写回HBM。这将用Flash Attention完成，所以这个数学有一些注意事项，但基本上在bf16中自注意力执行

$$\text{Q[B,T,N,H]} \rightarrow_\text{reshape} \text{Q[B, T, K, G, H]} \cdot \text{K[B, S, K, H]} \rightarrow \text{O[B, T, S, K, G]}$$

$$U=\text{softmax}_S(\text{O[B, T, S, K, G]})$$

$$\text{U[B, T, S, K, G]} \cdot \text{V[B, S, K, H]} \rightarrow \text{X[B, T, K, G, H]}$$

所以我们的总字节数是$$2 * \text{sizeof}(Q) + 2 * \text{sizeof(K or V)} = 4BTNH + 4BSKH = 4BHK * (TG + S)$$，总FLOP是$$4BTSNH + O(BTSN)$$，算术强度是$$4BTSKGH / (4BHK * (TG + S))$$。

所以基本上，在预填充期间我们有$$S=T$$，所以我们有算术强度$$4BT^2KGH / 4BHKT \cdot (G+1) = TG/(G + 1) = O(T)$$。在生成期间，$$T=1$$，所以我们有$$4BSKGH / (4BHK \cdot (G + S)) = SG / (G + S) \rightarrow G$$，假设$$S$$非常大。根据你如何解释问题，在预填充或训练期间，假设没有序列分片，自注意力在S=240时是计算受限的。在生成期间，我们永远不会计算受限，因为$$G$$很小。尽管如此，你可以看到增加$$G$$导致我们更接近计算受限。

{% enddetails %}

**问题5：** 在什么序列长度下自注意力FLOP等于QKVO投影FLOP？

{% details 点击这里查看答案。 %}

这纯粹是$$24BTDNH == 12BT^2NH$$的问题。简化我们得到$$2D = T$$，所以例如对于$$D=4096$$，这是$$8192$$。这告诉我们对于大多数合理的上下文长度，matmul FLOP更大。

{% enddetails %}

**问题6：** 假设我们在前向传递期间只保存Transformer层中7个主要matmul的输出（Q、K、V、O \+ 三个FFW矩阵）。我们需要多少额外的FLOP来在反向传递期间"重新计算"？

**问题7：** DeepSeek v3说它在14.8T token上训练了2.79M H800小时（[来源](https://arxiv.org/pdf/2412.19437v1)）。鉴于它有37B激活参数，他们大概实现了什么硬件利用率？*提示：注意他们使用了没有结构化稀疏性的FP8 FLOP。*

{% details 点击这里查看答案。 %}

从[这里](https://lenovopress.lenovo.com/lp1814.pdf)的规格表，我们找到3,026 TFLOP/s的带稀疏性FP8性能，或者通常是这个的一半（`1.513e15` FLOP/s）没有稀疏性。2.79M H800小时意味着`2.79e6 * 1.513e15 * 60 * 60 = 1.52e25`总FLOP。给定37B的激活参数数量，这个训练运行应该使用大约`6 * 37e9 * 14.8e12 = 3.3e24` FLOP。这意味着FLOP利用率大约是`3.3e24 / 1.52e25 = 21.7%`。

{% enddetails %}

**问题8：** 专家混合（MoE）模型有标准密集MLP块的$E$个副本，每个token激活其中的$k$个专家。在TPU v5e上使用int8权重的MoE需要多大的token批次大小才能成为计算受限？对于有256个（路由）专家和$k=8$的DeepSeek，这个数字是多少？

{% details 点击这里查看答案。 %}

因为我们有每个专家的$E$个副本，在int8中，我们需要加载$E \cdot D \cdot F$字节。因为每个token激活$k$个专家，我们有$2\cdot k \cdot B \cdot D \cdot F$ FLOP。要在bfloat16 FLOP下成为计算受限，我们需要超过240的算术强度，当$(2\cdot k \cdot BDF) / EDF > 240$或$k \cdot B / E > 120$时发生。

因此，我们需要$B > 120 \cdot E / k$才能成为计算受限。对于DeepSeek，这给我们$B > 120 \cdot 256 / 8 = 3840$。这在生成时间是一个非常大的批次大小。

{% enddetails %}

<h3 markdown=1 class="next-section">第4部分就到这里！对于第5部分（关于扩展Transformer训练），[点击这里](../training)！</h3>

## 附录

### 附录A：Flash Attention如何工作？

将Transformer扩展到非常长上下文的传统反对意见是注意力FLOP和内存使用随上下文长度二次缩放。虽然注意力QK乘积确实有形状$[B, S, T, N]$，其中B是批大小，S和T是Q和K序列维度，N是头数，但这个说法带有一些严重的注意事项：

1. 正如我们在第4节中指出的，即使这是二次的，只有当$$S > 8 \cdot D$$时注意力FLOP才占主导地位，特别是在训练期间，单个注意力矩阵的内存与生活在内存中的所有权重和激活检查点相比很小，特别是在分片时。
2. 我们不需要物化完整的注意力矩阵来计算注意力！我们可以计算局部和和最大值，避免物化超过数组的小块。虽然总FLOP仍然是二次的，但我们大大减少了内存压力。

第二个观察首先由[Rabe等人2021](https://arxiv.org/abs/2112.05682)提出，后来在[Flash Attention论文](https://arxiv.org/abs/2205.14135)（Dao等人2022）中提出。基本思想是以K/V块的形式计算注意力，我们计算局部softmax和一些辅助统计量，然后将它们传递给下一个块，将它们与其局部块结合。具体地，我们计算

1. **M：** 在序列维度上$$q \cdot k$$的运行最大值
2. **O：** 在序列维度上的运行完整注意力softmax
3. **L：** 运行分母$$\sum_i (q \cdot k_i - \text{运行最大值})$$

有了这些，我们可以只用恒定数量的内存计算新的最大值、新的运行和以及新的输出。为了粗略描述这是如何工作的，注意力大致是这个操作：

$$\text{Attn}(Q, K, V) = \sum_i \frac{\exp(Q \cdot K_i - \max_j Q \cdot K_j) V_i}{\sum_l \exp(Q \cdot K_l - \max_j Q \cdot K_j)}$$

为了数值稳定性减去最大值，可以在不影响结果的情况下添加，因为$$\sum_i \exp(a_i + b) = \exp(b) \sum \exp(a)$$。只看上面的分母，如果我们想象有两个连续的键向量块，$$K^1$$和$$K^2$$，我们为每个计算局部softmax和$$L^1$$和$$L^2$$

$$L^1 = \sum_i \exp(Q \cdot K_i^1 - \max_j Q \cdot K_j^1)$$

$$L^2 = \sum_i \exp(Q \cdot K_i^2 - \max_j Q \cdot K_j^1)$$

然后我们可以使用以下方法将这些组合成这两个块一起的完整softmax和：

$$L^\text{combined} = \exp(M^1 - \max(M^1, M^2)) \cdot L^1 + \exp(M^2 - \max(M^1, M^2)) \cdot L^2$$

其中

$$M^1 = \max_j Q \cdot K_j^1 \text{ and } M^2 = \max_j Q \cdot K_j^2$$

这也可以对完整softmax进行，给我们一种累积任意大softmax和的方法。这里是来自Flash Attention论文的完整算法。

{% include figure.liquid path="assets/img/flash-algo.png" class="img-fluid" %}

从硬件角度来看，这让我们将Q的块放入VMEM（上面算法称为片上SRAM），所以我们只需要在每次迭代时加载KV块，减少算术强度。我们也可以将运行统计量保持在VMEM中。

值得强调的最后一个微妙点是用于使Flash VJP（反向模式导数）计算对训练实用的注意力softmax属性。如果我们将中间softmax数组定义为：

$$S_{ij} = \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_j}}$$

在注意力中，我们从反向模式*dO*和*V*数组获得*dS*：

$$dS_{ij} = dO_{id} \cdot_d V_{jd} = \sum_d dO_{id} V_{jd}$$

在这个梯度反向传播到Q和K期间

$$d(q_i \cdot k_j) = (dS_{ij} - S_{ij} \cdot_j dS_{ij}) S_{ij}$$

我们利用一个恒等式，允许我们用沿特征**深度**维度的局部收缩交换沿大键**长度**维度的收缩。

$$\begin{align*}
S_{ij} \cdot_j dS_{ij} &= \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} \sum_d dO_{id} V_{jd} \\
&= \sum_d dO_{id} \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} V_{jd} \\
&= \sum_d dO_{id} O_{id} \\
&= dO_{id} \cdot_d O_{id}
\end{align*}$$

这个替换对于能够为VJP实现序列块*局部*计算至关重要，并启用了进一步巧妙的分片方案，如环形注意力。