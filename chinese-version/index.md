---
layout: distill
title: "如何扩展你的模型"
subtitle: '从系统视角看 TPU 上的大语言模型'
# permalink: /main/
description: "训练大语言模型常常感觉像炼金术，但理解和优化模型性能并不必如此。本书旨在揭开语言模型扩展的科学：TPU（和 GPU）如何工作以及它们如何相互通信，大语言模型如何在真实硬件上运行，以及如何在训练和推理期间并行化您的模型，使其在大规模上高效运行。如果您曾经想知道"训练这个大语言模型应该花多少钱"或"我需要多少内存来自己服务这个模型"或"什么是 AllGather"，我们希望这本书对您有用。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

giscus_comments: true

section_number: 0

previous_section_url: ""
previous_section_name: "第 0 部分：简介"

next_section_url: "../roofline"
next_section_name: "第 1 部分：Roofline 分析"

bibliography: main.bib

citation: true

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
  - name: 高层概述
  - name: 章节链接

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

{% include figure.liquid path="assets/img/dragon.png" class="img-fluid" %}

深度学习的大部分内容仍然归结为某种黑魔法，但优化模型性能并不必如此——即使在巨大规模下！相对简单的原则适用于各个地方——从处理单个加速器到数万个加速器——理解它们可以让您做许多有用的事情：

- 大致估算模型各部分距离其理论最优值有多近。
- 在不同规模下对不同并行方案做出明智的选择（如何在多个设备之间分割计算）。
- 估算训练和运行大型 Transformer 模型所需的成本和时间。
- 设计利用[特定](https://arxiv.org/abs/2205.14135)[硬件](https://arxiv.org/abs/1911.02150)[特性](https://arxiv.org/abs/2007.00072)的算法。
- 基于对当前算法性能限制的明确理解来设计硬件。

**预期背景：** 我们假设您对大语言模型和 Transformer 架构有基本了解，但不一定了解它们如何在规模上运行。您应该了解大语言模型训练的基础知识，理想情况下对 JAX 有一些基本的熟悉。一些有用的背景阅读可能包括[这篇关于 Transformer 架构的博客文章](https://jalammar.github.io/illustrated-transformer/)和[原始 Transformer 论文](https://arxiv.org/abs/1706.03762)。另外，查看[这个列表](conclusion#further-reading)以获取更多有用的并发和未来阅读材料。

**目标与反馈：** 到最后，您应该能够轻松估算给定硬件平台上 Transformer 模型的最佳并行方案，以及训练和推理大概需要多长时间。如果您做不到，请给我们发邮件或留下评论！我们很想知道如何使这更清晰。

### 为什么您应该关心？

三四年前，我认为大多数机器学习研究人员不需要理解本书中的任何内容。但今天，即使是"小"模型也运行得如此接近硬件极限，以至于进行新颖的研究需要您考虑规模上的效率。<d-footnote>从历史上看，机器学习研究遵循着系统创新和软件改进之间的某种滴答作响的周期。Alex Krizhevsky 必须编写邪恶的 CUDA 代码来使 CNN 快速运行，但在几年内，像 Theano 和 TensorFlow 这样的库意味着您不必这样做。也许这里也会发生同样的事情，本书中的所有内容将在几年内被抽象化。但扩展定律已将我们的模型永久推向硬件的最前沿，在不久的将来，进行尖端研究似乎将与理解如何有效地将模型扩展到大型硬件拓扑密不可分。</d-footnote> **如果在 roofline 效率上损失 20%，那么在基准测试上获得 20% 的胜利就是无关紧要的。** 有前途的模型架构经常失败，要么是因为它们*无法*在规模上高效运行，要么是因为没有人投入工作使它们这样做。

**"模型扩展"的目标是能够增加用于训练或推理的芯片数量，同时实现成比例的线性吞吐量增长。** 这被称为"*强扩展*"。尽管添加额外的芯片（"并行性"）通常会减少计算时间，但它也会带来芯片之间额外通信的成本。当通信花费的时间比计算更长时，我们变得"通信受限"并且无法强扩展。<d-footnote>随着计算时间的减少，您通常也会在单个芯片级别面临瓶颈。您闪亮的新 TPU 或 GPU 可能被评定为每秒执行 500 万亿次操作，但如果您不小心，如果它在内存中移动参数时陷入困境，它同样可以轻松地只执行十分之一。每芯片计算、内存带宽和总内存的相互作用对扩展故事至关重要。</d-footnote> 如果我们足够了解我们的硬件以预测这些瓶颈将在哪里出现，我们可以设计或重新配置我们的模型以避免它们。<d-footnote>硬件设计师面临着相反的问题：构建硬件，为我们的算法提供足够的计算、带宽和内存，同时最小化成本。您可以想象这个"协同设计"问题有多么压力：您必须押注当第一批芯片实际可用时算法会是什么样子，通常是 2 到 3 年后。TPU 的故事是这场游戏中的巨大成功。矩阵乘法是一种独特的算法，因为它每字节内存使用的 FLOP 比几乎任何其他算法都多（每字节 N 个 FLOP），早期的 TPU 及其脉动阵列架构在构建时实现了比 GPU 更好的性能/美元比。TPU 是为机器学习工作负载设计的，而带有 TensorCore 的 GPU 正在迅速改变以填补这一利基。但您可以想象，如果神经网络没有起飞，或者以某种 TPU（本质上比 GPU 更不灵活）无法处理的根本方式发生了变化，成本会有多高。</d-footnote>

*我们在本书中的目标是解释 TPU（和 GPU）硬件如何工作，以及 Transformer 架构如何演变以在当前硬件上表现良好。我们希望这对设计新架构的研究人员和努力使当前一代大语言模型快速运行的工程师都有用。*

## 高层概述

本书的整体结构如下：

[第 1 节](roofline)解释了 roofline 分析以及哪些因素可能限制我们的扩展能力（通信、计算和内存）。[第 2 节](tpus)和[第 3 节](sharding)详细讨论了 TPU 和现代 GPU 的工作原理，既作为单个芯片，也作为——至关重要的——具有有限带宽和延迟的芯片间链接的互连系统。我们将回答如下问题：

* 某个大小的矩阵乘法应该花多长时间？在什么时候它受计算或内存或通信带宽的限制？
* TPU 如何连接在一起形成训练集群？系统的每个部分有多少带宽？
* 在多个 TPU 上收集、分散或重新分配数组需要多长时间？
* 我们如何有效地乘以在设备上分布不同的矩阵？

{% include figure.liquid path="assets/img/pointwise-product.gif" class="img-small" caption="<b>图：</b> 来自<a href=\"tpus\">第 2 节</a>的图表，显示 TPU 如何执行逐元素乘积。根据我们数组的大小和各种链接的带宽，我们可能发现自己是计算受限（使用完整的硬件计算能力）或通信受限（受内存加载瓶颈限制）。" %}

五年前，机器学习有着丰富多彩的架构景观——ConvNet、LSTM、MLP、Transformer——但现在我们主要只有 Transformer<d-cite key="transformers"></d-cite>。我们坚信值得理解 Transformer 架构的每一部分：每个矩阵的确切大小、归一化发生的位置、每个部分有多少参数和 FLOP<d-footnote>浮点运算，基本上是所需的加法和乘法的总数。虽然许多来源将 FLOP 理解为"每秒操作数"，但我们使用 FLOP/s 来明确表示。</d-footnote>。[第 4 节](transformers)仔细介绍了这种"Transformer 数学"，展示了如何计算训练和推理的参数和 FLOP。这告诉我们模型将使用多少内存，我们将在计算或通信上花费多少时间，以及注意力相对于前馈块何时变得重要。

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图：</b> 标准 Transformer 层，每个矩阵乘法（matmul）显示为圆圈内的点。所有参数（不包括规范）以紫色显示。<a href=\"transformers\">第 4 节</a>更详细地介绍了这个图表。" %}

[第 5 节：训练](training)和[第 7 节：推理](inference)是本文的核心，我们在这里讨论基本问题：给定某个大小的模型和某个数量的芯片，我如何并行化我的模型以保持在"强扩展"范围内？这是一个简单的问题，但答案却出人意料地复杂。在高层次上，有 4 种主要的并行技术用于在多个芯片上分割模型（**数据**、**张量**、**管道**和**专家**），以及许多其他技术来减少内存需求（**重新材料化**、**优化器/模型分片（又名 ZeRO）**、**主机卸载**、**梯度累积**）。我们在这里讨论其中的许多。

我们希望在这些部分结束时，您应该能够自己为新的架构或设置选择它们。[第 6 节](applied-training)和[第 8 节](applied-inference)是应用这些概念到 LLaMA-3（一个流行的开源模型）的实用教程。

最后，[第 9 节](profiling)和[第 10 节](jax-stuff)研究如何在 JAX 中实现其中一些想法，以及当出现问题时如何分析和调试您的代码。

在整个过程中，我们尝试为您提供自己工作的问题。请不要感到有压力阅读所有部分或按顺序阅读它们。请留下反馈。目前，这是一份草稿，将继续修订。谢谢！

*我们要感谢 James Bradbury 和 Blake Hechtman，他们衍生了本文档中的许多想法。*

<h3 markdown=1 class="next-section">话不多说，[这里是关于 TPU rooflines 的第 1 节](roofline)。</h3>

## 章节链接

*这个系列可能比需要的更长，但我们希望这不会阻止您。前三章是预备知识，如果熟悉可以跳过，尽管它们介绍了后面使用的符号。最后三部分可能是最实用的，因为它们解释了如何使用真实模型。*

**第 1 部分：预备知识**

* [**第 1 章：Roofline 分析简介**](roofline)。算法受三个因素限制：计算、通信和内存。我们可以使用这些来近似我们的算法运行速度。

* [**第 2 章：如何理解 TPU**](tpus)。TPU 如何工作？这如何影响我们可以训练和服务的模型？

* [**第 3 章：分片矩阵及其乘法**](sharding)。在这里，我们通过我们最喜欢的操作来解释模型分片和多 TPU 并行性：（分片的）矩阵乘法。

**第 2 部分：Transformers**

* [**第 4 章：您需要了解的所有 Transformer 数学**](transformers)。Transformer 在其前向和后向传递中使用多少 FLOP？您能计算参数数量吗？KV 缓存的大小？我们在这里研究这些数学。

* [**第 5 章：如何并行化 Transformer 进行训练**](training)。FSDP。Megatron 分片。管道并行性。给定一定数量的芯片，我如何尽可能高效地训练给定大小和批量大小的模型？

* [**第 6 章：在 TPU 上训练 LLaMA 3**](applied-training)。我们如何在 TPU 上训练 LLaMA 3？需要多长时间？成本是多少？

* [**第 7 章：关于 Transformer 推理的一切**](inference)。一旦我们训练了一个模型，我们就必须服务它。推理增加了一个新的考虑因素——延迟——并改变了内存格局。我们将讨论分解服务的工作原理以及如何考虑 KV 缓存。

* [**第 8 章：在 TPU 上服务 LLaMA 3**](applied-inference)。在 TPU v5e 上服务 LLaMA 3 的成本是多少？延迟/吞吐量权衡是什么？

**第 3 部分：实用教程**

* [**第 9 章：如何分析 TPU 代码**](profiling)。真实的大语言模型从来没有上述理论那么简单。在这里，我们解释 JAX + XLA 堆栈以及如何使用 JAX/TensorBoard 分析器来调试和修复实际问题。

* [**第 10 章：在 JAX 中编程 TPU**](jax-stuff)。JAX 提供了一堆用于并行化计算的神奇 API，但您需要知道如何使用它们。有趣的例子和解决的问题。

* [**第 11 章：结论和进一步阅读**](conclusion)。关于 TPU 和大语言模型的结束语和进一步阅读。