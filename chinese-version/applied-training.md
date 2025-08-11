---
layout: distill
title: "在TPU上训练LLaMA 3"
# permalink: /main/
description: "让我们仔细看看如何使用我们在前一节中学到的知识在TPU v5p上训练LLaMA 3模型。它们有多大？在不同配置下训练成本如何？它们是如何分片的？让我们通过一些粗略的估算来看看前面几节的内容如何映射到真实模型上。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 6

previous_section_url: "chinese-version/training"
previous_section_name: "第5部分：训练"

next_section_url: "chinese-version/inference"
next_section_name: "第7部分：推理"

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
  - name: "LLaMA 3是什么样的？"
  - name: "计算参数和FLOPs"
  - name: "如何为训练分片LLaMA 3-70B"
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

_我们在本节的目标是将前一节的结果应用到一个非常实际的问题上：训练LLaMA 3系列模型。与前面的章节不同，我们希望你自己完成很多工作。为此，我们隐藏了每个部分的答案，以便你可以先尝试回答。试着拿支笔手工计算！_

### LLaMA 3是什么样的？

LLaMA-3模型系列<d-cite key="llama3"></d-cite>包括3个主要模型：LLaMA 3 8B、70B和405B。我们将主要关注70B，将8B和405B留给你在最后的问题部分探索。这是LLaMA 3-70B的架构，来自LLaMA [HuggingFace页面](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json)。

| **超参数**                  | **值**    |
| --------------------------- | --------- |
| $$n_\text{layers}$$ (L)     | 80        |
| $$d_\text{model}$$ (D)      | 8,192     |
| $$d_{ff}$$ (F)              | 28,672    |
| $$n_\text{heads}$$ (N)      | 64        |
| $$n_\text{kv_heads}$$ (K)   | 8         |
| $$d_\text{qkv}$$ (H)        | 128       |
| $$n_\text{embeddings}$$ (V) | 128,256   |

为了说明找到这些信息有多容易，这里是配置本身，以及映射：

{% include figure.liquid path="assets/img/llama-json.png" class="img-fluid" %}

_为许多不同的开源LLM制作一个包含这些数字的大表格很有用，这样你可以快速比较它们所做的设计决策。_

### 计算参数和FLOPs

**问题：** 从这个表格中，我们能计算出LLaMA 3-70B的参数数量吗？🤫 让我们应用[第4节](../transformers)的内容，看看能否得到70B！

| 参数             | 公式                                                                                                                                              | 数量                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| FFW参数          | d_model * d_ff * 3（用于gelu + 输出投影）* n_layers                                                                                              | 8,192 * 8,192 * 3.5 * 3 * 80 = **56.3e9**                   |
| 词汇表参数       | 2（输入和输出嵌入）* n_embeddings * d_model                                                                                                      | 2 * 128,256 * 8,192 = **2.1e9**                             |
| 注意力参数       | n_layers * [ 2（用于q嵌入和拼接的输出投影）* d_model * n_heads * d_qkv + 2（用于k和v）* d_model * n_kv_heads * d_qkv]                          | 80 * (2 * 8,192 * 64 * 128 + 2 * 8,192 * 8 * 128) = **12e9** |
|                  |                                                                                                                                                   | 56.3e9 + 2.1e9 + 12e9 = **70.4e9**                          |

太好了！我们得到了预期的数字。你会注意到，正如预期的那样，FFW参数完全主导了整体参数数量，尽管注意力部分也不容忽视。

<p markdown=1 class="takeaway">**要点**：MLP块中的3个大权重矩阵比Transformer中的所有其他数组都要大得多，因此在推理模型内存或FLOPs时，我们通常几乎可以忽略所有其他参数。对于LLaMA 3-70B，它们占70B参数中的56B。</p>

现在让我们看看FLOPs！*记住[第4节](../transformers)中的训练通用规则。*

**问题：** LLaMA-3每个token每个训练步骤执行多少FLOPs？_这有助于我们确定整个训练过程的成本。_

{% details 思考后点击这里查看答案！ %}

**答案**：如[第4节](../transformers)所示，我们每个token大约执行$$6 \cdot \text{参数数量}$$的FLOPs，所以这里大约是`6 * 70e9 = 4.2e11` FLOPs / token。这大约是每个token每步半个TFLOP。假设我们受计算限制，在单个TPU v5p芯片上，假设完美的FLOPs利用率，这应该需要大约`4.2e11 / 4.59E+14 = 1ms`。

{% enddetails %}

**问题：** LLaMA 3训练了大约15万亿个token。总共需要多少FLOPs？

{% details 思考后点击这里查看答案！ %}

**答案**：这很简单，就是`4.2e11 * 15e12 = 6.3e24 FLOPs`。6.3 yottaFLOPs。这是很多！在单个TPU上，这将需要`6.3e24 / 4.59E+14 = 435年`。这也是很多！

{% enddetails %}

**问题：** 假设我们想在一个完整的TPU v5p pod上训练，有16x20x28 = 8960个芯片。在bfloat16中以40% MFU训练需要多长时间，假设我们受计算限制？

{% details 思考后点击这里查看答案！ %}

**答案**：我们知道每个TPU v5p可以执行4.59e14 FLOPs/秒。在40% MFU下，这将需要大约`T = 6.3e24 / (8960 * 4.59e14 * 0.4) = 3.8e6秒`。**这大约是44天！** 这相当合理，假设我们实际上可以达到40% MFU。

{% enddetails %}

**问题：** LLaMA 3-70B使用约4M个token的批量大小进行预训练。我们至少需要多少个TPU才能使用这个批量大小进行训练？_你可以假设bfloat16参数和float32优化器状态，并且每层检查点梯度4次。_

{% details 思考后点击这里查看答案！ %}

**答案**：这个问题主要是询问内存使用，因为这是对可用计算的唯一严格限制。在训练期间，我们有三个主要的HBM用途：模型参数、优化器状态和梯度检查点。如果我们假设bfloat16权重、float32优化器状态和一个_非常_保守的梯度检查点方案（每层4次），我们有：

| **参数** | 2 * 70GB | ~140GB |
| **优化器状态** | 8 * 70GB | ~560GB |
| **梯度检查点** | 2 * 8192 * 4e6 * 4 * 80 | ~20.9TB |
| **总计**                |                         | ~21.6TB |

总计约为21.6TB。你会注意到梯度检查点强烈主导了内存图景，即使使用非常保守的检查点方案。我们技术上可以达到每层1个检查点，或者进行微批处理，但这是一个合理的图景。在这些假设下，由于每个TPU v5p有96GB的HBM，我们需要`21.6e12 / 96e9 = 225`个TPU。实际上并不多！

*为什么我们不这样做？* 嗯，因为训练将需要`44天 * 8960 / 225 = 1752天`。这将近四年。**这太多了。** 不过，这清楚地表明我们使用这些大型集群不是因为我们受内存限制，而是因为我们需要额外的FLOPs。

{% enddetails %}

**问题：** 在与上述问题相同的假设下，如果我们使用8960个TPU v5p芯片，我们每个芯片将使用多少内存？

{% details 思考后点击这里查看答案！ %}

**答案**：我们的总内存仍然约为21.6TB，所以每个芯片我们将使用约2.4GB，这基本上什么都不是。如果我们进行更积极的检查点，例如每层12个检查点，我们每个芯片仍然只有8GB。在这些规模的训练中，我们远远没有受到内存限制。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：技术上可以在非常小的拓扑上训练非常大的模型，但需要注意的是它们可能需要很长时间。能够计算训练运行的总FLOPs使我们可以通过假设适度的MFU和已知的拓扑来大致估算其训练时间。</p>

### 如何为训练分片LLaMA 3-70B

让我们坚持上面的设置，假设我们想在8960个芯片的TPU v5p pod上使用4M token批量大小（每批1024个长度为4096的序列）训练LLaMA 3-70B。让我们讨论这个模型的最佳分片策略。

**问题：** 在上述假设下，我们能否仅使用FSDP训练我们的模型？首先，假设我们不能进行任何序列/上下文并行。_这应该是你的第一个想法，因为它简单，如果可行的话不会引入额外的通信。_

{% details 思考后点击这里查看答案！ %}

**答案**：这个答案会有点学究式。如上所述，LLaMA 3-70B最初使用长度为4K的序列进行训练，因此4M token的批量大小给我们一个*序列批量大小*为1024。这意味着我们实际上只能进行纯数据并行/FSDP最多1024个芯片_因为这就是我们有多少序列来进行数据并行_。所以在"完全数据并行且没有额外通信"的简单意义上，答案是否定的。下一个问题将回答这个问题的一个稍微不那么学究的版本。

{% enddetails %}

**问题：** 让我们放宽不进行任何序列分片的要求。如果我们允许自己在批次_和_序列轴上都进行FSDP，我们能否在8960个芯片上仅使用FSDP训练LLaMA 3-70B？

{% details 思考后点击这里查看答案！ %}

**答案**：现在我们允许自己也进行序列/上下文并行，我们可以扩展得更多。首先让我们计算每个设备的批量大小。如果我们进行8960路FSDP，我们最终每个TPU的批量大小为`4 * 1024 * 1024 / 8960 = 468个token`。我们从前一节知道，当$$\text{每设备批量大小} < 2550 / n_\text{axes}$$时，我们会被FSDP的ICI限制。由于我们可以在这里使用完整的3D pod分配3个轴，这将给我们一个下限850，我们远低于此。**所以答案是否定的，即使有3个轴。我们将完全受通信限制。**

{% enddetails %}

**问题：** 现在让我们看看混合张量并行和FSDP。是否存在某种组合让我们保持计算限制？如果有，我们应该进行多少FSDP和张量并行？

{% details 思考后点击这里查看答案！ %}

**答案**：首先让我们检查这是否适合。我们知道，如果我们的每芯片批量大小小于$$2 \cdot 2550^2 / F = 453$$，我们将受通信限制。如上所见，我们略高于此。所以很好！现在要选择最佳的FSDP量，我们可以使用公式

$$X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \cdot 4.19e6 \cdot 8960}{28672}} = 1618$$

四舍五入到合理的2的倍数，这给我们大约2048路FSDP和4路模型并行。这应该工作得很好！

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：我们可以在完整的TPU v5p pod上使用数据并行（1024路）、序列并行（2路）和张量并行（4路）的混合来训练LLaMA-3，批量大小为4M token，而不会受到通信限制。如果我们尝试进行纯FSDP或FSDP +序列并行，我们将受到通信限制。我们在前一节中制定的方程式非常实用。</p>

## 练习题

**问题1 [将LLaMA 70B扩展到更多芯片]：** 假设我们想在4个pod上以相同的批量大小训练LLaMA 3-70B。我们会使用什么并行方案？我们会受到计算还是通信限制？大约需要多长时间来训练？*确保使用正确的屋顶线界限。*

**问题2 [LLaMA 405B]：**

(a) 使用LLaMA 3-405B [配置](https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json)，如上所述编写一个包含所有关键超参数的表格。这个模型有多少总参数？每个训练步骤有多少FLOPs？如果我们训练15T个token，我们执行多少FLOPs？

(b) 假设我们想在8个TPU v5p pod上训练。我们会使用什么并行方案？训练需要多长时间？会受到计算还是通信限制？

<h3 markdown=1 class="next-section">第6节到此结束。要查看关于Transformer推理的第7节，请点击[这里](../inference)。</h3>