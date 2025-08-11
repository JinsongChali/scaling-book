---
layout: distill
title: "结论与延伸阅读"
# permalink: /main/
description: "感谢您的阅读！这里我们将包含一些进一步研究的参考资料。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 11

previous_section_url: "chinese-version/jax-stuff"
previous_section_name: "第10部分：JAX"

next_section_url: "chinese-version/gpus"
next_section_name: "第12部分：GPU"

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
  - name: "致谢"
  - name: "延伸阅读"
  - name: "反馈"

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
  .algorithm {
    padding: 10px;
    margin-top: 5px;
    margin-bottom: 5px;
    border-style: dashed;
    background-color: #fffaf2;
  }

  .algorithm li {
    margin-bottom: 0px;
  }
---

**感谢您阅读这套文章，祝贺您一路读到最后。** 在结束之前，让我们简短地致谢：

## 致谢

这份文档代表了Google DeepMind许多人的重大集体投入，我们想简要致谢！

* James Bradbury、Reiner Pope和Blake Hechtman最初推导了这份手稿中的许多想法，并且较早地理解了Transformer的系统视图。
* Sholto Douglas写了这份文档的第一个版本，并负责启动该项目。他比任何人都更负责这份文档的整体叙述。
* Jacob Austin领导了将这个第一版本从粗略笔记转换为更精良和综合产物的工作。他完成了编辑、格式化和发布这份文档的大部分工作，并协调了其他作者的贡献。
* 大部分图形和动画由Anselm Levskaya和Charlie Chen制作。
* Charlie Chen写了推理部分并绘制了许多推理图形。
* Roy Frostig帮助了发布、编辑和旅程的许多其他步骤。

我们还要感谢在整个过程中给予关键反馈的许多其他人，特别是Zak Stone、Nikhil Sethi、Caitlin Stanton、Alex Dimitriev、Sridhar Lakshmanamurthy、Albert Magyar、Diwakar Gupta、Jeff Dean、Corry Wang、Matt Johnson、Peter Hawkins和许多其他人。感谢Ruiqi Gao在HTML格式化方面的帮助。

**谢谢大家！**

## 延伸阅读

有一系列相关写作，包括以下内容：

* [**TPU深入解析**](https://henryhmko.github.io/posts/tpu/tpu.html)：以本书精神对TPU架构的出色深入研究。
* [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html)：更专注于GPU和PyTorch的LLM峰值线和性能工程教程。
* [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)：越来越多地，TPU编程涉及在Pallas中编写自定义内核。这个系列讨论如何编写内核和许多这里没有提到的较低层级TPU细节。
* [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM)：虽然特定于GPU和CUDA，这是一篇优秀的博客文章，展示了如何在CUDA中优化matmul内核。这可能是深入了解TPU和GPU如何不同的好方法。
* [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)：这是JAX中并行性API的非常好的指南，是学习如何实际实现我们在这里讨论的一些想法的好方法。
* [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024)：我们的前同事Rafi开设了一个关于TPU性能工程的精彩课程，幻灯片都在GitHub上。这比我们在这里做的更深入地涵盖了一些内容。
* [**\[2211.05102\] Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102)：关于Transformer推理数学的详细论文。这是本文档许多内容的灵感来源。
* [**Huggingface Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：某种程度上是本书的GPU类似物，这更深入地讨论了PyTorch如何实现并行性技术和训练期间的内存节省技术。
* [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/)：一个与本书有许多相同想法的博客，有一些出色的插图。
* [**Stanford CS336 Slides and Videos**](https://stanford-cs336.github.io/spring2025/index.html#coursework)：一个涵盖LLM训练和服务许多细节的出色斯坦福课程，有一些有用的练习。作业1和2特别相关。
* [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering)：ML基础设施的高度实用指南，涵盖本书未涉及的主题，如如何与云提供商谈判、集群管理和GPU吞吐量的经验测量。

在这个领域仍有很大的综合写作空间，所以我们希望这份手稿能鼓励更多此类写作！我们也相信这是一个富有成果的研究和学习领域。在许多情况下，即使没有很多硬件加速器也能完成。

## 反馈

请留下评论或问题，以便我们能进一步改进这个。您可以通过jaaustin [at] google [dot] com联系我们的通讯作者Jacob Austin，或通过在[GitHub](https://github.com/jax-ml/scaling-book)上发布issues、pull requests或discussions来建议编辑。