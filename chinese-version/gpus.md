---
layout: distill
title: "如何看待GPU"
description: "在谈论了这么多TPU之后，可能值得看看世界其他地方大量使用的东西：NVIDIA GPU。这将深入探讨现代NVIDIA ML GPU（例如H100或B100）的芯片和网络层面，以及它们允许的LLM并行性类型。建议您在阅读本书其余部分后阅读此内容。"
date: 2025-07-25
future: true
htmlwidgets: true
hidden: false

section_number: 12

previous_section_url: "../conclusion"
previous_section_name: "第11部分：结论"

next_section_url: ""
next_section_name: ""

bibliography: main.bib

giscus_comments: true

authors:
  - name: To Be Determined
    url: "https://www.jacobaustin.org/"


# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: 什么是GPU？
  - subsections:
    - name: "GPU规格摘要"
    - name: "Grace Hopper"
    - name: 芯片层面的GPU vs. TPU
    - name: 实践问题
  - name: 网络
  - subsections:
    - name: 节点级别
    - name: 实践问题
    - name: 节点级别之外
  - name: GPU上的集合通信如何工作？
  - subsections:
    - name: 节点内
    - name: 网络内约简
    - name: 节点级别之外
    - name: 实践问题
  - name: "GPU上LLM扩展的要点"
  - subsections:
    - name: 实践问题
  - name: "B100和GB200 NVL72如何改变这一点？"

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

GPU最初是用于渲染视频游戏的专用硬件，但自从2010年代AI需求爆发以来，它们开始越来越像专用矩阵乘法机器——换句话说，就像TPU。TPU和GPU都像连接到CPU的矩阵乘法加速器。它们在两个关键方面不同：如何网络连接在一起，以及它们在软件上承担多少做正确事情的责任（*提示：TPU需要编译器做更多工作*）。

## 什么是GPU？

现代GPU（例如H100、B100）基本上是一堆专门用于矩阵乘法的计算核心（称为流式多处理器或SM），全部连接到一条快速内存（称为DRAM或HBM）。这里是一个图表：

{% include figure.liquid path="assets/gpu/gpu-diagram.png" class="img-fluid" caption="<b>图：</b> 现代NVIDIA GPU的基本组件。该图显示了包含一组Tensor Core和Warp调度器（包含CUDA核心）的SM、SMEM、共享L2缓存和主GPU内存（HBM）。" %}

与最多有2个Tensor Core的TPU不同，**现代GPU有超过100个这样的SM**（H100上有132个）。因此，这些SM中的每一个都比TPU TensorCore功能弱得多，但系统整体更灵活。每个SM基本上完全独立，所以GPU可以同时做数百个任务，尽管所有SM共享一个50MB的L2缓存和大量DRAM（H100上80GB，B100上192GB），如上所示。这里是B100 SM的更详细视图：

{% include figure.liquid path="assets/gpu/broadwell-sm.png" class="img-small" caption="<b>图：</b> 单个Broadwell（B100）SM的深入视图，显示了四个SM子分区及其CUDA核心，以及4个TensorCore和一些辅助单元。" %}

每个SM被分解为4个相同的象限，NVIDIA称之为"SM子分区"，每个包含一个Tensor Core、16k个32位寄存器和一组NVIDIA称之为"CUDA核心"的SIMD通道。每个分区的核心组件可以说是Tensor Core，它执行矩阵乘法并构成绝大部分FLOPs/s，但像TPU一样，还有一些其他值得注意的组件。

* **CUDA核心：** 每个子分区包含一组称为**CUDA核心**的ALU，它们进行SIMD矢量运算，很像TPU的VPU。每个子分区包含32个fp32核心（以及较少数量的int32和fp64核心），它们作为单个SIMD单元在每个周期执行相同的矢量操作。**尽管NVIDIA的命名令人怀疑，您应该将这些视为32宽SIMD单元中的通道，在每个周期执行相同的矢量操作。**

    * 子分区内特定精度的每个CUDA核心（SIMD通道）以锁步执行，所以每个周期，每个通道必须执行相同的工作，就像TPU的VPU一样。
    * 对于ML模型，CUDA核心通常用于执行逐点操作，如ReLU、矢量加法、归一化和其他非matmul工作。
    * 子分区内的CUDA核心由称为**warp调度器**的调度单元控制，它充当SIMD单元的控制器。每个warp调度器运行有点像多线程CPU，因为它可以同时运行许多程序（称为**warp**）（每个子分区最多16个），但在每个时钟周期只执行来自单个程序的指令。硬件自动在活动warp之间切换以隐藏I/O操作，如内存加载。
    * 每个warp调度器都有自己的寄存器文件（H100/B100上16,384个32位字，每个SM总计`4 * 16384 * 4 = 256kB`寄存器内存）。每个CUDA核心最多只能访问256个寄存器，所以虽然我们可以每个warp调度器调度最多16个"常驻warp"，但如果每个核心使用256个寄存器，你一次只能装下2个。

* **Tensor Core (TC)：** 每个子分区都有自己的Tensor Core，这是一个专用矩阵乘法单元，如TPU MXU。Tensor Core代表GPU FLOPs/s的绝大部分（例如在H100上，我们有990 bf16 FLOP/s，而CUDA核心只有66 FLOPs/s）。

    * H100的峰值bfloat16 matmul吞吐量为每秒990 bfloat16 TFLOPs，所以每个SM可以做大约7.5TFLOPs峰值。由于每个SM有4个TC，每个SM以1.76GHz的峰值频率运行，每个TC大致可以做`7.5e12 / 1.76e9 / 4 ~ 1024` bf16 FLOPs/s，所以大约每周期一个`8x8x8` matmul。
    * 每个GPU世代都有更大的Tensor Core，因为矩阵乘法计算变得越来越重要（[关于这个的好文章](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)）。
    * 像TPU一样，GPU可以以更高吞吐量进行更低精度的matmul。H100可以做990 bf16 TFLOPs/s和1979 fp8 TFLOPs/s，大约2倍。这意味着如果您可以有效地以更低精度训练或服务模型，您将看到性能的显著提升。
    * 历史上，GPU Tensor Core从SMEM或寄存器内存加载其输入，但随着TC变得更大，变得更难装下完整输入。B100/B200s引入了一种称为Tensor Memory（或TMEM）的新型片上内存，用于存储TC中进行的matmul的输入。

除了计算单元，GPU有一个内存层次结构，最大的是HBM（主GPU内存），然后是一系列较小的缓存（L2、L1、TMEM、寄存器内存）。

* **SMEM (L1 Cache)：** 每个SM都有自己的小型片上缓存，称为SMEM，它可以由程序员控制为"共享内存"或由硬件用作片上缓存。

    * SMEM大量用于存储TC matmul的输入和存储正在处理的激活，所以我们不需要完全从DRAM加载。
    * 因为SMEM比TPU VMEM小得多，更难将模型的整层装入片上内存。

* **L2 Cache：** 所有SM共享一个相对较大的~50MB L2缓存，用于减少主内存访问。

    * 这与TPU的VMEM大小相似，到TC的带宽比HBM高，但不是程序员控制的，而且**要慢得多**。这导致一些"远程诡异行动"，程序员需要修改内存访问模式以确保L2缓存得到良好利用。
    * NVIDIA不公布其芯片的L2带宽，但已测量约为5.5TB/s，或大约HBM带宽的1.6倍。相比之下，TPU的VMEM大2倍*并且*有更多带宽（约40TB/s）。

* **HBM：** 主GPU内存，用于存储模型权重、梯度、激活等。

    * HBM大小从Volta的32GB大幅增加到Blackwell（B100）的192GB。
    * 从HBM到CUDA Tensor Core的带宽称为HBM带宽或内存带宽，H100上约为3.35TB/s。

这里是比较GPU和TPU组件的有用速查表：

|              GPU              |     TPU     |              这是什么？              |
| :---------------------------: | :---------: | :-----------------------------------: |
| 流式多处理器 (SM) | Tensor Core | 包含其他单元的核心"单元" |
|        Warp调度器         |     VPU     |      SIMD矢量运算单元      |
|           CUDA核心           |  VPU通道   |            SIMD ALU"通道"            |
|        SMEM (L1 Cache)        |    VMEM     |       快速片上缓存内存       |
|          Tensor Core          |     MXU     |      矩阵乘法单元       |
|             DRAM              |     HBM     |  高带宽高容量内存  |

### GPU规格摘要

这里是最近型号的GPU规格摘要：

|  GPU  | 世代 |  SM数  | 每个SM的SMEM (kB) | L2缓存 (MB) | 时钟速度 (GHz) | DRAM (GB) | DRAM BW (TB/s) | BF16 TFLOPs | FP8 TFLOPs | FP4 TFLOPs |
| :---: | :--------: | :---: | :--------------: | :-----------: | :---------------: | :-------: | :------------: | :---------: | :--------: | :--------: |
| V100  |   Volta    |  80   |        96        |       6       |     1.25/1.38     |    32     |      0.9       |      —      |     —      |     —      |
| A100  |   Ampere   |  108  |       192        |      40       |     1.10/1.41     |    80     |      2.0       |     312     |     —      |     —      |
| H100  |   Hopper   |  132  |       256        |      50       |     1.59/1.76     |    80     |      3.35      |     990     |    1979    |     —      |
| H200  |   Hopper   |  132  |       256        |      50       |     1.59/1.76     |    141    |      4.8       |   (同上)    |   (同上)   |     —      |
| B100  | Blackwell  |  144  |       256        |      50       |     1.67/1.83     |    192    |       8        |    1800     |    3500    |    7000    |
| B200  | Blackwell  |   ?   |       256        |      50       |         ?         |    192    |       8        |    2250     |    4500    |    9000    |

所有世代每个SM都有256kB寄存器内存。Blackwell还每个SM添加256kB TMEM。一些规格略微依赖于GPU的精确版本。

### Grace Hopper

NVIDIA还销售将一些GPU与Grace Hopper CPU配对的GH200和GB200系统。例如，GH200有1个H200和1个GH CPU，而GB200系统有2个B200和1个GH CPU。这个系统的优势是CPU使用全带宽NVLink连接（称为NVLink C2C）连接到GPU，所以你有非常高的CPU到GPU带宽，对于卸载参数有用。换句话说，对于任何给定的GPU，到达主机内存的带宽与到达另一个GPU的HBM相同。

### 芯片层面的GPU vs. TPU

如您希望注意到的，GPU和TPU在芯片层面看起来相当相似。它们都有matmul加速器、SIMD矢量单元和缓存内存。一个关键区别是TPU有1-2个大Tensor Core，而GPU有数百个小SM。同样，每个Tensor Core有1个大VPU，带4096个ALU，而GPU的H100有132 * 4 = 528个小独立SIMD单元。

这里是GPU与TPU的1:1比较：

| GPU                           | TPU                      | 多少？                                                                                           |
| :---------------------------- | :----------------------- | :-------------------------------------------------------------------------------------------------- |
| SM (流式多处理器) | Tensor Core              | H100有132个，TPU有1-2个。                                                                          |
| Warp调度器                | VPU                      | 每个SM有4个，所以H100上132 * 4 = 528个。TPU v5每个Tensor Core实际上有4个，总共8个。      |
| SMEM (L1缓存)               | VMEM                     | GPU每SM有256kB，总共约32MB。TPU有约120MB的VMEM总量，带宽甚至更高。 |
| 寄存器                     | 矢量寄存器 (VRegs) | GPU每SM有256kB，TPU总共有256kB。                                                            |
| Tensor Core                   | MXU                      | TPU v5p每TC有4个MXU，总共8个。每个H100 SM有4个，总共528个。                        |

如您所见，TPU比GPU模块化程度要低得多！这使它们构建更便宜但编程更复杂。例如，TPU要求matmul是核心`[8, 128] x [128, 128]`大小的倍数，矢量工作以`[8, 128]`增量完成，并将输入数组填充到此大小。如果例如矢量运算比给定矩阵乘法花费更长时间，TPU也容易停滞，因为每个只有1个并且它们经常融合在一起。但如果正确使用，它们也避免了来自GPU模块化性质的大量成本和硬件复杂性。

TPU还有更多可用于存储权重和激活的快速缓存内存。如果您可以一致地将权重获取到VMEM中，这可以使它们在LLM推理方面更快。

### 实践问题

这里有一些实践问题来测试上面内容的一些内容。提供了答案，但在查看之前尝试回答问题可能是个好主意，手头有纸笔。

**问题1 [CUDA核心]：** H100有多少CUDA核心？这与TPU v5p中独立通道数量相比如何？

{% details 点击这里看答案。 %}

**答案：** `132 * 32 * 4 = 16896`个CUDA核心。TPU v5p有2个TensorCore（通常通过Megacore连接），每个都有一个带(8, 128)通道的VPU和每通道4个独立ALU，所以2 * 4 * 8 * 128 = 8192。这是H100矢量通道数量的一半，运行频率大致相同。

{% enddetails %}

**问题2 [矢量FLOPs计算]：** 单个H100有132个SM，运行时钟速度为1.59GHz（最高1.98GHz boost）。假设它每个线程每周期可以做一个矢量操作。每秒可以做多少矢量FP32 FLOPs？有boost时呢？这与matmul FLOPs相比如何？

{% details 点击这里看答案。 %}

**答案：** `132 * 32 * 4 * 1.59e9 = 26.9` TFLOPs/s。有boost时是33.5 TFLOPs/s。这是[规格表](https://www.nvidia.com/en-us/data-center/h100/)中报告的一半，因为从技术上我们可以在一个周期内做FMA（融合乘加），算作两个FLOPs，但这基本上从不可实现。我们可以做990 bfloat16 matmul TFLOPs/s，所以忽略FMA，Tensor Core做大约30倍多的FLOPs/s。

{% enddetails %}

**问题3 [GPU matmul强度]：** H100上的峰值bf16 matmul强度是多少？B200呢？

{% details 点击这里看答案。 %}

**答案：** 对于H100，我们有峰值990e12 bf16 FLOPs和3.35e12字节/秒带宽。所以临界强度是990e12 / 3.35e12 = 295，与TPU中的240相当相似。对于B200是2250e12 / 8e12 = 281，非常相似。这意味着，与TPU相似，我们需要大约280的批次大小才能在matmul中是计算受限的。

{% enddetails %}

**问题4 [L1缓存容量]：** H100的总L1缓存/SMEM容量是多少？寄存器内存呢？这与TPU VMEM容量相比如何。

{% details 点击这里看答案。 %}

**答案：** 我们每SM有256kB，所以各约33MB，或总共约66MB。这约为现代TPU VMEM的120MB的一半，尽管TPU总共只有256kB寄存器内存！TPU VMEM延迟比SMEM延迟低，这是我们在GPU上能用如此少寄存器内存的原因之一。

{% enddetails %}

## 网络

网络可以说是GPU和TPU差异最大的领域。如我们所见，TPU连接成2D或3D环面，其中每个TPU只连接到其邻居。这意味着在两个TPU之间发送消息必须通过每个中间TPU，并迫使我们仅使用网格上的均匀通信模式。虽然在某些方面不方便，但这也意味着每个TPU的链路数量是恒定的，我们可以扩展到任意大的TPU"pod"。

另一方面，GPU使用更传统的分层基于树的交换网络。称为**节点**的8个GPU集合（B200最多72个）在彼此的1跳内以非常高的带宽连接，这些节点通过网络交换机（品牌为NVSwitch）连接到更大的单元（称为SU或可扩展单元），这些单元又通过更高级别交换机连接到更大单元。

### 节点级别

GPU节点是一个小单元，通常为8个GPU（B200最多72个），具有全对全、全带宽连接。每个节点有一些NVSwitch，通过高带宽Infiniband NVLink连接到所有本地GPU。

实际的节点级拓扑随时间变化很大，包括每个节点的交换机数量，但对于H100，我们每个节点有4个NVSwitch，GPU以5 + 4 + 4 + 5链路模式连接到它们：

{% include figure.liquid path="assets/gpu/nvlink-nodes.png" class="img-fluid" caption="<b>图：</b> 不同NVIDIA GPU世代如何将其GPU连接到节点。注意网络配置和每个节点的NVSwitch数量从世代到世代都有变化。" %}

对于H100，每个NVLink链路每个方向有25GB/s带宽（B100为50GB/s），给我们每个GPU到网络18 * 25=450GB/s的全双工带宽。这些大型交换机最多有64个NVLink端口，意味着对于有4个交换机的H100，它们可以处理总共64 * 25e9 * 4=6.4TB/s的带宽。

{% include figure.liquid path="assets/gpu/nvlink4.png" class="img-fluid" caption="<b>图：</b> 显示单个NVLink4交换机如何工作的NVIDIA销售图表（包括64个端口，每个50GB/s带宽）。" %}

这里是这些数字随GPU世代如何变化的概述：

| NVLink世代 | NVSwitch世代 | GPU世代 | NVLink带宽 (GB/s, 全双工) | 每GPU最大链路 | 节点GPU到GPU带宽 (GB/s 全双工) | 节点大小 (NVSwitch域)         |   每节点NVSwitch    |
| :--------: | :----------: | :-------------: | :----------------------------------: | :-------------: | :------------------------------------------: | :----------------------------------: | :----------------------: |
|  **3.0**   |   **2.0**    | Ampere         |                  25                  |       12        |                     300                      | 8                                   |            6             |
|  **4.0**   |   **3.0**    | Hopper         |                  25                  |       18        |                     450                      | 8                                   |            4             |
|  **5.0**   |   **4.0**    | Blackwell      |                  50                  |       18        |                     900                      | H200/B200为8，GB200 NVL72为72 | B200为2，NVL72为18 |

### 实践问题

这里有更多关于网络的问答问题。我发现这些特别有用，因为它们让你工作实际的通信模式。

**问题1 [H100节点总带宽]：** 在有4个交换机的8xH100节点中，每个节点我们有多少总带宽？*提示：* 考虑NVLink和NVSwitch带宽。

{% details 点击这里看答案。 %}

**答案：** 我们有Gen4 4xNVSwitch，每个有64*25e9=1.6TB/s的单向带宽。那会给我们4 * 1.6e12=6.4e12的交换机级带宽。然而，注意每个GPU只能处理450GB/s的单向带宽，所以这意味着我们最多有450e9 * 8 = 3.6TB/s带宽。由于这更小，峰值带宽是3.6TB/s。

{% enddetails %}

**问题2 [平分带宽]：** 平分带宽定义为网络任何偶数分区之间可用的最小带宽。换句话说，如果将网络分成两个相等的一半，两半之间有多少带宽穿过？你能计算8x H100节点的平分带宽吗？*提示：* 平分带宽通常包括两个方向的流量。

{% details 点击这里看答案。 %}

**答案：** 任何偶数分区在每一半都有4个GPU，每个可以向另一半出口4 * 450GB/s。考虑两个方向的流量，这给我们8 * 450GB/s的字节穿过分区，或3.6TB/s的平分带宽。这是NVIDIA例如[在这里](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)报告的。

{% enddetails %}

**问题3 [AllGather成本]：** 给定B字节的数组，在8xH100节点上（吞吐量受限）AllGather需要多长时间？对bf16[DX, F]做数学，其中D=1024, F=16,384。*在回答之前值得阅读TPU集合通信[部分](https://jax-ml.github.io/scaling-book/sharding/)。在这里思考，但我们稍后会更多讨论集合通信。*

{% details 点击这里看答案。 %}

**答案：** 每个GPU可以出口450GB/s，每个GPU有$B / N$字节（其中N=8，节点大小）。我们可以想象每个节点将其字节依次发送给其他$N - 1$个节点，导致总共$N - 1$轮，每轮$T_\text{comms} = (B / (N * W_\text{单向}))$，或$T_\text{comms} = (N - 1) * B / (N * W_\text{单向})$。这大约是$B / (N * W_\text{uni})$或$B$ / 3.6e12，即平分带宽。

对于给定数组，我们有B = `1024*16384*2=32MB`，所以总时间是`33.5e6 / 3.6e12 = 9us`。这可能是延迟受限的，所以实际上可能需要更长时间。

{% enddetails %}

### 节点级别之外

在节点级别之外，GPU网络的拓扑不太标准。NVIDIA发布了一个参考DGX SuperPod架构，连接比单个节点更大的GPU集合，但客户和数据中心提供商可以自由定制以满足他们的需要。

这里是标准1024 GPU H100系统的图表，底行每个DGX pod有8个GPU。每组32个节点称为"可扩展单元"（或SU），在单组8个叶交换机下。该SU有256个GPU，每个节点4个NVSwitch和8个叶交换机。整个SuperPod然后添加16个顶层"脊柱"交换机，给我们1024个GPU，有512个节点级交换机、32个叶交换机和16个脊柱交换机，总共512 + 32 + 16 = 560个NVSwitch。叶交换机以32个节点组连接到节点，所以每组256个GPU有8个叶交换机。所有叶交换机连接到所有脊柱交换机。

在每个级别，我们可能受到可用NVLink带宽、布线或总交换机带宽的瓶颈。

  * **节点级别：** 在节点级别，我们有4 * 1.6TB/s = 6.4TB/s的单向交换机带宽，但我们8个GPU中的每个只能向交换机出口450GB/s，意味着我们实际上在节点内有450e9 * 8 = 3.6TB/s的峰值带宽。
  * **SU/叶级别：** 在SU级别，我们有8个交换机以全对全方式连接32个节点，使用1x400 Gbps Infiniband。这给我们`8 * 32 * 400 / 8 = 12.8TB/s`的节点出口带宽，我们在交换机级别有8 * 1.6TB/s = 12.8TB/s，所以两者精确一致。
  * **脊柱级别：** 在脊柱级别，我们有16个交换机用2x400 Gbps链路连接32个叶交换机，所以我们有32 * 16 * 400 * 2 / 8 = 51.2TB/s的出口带宽。16个交换机给我们16 * 1.6TB/s = 25.6TB/s的带宽，所以这是此级别的瓶颈。

每个GPU，这在节点级别给我们450GB/s的GPU到GPU带宽，在SU级别50GB/s，在脊柱级别25 GB/s。

| 级别     | GPU数量 | 每单元NVSwitch数量 | 每单元总带宽 (TB/s, 全双工) | GPU到GPU带宽 (GB/s, 全双工) |
| :--------: | :-------------: | :----------------------------: | :-------------------------------------------: | :---------------------------------------: |
| 节点      | 8              | 4                             | 3.6                                          | 450                                      |
| 叶 (SU) | 256            | 8                             | 12.8                                         | 50                                       |
| 脊柱     | 1024           | 16                            | 25.6                                         | 25                                       |

相比之下，TPU v5p每个链路有约90GB/s出口带宽，或540GB/s沿所有轴出口。这不是点到点的，所以只能用于受限、均匀的通信模式，但可以扩展到8000个TPU而不损失带宽。

GPU交换结构理论上可以通过添加额外交换机或间接层扩展到任意大小，代价是额外延迟和最远距离带宽降低。

## GPU上的集合通信如何工作？

GPU可以执行与TPU相同的所有集合通信：ReduceScatter、AllGather、AllReduce和AllToAll。与TPU不同，这些工作方式取决于它们是在节点级别还是更高级别执行。所有这些集合通信都由NVIDIA在NCCL（发音"nickel"）库中实现，实际实现是黑盒。从这里开始，我们将讨论NVSwitch树上理论最优模型。

### 节点内

对于节点级别的AllGather或ReduceScatter，您可以像TPU一样在环上执行它们，在每跳使用完整GPU到GPU带宽。任意排序GPU并使用完整GPU到GPU带宽在环上发送数组的一部分：

$$T_\text{AG或RS通信} = \frac{\text{字节} \cdot (N - 1)}{N \cdot \text{GPU出口带宽}} \rightarrow \frac{\text{字节}}{\text{GPU出口带宽}}$$

对于AllReduce，您可以像往常一样组合RS + AG，成本是两倍。如果您关心延迟（例如如果您的数组非常小），您可以同时直接发送给节点中的每个GPU，将总发送字节数增加N（节点大小）但在一跳中执行整个操作。

到目前为止，这与TPU完全相同，只是整体带宽略有不同。例如，对于有450 GB/s单向带宽的H100，AllGather(bf16[B<sub>X</sub>, F])大致需要$T_\text{通信} = (2 \cdot B \cdot F) / 450e9$。

#### 网络内约简

自Hopper世代以来，NVIDIA交换机支持"SHARP"（*可扩展分层聚合和约简协议*），允许"网络内约简"。这意味着网络交换机本身可以做约简操作并将结果复用或"多播"到多个目标GPU。这有效地将AllReduce的成本减半，因为它意味着每个GPU可以将其数据发送到顶层交换机，在那里执行约简，然后广播结果，而不必每个GPU出口两次。

$$T_\text{AR通信} = \frac{\text{字节}}{\text{GPU出口带宽}}$$

注意：这是精确的，不是$1 / N$的因子，因为每个GPU首先出口$B * (N - 1) / N$，然后接收其本地分片的部分约简版本（$B / N$的入口），完成约简，然后再次出口$B / N$，然后入口完全约简的结果（$B * (N - 1) / N$的入口），导致正好字节$B$字节入口。

### 节点级别之外

当我们超越节点级别时，成本更微妙一些。我们可以继续做一种环约简，我们将环视为在叶或脊柱交换机上。例如，对于AllReduce，我们可以首先在节点内AllReduce，然后在叶内，然后在脊柱内，每次做一种环约简。在理想世界中，我们可以重叠这些，最终只受最慢交换机的瓶颈。作为第一近似，

$$T_\text{通信} = \max_i\left(\frac{\text{字节} \cdot N_{\text{子域}_i}}{W_i}\right)$$

其中$W_i$是级别$i$的聚合交换机带宽。所以例如，在上述情况下，我们在节点级别有3.6TB/s，有8个子域，在SU级别有12.8TB/s，有32个子域，在脊柱级别有25.6TB/s，有4个子域。这意味着实际上我们将受到最大比率的瓶颈，即`max(8 / 3.6e12, 32 / 12.8e12 , 4 / 25.6e12) = max(2.2e-12, 2.5e-12, 1.56e-13)`，所以实际上$T_\text{通信} = B \cdot \text{2.5e-12} = B / 400e9$，即我们即使在最高级别也有大约400GB的AllReduce带宽。

一般来说，AllReduce带宽是$\max_i(N_{\text{子域}_i} / W_i)$，所以上面是400GB/s，由叶级交换机确定。AllGather有点复杂，因为每个级别实际通信的量会改变。它大致相同但更接近$\max_i(\text{字节} \cdot (N - 1) / W)$。

### 实践问题

**问题1 [单节点AR]：** 考虑每个节点有N个GPU的单个节点。在AllReduce期间，交换机精确地入口和出口多少字节？

{% details 点击这里看答案。 %}

**答案：** 让我们逐步做。

1.  每个GPU发送$B \cdot (N - 1) / N$字节，所以我们有$N \cdot B \cdot (N - 1) / N = B \cdot (N - 1)$入口。
2.  我们累积部分和，向每个GPU发送回$B / N$字节，所以$N \ B / N = B$字节出口。
3.  我们在本地对残差做部分和，然后发送回交换机。这总共是$N * B / N = B$字节入口。
4.  我们捕获所有分片并多播它们，发送$B * (N - 1) / N$到$N$个目的地，总共$B * (N - 1) / N * N = B * (N - 1)$出口。

因此总计是$B * (N - 1) + B = B\cdot N$字节入口和出口。这支持整体吞吐量正好是$B\cdot N / W_\text{交换机}$。

{% enddetails %}

**问题2 [SU AllGather]：** 只考虑有M个节点和每个节点N个GPU的单个SU。在AllGather期间，节点级交换机精确地入口和出口多少字节？顶层交换机呢？

{% details 点击这里看答案。 %}

**答案：** 这与上面类似，我们再次分阶段做。

1.  每个GPU向交换机发送$B / MN$字节，总入口$NB / MN = B / M$字节。
2.  我们将完整$B / M$字节出口到脊柱交换机。
3.  我们从脊柱交换机入口$B * (M - 1) / M$字节
4.  我们出口$B - B / MN$字节$N$次，总共$N * (B - B / MN) = NB - B / M$。

总计是$B$入口和$BN$出口，所以我们应该受到出口的瓶颈，总时间会是

$$T_\text{AllGather} = \frac{BN}{W_\text{节点}} = \frac{B}{450e9}$$。

对于脊柱交换机，数学实际更简单。我们必须有B / M字节入口M次（总共B字节），然后B (M - 1) / M出口M次，总共B * (M - 1)出。由于这显著更大，成本是

$$T_\text{AllGather} = \frac{B \cdot (M - 1)}{W_\text{顶层}}$$

{% enddetails %}

## GPU上LLM扩展的要点

复制上面H100 SuperPod的表格，我们有

| 级别 | GPU数量 | GPU到GPU带宽 (全双工, GB/s) | 总带宽 (全双工, TB/s) | AllReduce带宽 (GB/s)        |
| :---: | :------------- | :--------------------------------------- | :---------------------------------- | :-------------------------------- |
| 节点  | 8              | 450                                      | 3.6 (受NVLink限制)             | 450GB/s                           |
| 叶  | 256            | 50                                       | 8 * 1.6 = 12.8                     | 400GB/s                           |
| 脊柱 | 1024           | 25                                       | 16 * 1.6 = 25.6                    | 400GB/s (从叶继承) |

让我们像对TPU那样看计算通信峰值线。这里，[如前](../training)，我们将比较计算和通信时间，看什么点$T_\text{math} \gt T_\text{comms}$。

**数据并行性：** 要在单个节点内纯数据并行性中是计算受限的，有网络内约简，我们有

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot DF}{W_\text{AllReduce}}$$

其中我们在通信中失去因子2，因为我们启用了网络内约简。因此，要是计算受限，我们需要$2B / (XC) \gt 1 / W_\text{AllReduce}$或$B / X \gt C / (2 \cdot W_\text{AllReduce})$，所以我们只需要每GPU批次大小> `989e12 / (2 * 450e9) = 1098`，与TPU相当相似（其中数字是850，有三个轴）。如果我们试图在SU或脊柱级别做这个，我们得到$BS \gt 989e12 / (2 \cdot 400e9) = 1236$。

**FSDP：** 对于FSDP，这是唯一真正有用的东西，数字是这个的两倍，所以对于FSDP我们需要每GPU BS > 2472。注意这个数字比TPU大多少（尽管B100会将此减少因子2）。

由于我们需要某种FSDP，这意味着例如，对于2048个GPU，我们至少需要5M token的批次大小，这相当可行。

**模型并行性：** 对于模型并行性，这表明在单个节点内，我们需要$Y \< F / (898e12 \cdot (8-1) / (8 \cdot 450e9)) = F / 1746$，所以对于F=16k，我们可以达到9路并行性（或真的8路，因为这是节点有多大）。显然，我们不能比这更大，但不是因为跨节点带宽低，而是因为我们的整体带宽低。

**混合FSDP + 模型并行性：** 将某种形式的模型并行性与DP结合不像在TPU网格中那样简单，我们通过Y（TP的数量）减少AllReduce的成本。树$\text{AllReduce}_X(A_Y { U_X })$（假设Y是内轴）的一般规则似乎是

$$T_\text{通信} = \max_i\left[\frac{B \cdot S_i}{\max(Y, S_{i-1}) \cdot W_i}\right]$$

其中$S_i$是M * N * …，树中级别$i$子节点的大小。所以如果Y是64，那么在节点级别我们有`B * 8 / (64 * 3.6e12)`，在叶级别我们有`B * 256 / (64 * 12.8e12)`，在脊柱级别我们有B * 1024 / (256 * 25.6e12)，或换句话说带宽28.8e12、3.2e12和6.4e12。这意味着我们在节点级别有效获得64倍因子，在叶级别8倍带宽，在脊柱级别保持恒定。如果我们做256路专家分片，我们在节点级别得到256倍加速，在叶级别32倍，在脊柱级别没有加速。

因此，如果我们做8路模型并行性，我们确实将节点级约简的成本减少8倍，其他所有保持相同，所以基本免费但在减少约简总成本方面没有用。

**专家并行性：** 专家并行性在本书主体中没有详细讨论，但作为第一近似，MoE将密集模型的MLP权重复制E次，即将$W_\text{in}[D, F]$和$W_\text{out}[F, D]$转换为$W_\text{in}[E, D, F]$和$W_\text{out}[E, F, D]$，但每个token只激活其中k个。这将总内存增加E倍但FLOPs只增加k倍。这添加两个AllToAll来发送token到它们选择的专家，但在DP设置中也要求我们AllReduce E倍更多字节。

AllToAll在GPU中比TPU更简单，因为它们可以直接点到点完成，所以在这个意义上故事更清洁。

但额外内存是更大的问题。在TPU上，我们可以简单地沿完全独立的轴分片专家，例如$W[E_z, D_X, F_Y]$，这将我们减少到更或多标准密集模型的2轴。但在GPU上，我们不能再沿单独轴分片我们的专家，所以我们不能再完全缓解额外通信的成本。如我们上面注意到的，通过做更多任何类型的分片，但特别是专家并行性，我们可以在某种程度上减少AllReduce的成本。如果我们在节点级别做8路专家并行性，我们减少节点级成本但不减少脊柱级成本，这不好。如果我们做64路甚至256路，我们得到显著收益，但峰值线不再那么简单，因为对于DP=X和EP=Y，每个专家k个token，我们有（在后向传递中）

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot B \cdot D \cdot F}{X \cdot Y \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot E \cdot D \cdot F}{W_\text{AllReduce}}$$

其中$W_\text{AllReduce}$在叶级别将高8倍。这仍然给我们

$$\frac{B}{X} \gt \frac{E \cdot C}{2 \cdot k \cdot W_\text{AllReduce}}$$

所以有8路EP，我们在叶级别真的没有受益。采用256个专家和8个激活（大致）的DeepSeek，64路EP（忽略流水线并行性），突然成本降到`1236 * 256 / 8 / 8 = 4944`，意味着我们需要`4944 * 2048 = 10M`个token，这实际可行。这忽略了FSDP的额外因子2，会将此带到20M。

**流水线并行性：** 流水线并行性成本很小，因为我们只是通过顶层交换机跳跃小消息（激活或激活梯度）。这算作额外形式的模型并行性，尽管它使FSDP更具挑战性。

**DeepSeek做什么？** 作为参考，DeepSeek用2048个H800 GPU训练：

  * 跨8个节点的64路专家并行性(EP)
  * 16路流水线并行性(PP)
  * 2路ZeRO-1数据并行性(DP)

他们有`4096 * 15360 = 62,914,560`token的稳态批次大小。您可以看到有64路EP和16路PP，我们最终总共有1024路模型并行性，这意味着AllReduce在脊柱级别完成。这给我们来自胖树更多带宽来工作，所以我们对更多数据并行性没有问题。我们可以只做4路流水线并行性仍然在这种情况下，尽管这有自己的问题（气泡）。

**GPU扩展的TLDR：**

* 纯数据并行性很棒，因为SHARP减少AllReduce成本，但对大模型不是很有用。
* FSDP还可以，成本2倍，峰值线是纯DP的因为我们必须做AG + RS或AR + AG（第二个对流水线更好）。ZeRO-1与流水线一起工作，ZeRO-3不行。每GPU需要约2k token。
* MP + FSDP还可以但不太好，模型并行性由于纯带宽原因（与跨节点带宽较少无关）不能扩展超过节点。在B100/B200中更好。减少每GPU内存但在其他方面没有帮助，即不减少临界批次大小，因为它不减少叶级带宽。
* 一般来说，MoE + DP有点困难，因为专家如此分块，所以DP/FSDP峰值线大幅上升，除非我们做大量专家并行性。真的需要超出节点级别的专家并行性，理想情况下EP + MP占用整个SU，所以例如8路模型并行性，32路专家并行性工作得很好。需要跨许多节点来减少叶级带宽。
* 流水线并行性工作得很好，如果您可以处理零气泡流水线的代码复杂性。使ZeRO-3不可能，所以必须做ZeRO-1代替。算作模型并行性。
* **主要TLDR：**
    * 对于小一些的密集模型，8路TP +纯数据并行性会非常强。对于H100，您可以在bf16中达到64B模型。
    * 对于较大密集模型，可以做TP + PP或FSDP取决于批次大小。更多PP让你去更小批次大小，但FSDP在大多数情况下还可以。
    * 对于MoE，可以做一些TP + EP + PP组合，让你达到脊柱级别，然后你就可以了，因为你在脊柱级别有胖树的大量带宽。不能超过节点级别TP，EP受专家数量和不平衡+ A2A成本约束，PP受微批次大小约束。然后在此之外做纯DP或ZeRO-1/3。

### 实践问题

**问题1 [跨节点AR成本]：** 考虑在N个GPU的单个节点上分片的数组bf16[D<sub>X</sub>, F<sub>Y</sub>]。$\text{AllReduce}(bf16[D, F_Y] { U_X })$需要多长时间？您可以假设我们做网络内约简。如果我们有超过单个节点，这如何不同？

{% details 点击这里看答案。 %}

**答案：** 我们可以尝试修改上面类似问题的答案。基本上，我们首先从每个GPU出口$B * (X - 1) / XY$字节，然后发送回$B / XY$到每个GPU，然后发送相同量回交换机，然后发送$B * (X - 1) / XY$回每个GPU。总计是$NB / Y$入口和出口，所以总时间是$T_\text{通信} = NB / (Y \cdot W_\text{交换机}) = N \cdot 2DF / (\left Y\rvert \cdot W_\text{交换机})$，所以总时间确实随Y减少。

如果我们超过单个节点，我们可以做大致相同的约简如上，但当我们出口节点级交换机时，我们需要发送所有B字节，不只是B / Y。这是因为我们需要保持每个模型分片分离。

{% enddetails %}

**问题2 [脊柱级别AR成本]：** 考虑与上面相同的设置，但有256路模型并行性（所以AR在脊柱级别发生）。AllReduce需要多长时间？我们在这里可以处理什么每GPU批次大小？

{% details 点击这里看答案。 %}

**答案：** 这让我们利用脊柱级别相当惊人的带宽量。我们在4个节点上有25.6TB/s带宽，所以AllReduce带宽6.4TB/s。使用SHARP，这可能只需要$2 \cdot D \cdot F / 6.4e12$秒。

这意味着理论上我们可以有小到每GPU`989e12 / (2 * 6.4e12) = 77`个token的批次大小，或每SU 19,712，这相当疯狂。如果我们做类似节点内8路模型并行性和跨节点32路专家并行性，或某种形式的流水线，这可能是情况。

{% enddetails %}

## B100和GB200 NVL72如何改变这一点？

Broadwell引入了一系列主要网络变化，包括带宽翻倍（900GB/s）的NVLink 5和大得多的节点（NVL72中72个GPU）。这里是一个图表：

{% include figure.liquid path="assets/gpu/b100-node.png" class="img-fluid" caption="<b>图：</b> 显示B100/B200 NVL72节点如何构建的图表，有18个交换机和72个GPU。" %}

第一阶效应是我们所有峰值线大约好两倍：我们所有AllReduce和AllGather都快两倍，所以我们可以做两倍多。我们为纯数据并行性计算的BS > 1098界限降到549，接近TPU v5p。F = 16000的模型并行性界限增加到18，意味着我们可以做几乎两倍的模型并行性。

NVIDIA还计划构建576 GPU GB200 NVL576拓扑，有两层交换机但可以在所有GPU之间实现全带宽。这大致是一个节点，尽管它在更远距离GPU之间会有一些小的额外延迟。这尚未发布。

{% include figure.liquid path="assets/gpu/nvl-576.png" class="img-small" caption="<b>图：</b> 显示我们如何在Broadwell中看到576 GPU节点的图表。" %}