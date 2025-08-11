---
layout: distill
title: "如何对TPU程序进行性能分析"
# permalink: /main/
description: "到目前为止，这个系列完全是理论性的：基于硬件roofline的粗略计算。这种理解能让你走得很远，但很多优化都归结于实际细节：XLA编译器如何工作，以及当它失败时如何使用JAX/Tensorboard分析器等性能分析工具来确定该做什么。我们在这里讨论这些。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 9

previous_section_url: "../applied-inference"
previous_section_name: "第8部分：服务LLaMA"

next_section_url: "../jax-stuff"
next_section_name: "第10部分：JAX"

giscus_comments: true

authors:
  - name: Jacob Austin
    url: "https://www.jacobaustin.org/"
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
  - name: Translator - Jinsong Hao
    url: ""

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "TPU软件栈的千英尺视图"
  - name: "TensorBoard分析器：多用途TPU分析器"
  - subsections:
    - name: "跟踪查看器"
    - name: "如何阅读XLA操作"
    - name: "图查看器"
    - name: "查看真实（ish）示例性能分析"
    - name: "内存分析"
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

## TPU软件栈的千英尺视图

Google为TPU编程公开了一系列API，从高级JAX代码到低级Pallas或HLO。大多数程序员专门编写JAX代码，这让你可以编写抽象的NumPy风格线性代数程序，这些程序会自动编译以在TPU上高效运行。

这里是一个简单的例子，一个将两个矩阵相乘的JAX程序：

```py
import jax
import jax.numpy as jnp

def multiply(x, y):
  return jnp.einsum('bf,fd->db', x, y)

y = jax.jit(multiply)(jnp.ones((128, 256)), jnp.ones((256, 16), dtype=jnp.bfloat16))
```

通过调用`jax.jit`，我们告诉JAX跟踪这个函数并发出一个名为[StableHLO](https://openxla.org/stablehlo)的较低级IR，这是一个用于ML计算的平台无关IR，然后XLA编译器将其降低为HLO。编译器运行许多遍来确定融合、布局和其他因素，这些因素导致在JAX性能分析中可观察到的HLO。这个HLO以LLVM风格的图视图表示JAX代码中的所有核心线性代数操作（矩阵乘法、点运算、卷积等）。例如，这里是上述程序作为HLO的删节版本<d-footnote>要获得这个HLO，你可以运行`jax.jit(f).lower(*args, **kwargs).compile().as_text()`。</d-footnote>：

```c
ENTRY %main.5 (Arg_0.1: f32[128,256], Arg_1.2: bf16[256,16]) -> f32[16,128] {
  %Arg_1.2 = bf16[256,16]{1,0} parameter(1), metadata={op_name="y"}
  %convert.3 = f32[256,16]{1,0} convert(bf16[256,16]{1,0} %Arg_1.2),
  %Arg_0.1 = f32[128,256]{1,0} parameter(0), metadata={op_name="x"}
  ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1},
}
```

我们马上会解释HLO的语法，但现在只需注意它实际上与上面的JAX代码匹配得相当好。例如，

```c
ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
```

是上面实际的矩阵乘法，它分别沿0和1维度乘以两个f32矩阵。

**要将这个HLO转换为可以在TPU上执行的代码，XLA编译器首先将其降低为LLO**（低级优化器）IR。LLO直接编程TPU，调度内存之间的复制，将数组推送到脉动数组等。LLO代码包含将缓冲区推送到脉动数组、拉出结果以及调度在不同TPU内存片段之间通信的DMA的原语。一旦降低为LLO，它就会被编译为机器代码，加载到TPU IMEM中并执行。

当程序运行得比我们希望的慢时，我们主要在JAX级别工作来改善性能。然而，这样做经常需要我们理解HLO的一些语义以及代码实际在TPU上的运行方式。当较低级别出现问题时，我们拉出另一个逃生舱口，在[Pallas](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)中编写自定义内核。要查看程序的HLO及其运行时统计信息，我们使用JAX分析器。

## TensorBoard分析器：多用途TPU分析器

JAX提供了一个多用途TPU分析器，具有一系列有用的工具，用于理解程序运行时TPU上发生的情况。你可以使用`jax.profiler`模块在程序运行时跟踪程序并记录从每个子组件的持续时间、每个程序的HLO、内存使用等一切。例如，这段代码会将跟踪转储到`/tmp/tensorboard`中的文件，可以在TensorBoard中查看（[这里](https://docs.jax.dev/en/latest/profiling.html#tensorboard-profiling)是分步指南）。

```python
import jax
with jax.profiler.trace("/tmp/tensorboard"):
  key = jax.random.key(0)
  x = jax.random.normal(key, (1024, 1024))
  y = x @ x
  y.block_until_ready()

# 现在你可以在Google Colab中用以下方式加载TensorBoard
#
# !pip install tensorboard-plugin-profile
# %load_ext tensorboard
# %tensorboard --logdir=/tmp/tensorboard
#
# 或者外部用
#
# > tensorboard --logdir=/tmp/tensorboard
#
```

这里是你可以在分析器中做什么的概述：

{% include figure.liquid path="assets/img/xprof-overview.png" class="img-fluid" %}

一旦进入TensorBoard，分析器有几个关键选项卡，帮助你理解你的程序：

1. **跟踪查看器**显示TPU上实际发生的事情的详细时间轴。
2. **图查看器**显示HLO图，让你看到程序的哪些部分相互输入以及事物是如何分片的。
3. **内存分析和内存查看器：**这些显示你的程序使用多少内存。

虽然共享性能分析稍微困难，[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)是一个Perfetto链接，至少包含简单Transformer的跟踪查看器组件。[这个Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)让你生成完整的JAX/TensorBoard跟踪并试用它。

### 跟踪查看器

**跟踪查看器可能是分析器最有用的部分。**下面的例子显示了一个简单的Transformer，其中的部分已被注释。名称来自代码中提供的标签。

{% include figure.liquid path="assets/img/trace-viewer.png" class="img-fluid" %}

跟踪查看器显示每个TPU核心上所有操作的时间顺序时间轴。我们这里只看TPU:0，因为通常所有TPU都执行相同的指令。几个关键注意事项：

1. 顶行（XLA操作）显示实际的TPU操作（名称是HLO名称）。其他一切都是基于`jax.named_scope`、`jax.named_call`和Python堆栈跟踪的近似跟踪。
2. 注意重复的块，我们可以在这里隔离单层。我们也可以看到（从查看代码/理解Transformer如何工作）哪些部分是注意力，哪些部分是MLP。
3. 通过点击XLA操作，我们可以查看它在代码中来自哪里（对理解跟踪有用）并看到图查看器的链接。

<p markdown=1 class="takeaway">**提示：**你可以使用"电子游戏"风格的控件导航跟踪查看器，用A/D左右平移，用W/S缩放。这些控件使导航更加容易。</p>

### 如何阅读XLA操作

HLO实际上并不难阅读，对于理解上面跟踪的给定部分对应什么非常有帮助。这里是一个名为fusion.3的示例操作。

```py
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3
```

让我们将其分解为各个部分。

* **操作名称**：fusion.3
  * 点或融合操作是包含最多1个矩阵乘法和可能一堆相关逐点VPU操作的操作集合。
* **形状/布局**：`bf16[32,32,4096]`
  * 这是操作的输出形状。我们可以看到dtype是bf16（每参数2字节），`[32,32,4096]`是形状。
* **布局：**`{2,1,0:T(8,128)(2,1)}`
  * `{2,1,0:T(8,128)(2,1)}`告诉我们内存中轴的顺序（列主、行主等）和数组填充。下面有更多。
* **内存位置：**S(1)
  * S(1)告诉我们这个数组存在于VMEM中。S(0)（有时省略）是HBM。S(2)和S(3)是其他内存空间。
* **参数**：`bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32`
  * 这个操作有一个输入，一个名为fusion.32的具有特定形状的bf16数组。这告诉我们什么函数输入到这个函数中。

让我们试着更好地理解这个符号。让我们以这个为简单例子：

`f32[3,5]{1,0:T(2,2)}`

它再次告诉我们这个操作返回一个形状为`[3, 5]`的float32数组，具有特定的平铺`{1,0:T(2,2)}`。虽然平铺不*太*重要，简单地说，平铺告诉我们N维数组在内存中是如何顺序布局的。这里是一个显示这个数组如何布局的图表：

{% include figure.liquid path="assets/img/tiling.png" class="img-fluid" %}

在`{1,0:T(2,2)}`中，`1,0`部分告诉我们物理内存中数组维度的顺序，从最次要到最主要。你可以从右到左读取这部分，并选出`f32[3,5]`中的相应维度来确定数组的物理布局。在这个例子中，物理布局是`[3,5]`，与逻辑形状相同。
之后，`T(2,2)`告诉我们数组以`(2, 2)`的块进行平铺，在每个块内，数组首先有行（**行主**），然后是列，即`(0, 0)`后面跟着`(0, 1)`，然后是`(1, 0)`和`(1,1)`。由于`T(2, 2)`平铺，数组被填充到`[4, 6]`，将其内存使用增加大约1.6倍。对于上面给出的大bf16数组，`bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)}`，我们做`T(8,128)(2,1)`，它告诉我们数组有两级平铺，外部`(8, 128)`平铺和该单元内的内部`(2, 1)`平铺（用于bf16，所以我们的加载总是4字节的倍数）。例如，这里是`bf16[4,8]{1,0,T(2,4)(2,1)}`（颜色是(2,4)平铺，红框是(2,1)平铺）：

{% include figure.liquid path="assets/img/tiling2.png" class="img-fluid img-small" %}

平铺可能影响张量块加载到VMEM的效率，XLA有时会引入在程序内"重新平铺"或"重新布局"张量的复制，有时会有非平凡的开销。<d-footnote>JAX提供了一个实验性功能来解决这个问题，通过允许XLA计算程序输入的"首选"布局。当你用`jax.jit`"即时"编译程序时，你通常传入"模拟"输入，告诉JAX期望什么形状和dtype。这些通常也携带可能不是最优的平铺信息。相反，你可以将输入布局指定为AUTO，`jax.jit`会返回编译程序首选的布局。然后你可以显式地以该布局加载张量，以避免在程序内引起复制。</d-footnote>

### 图查看器

虽然上面的一些融合看起来可能很复杂，XLA图查看器使它们更容易解析。例如，这里是一个相当复杂融合的视图：

{% include figure.liquid path="assets/img/graph-viewer.png" class="img-fluid" %}

盯着一堆HLO图并试图将HLO操作映射到你正在分析的代码上真的很有帮助。通过悬停在框上，你经常会看到定义函数的代码行。

### 查看真实（ish）示例性能分析

[这个Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)有一个虚假Transformer的示例性能分析。[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)是一个Perfetto链接，如果你匆忙至少可以看到跟踪查看器。我比平时更费心地用`jax.named_scope`调用注释跟踪，所以你可以识别发生了什么。

{% include figure.liquid path="assets/img/transformer-xprof.png" class="img-fluid" %}

看看性能分析，试着真正理解每个部分在做什么。让我们稍微分解一下，从FFW块开始：

{% include figure.liquid path="assets/img/transformer-ffw.png" class="img-fluid" %}

这里我们已经缩放到FFW块。你会看到上投影操作是一个融合（矩阵乘法），输入为`bf16[8, 1024, 8192]`和`bf16[8192, 16384]`，输出为`bf16[32, 1024, 16384]`。我知道（因为我写了这个代码）这是一个4路DP，2路MP分片矩阵乘法的局部视图，所以我们实际上在做

**X：** `bf16[32, 1024, 8192]` \* **W<sub>in</sub>**：`bf16[8192, 32768]` -> **Tmp**：`bf16[32, 1024, 32768]`

**我们预期这需要多长时间？**首先，我们每个数据并行分片的批大小是`8 * 1024 = 8192`，所以我们应该牢固地受计算限制。这是在8个TPUv2核心上（在Google Colab上免费提供），所以我们预期它需要大约`2 * 32 * 1024 * 8192 * 32768 / (23e12 * 8) = 95.6ms`，这几乎正是它所需的时间（96ms）。太好了！这意味着我们得到了很棒的FLOP利用率！

**通信呢？**你会注意到隐藏在第二个矩阵乘法末尾的小融合。如果我们点击它，你会看到

```py
%fusion.1 = bf16[8,1024,4096]{2,1,0:T(8,128)(2,1)} fusion(bf16[8,1024,8192]{2,1,0:T(8,128)(2,1)} %fusion.31), kind=kCustom, calls=%all-reduce-scatter.1
```

这基本上是一个小的ReduceScatter（这里是GraphViewer）；

{% include figure.liquid path="assets/img/reduce-scatter-xprof.png" class="img-fluid" %}

我们预期这需要多长时间？嗯，我们在TPUv2 4x2上进行ReduceScatter，它应该只需要在1.2e11双向带宽上进行一跳。数组大小为`2*32*1024*8192`，批轴分片4路，所以每个分片是`2*8*1024*8192=134MB`。所以这应该大约需要1.1ms。**实际上需要多长时间？**性能分析中报告的1.13ms。所以我们真的很接近roofline！

**让我们也看看注意力！**这里是注意力组件的性能分析：

{% include figure.liquid path="assets/img/attn-xprof.png" class="img-fluid" %}

我点击了Q投影操作，它使用形状为[d<sub>model</sub> = 8192, n<sub>heads</sub> = 32, d<sub>qkv</sub> = 256]的矩阵$$W_Q$$。我们沿头维度进行Megatron分片。试着做同样的计算这些应该需要多长时间的练习。

### 内存分析

内存分析使得查看程序内存作为时间函数变得容易。这对于调试OOM很有帮助。你可以看到这里大约7.5GB分配给模型参数，大约10GB空闲。所以我们可以在内存中放入更多。

{% include figure.liquid path="assets/img/memory-viewer.png" class="img-fluid" %}

## 练习题

**问题1**：看看[这个](https://colab.research.google.com/drive/1LfLO3OTr-_MWFPxUN36KJ3cqH0BcAoli?usp=sharing) Colab/性能分析，找出看起来可疑的地方以及这里发生了什么。你能确切告诉我正在进行什么计算以及每个操作在做什么吗？涉及的每个矩阵的真实形状是什么，它们是如何分片的？*试着先看性能分析而不读代码。*

{% include figure.liquid path="assets/img/all-reduce-profile.png" class="img-fluid" %}

{% details 点击这里查看答案。 %}

这是两个矩阵乘法，即具体地说是这个：

```py
def matmul(w1, w2, x):
  return jnp.einsum('wf,bf->bw', w2, jnp.einsum('fw,bw->bf', w1, x))
```

你可以看到一个reduce、两个大融合和一个all-reduce。第一个大融合是：

```%fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1```

它告诉我们每分片形状是`bf16[8192] * bf16[4096, 8192] -> bf16[4096]`（在8192维度上）。通过观察最终的AllReduce与`replica_groups=\{\{0,16,32,48,64,80,96,112\}, ...\}`，我们可以知道我们在进行8路模型并行，所以真实形状是`[8, 8192] * bf16[32,768, 8192] -> bf16[8, 32,768]`。

{% enddetails %}

**问题2：**[之前的Transformer Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)实现了一个简单的模拟Transformer。按照Colab中的说明，用GSPMD分区获得朴素Transformer的基准。每个部分需要多长时间？应该需要多长时间？使用什么分片。试着修复分片！*提示：使用`jax.lax.with_sharding_constraints`来约束行为。有了这个修复，你能得到的最佳MXU是什么？*

作为参考，初始版本大约获得184ms/层，优化后的性能分析获得67ms/层。一旦你完成了这个，试着盯着性能分析，看看你能否纯粹从性能分析中回答这些问题：

- 这是什么分片策略？
- 批大小、$$d_\text{model}$$、$$d_\text{ff}$$是什么？
- 多少时间花费在注意力上vs. MLP块？
- 在roofline上每个操作应该花费多少时间的比例？

**注意：**自从编写这个问题以来，XLA编译器已经改进了。初始版本现在大约是90ms/层，优化后的性能分析只好了大约10ms/层（80ms/层）。尽管如此，还是值得试用并看看你能否做得更好。

<h3 markdown=1 class="next-section">第9部分就到这里。对于第10部分，深入探讨JAX并行性，点击[这里](../jax-stuff)。</h3>