---
layout: distill
title: "用JAX编程TPU"
# permalink: /main/
description: "如何使用JAX高效编程TPU！本节大部分内容来自<a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>这里</a>。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 10

previous_section_url: "../profiling"
previous_section_name: "Part 9: Profiling"

next_section_url: ../conclusion
next_section_name: "Part 11: Conclusions"

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
  - name: "JAX中的并行性如何工作？"
  - subsections:
    - name: "jax.jit: 自动并行性解决方案"
    - name: "shard_map: 对程序的显式并行性控制"
  - name: "实践问题"

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

## JAX中的并行性如何工作？

JAX支持多设备编程的两种思路：

1. **编译器，交给你了！** 让编译器自动分割数组并决定添加什么通信来促进给定程序。这让你在单个设备上写程序并自动在数百个设备上运行，而无需更改任何东西。
2. **让我写我想要的，该死的！** 虽然编译器很好，但它们有时做错事并添加你不想要的通信。有时我们想要非常明确我们在做什么。

相应地，JAX为每种思路提供两个API：**jit**（`jax.jit`）和**shard\_map**（`jax.experimental.shard_map.shard_map`）。

1. `jax.jit`让你指定程序输入和输出的分片（通过`in_shardings`和`out_shardings`），并使用[GSPMD](https://arxiv.org/abs/2105.04663)编译器推断其余部分。虽然不完美，但通常在自动将程序扩展到任意数量芯片方面做得相当不错。
2. `jax.experimental.shard_map.shard_map`是更明确的对应物。你得到程序的设备本地视图，必须明确写出你想要的任何通信。有一个分片数组并想要整个东西在每个设备上？添加一个`jax.lax.all_gather`。想要跨设备对数组求和？添加一个`jax.lax.psum`（一个AllReduce）。编程更难但做你不想要的事情的可能性要小得多。

<h3 id="jax-jit-the-automatic-parallelism-solution">jax.jit: 自动并行性解决方案</h3>

jax.jit在JAX内部扮演两个角色。顾名思义，它"即时"将函数从Python编译为字节码（通过XLA/HLO/LLO），所以运行更快。但如果输入被分片或用户指定`in_sharding`或`out_sharding`，它也让XLA跨多个设备分布计算并根据需要添加通信。例如，这是你如何使用jax.jit写一个分片matmul：

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

P = shd.PartitionSpec

# 在TPU v5e 2x2上运行。这为硬件的两个物理轴分配名称。
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'))

# 这告诉JAX为所有操作使用这个网格，所以你可以只指定PartitionSpec P。
shd.set_mesh(mesh)

# 我们创建一个矩阵W和输入激活In，跨设备分片。
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=P('Y', None))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# 我们可以在这里明确编译分片matmul函数。这添加了所有
# 必要的通信（例如matmul后的AllReduce）。
jit_matmul = jax.jit(matmul_square, out_shardings=P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

这将在任何分片下自动运行并跨我们的设备分割计算。**但在硬件层面实际发生了什么？**

1. 首先我们创建跨设备分片的In和W<d-footnote>注意我们是如何做到这一点的。这是创建具有特定分片的数组的一种方式（即通过向创建函数添加device参数）。另一种是用`jnp.array(....)`正常创建数组，然后做例如`jax.device_put(..., P('x', 'y'))`。还有一种是写一个创建你想要的数组的函数，并用`out_shardings`为你想要的jit编译它。</d-footnote>。W在收缩维度上2路分片，而In是4路分片（沿收缩和输出维度）。这对应于分片W[D<sub>X</sub>, F]和In[B<sub>X</sub>, D<sub>Y</sub>]，即一种模型和数据并行性。
2. 如果我们在本地运行这个（即在一个设备上），`matmul_square`将简单地平方输入并执行简单matmul。但因为我们将`out_shardings`指定为`P('X', None)`，我们的输出将沿批次分片但在模型维度上复制，需要AllReduce来计算。

使用我们前面章节的记号，这可能会做类似

1. Out[B<sub>X</sub>, F] { U<sub>Y</sub> } = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]
2. Out[B<sub>X</sub>, F] = **AllReduce**(Out[B<sub>X</sub>, F] { U<sub>Y</sub> })

`jax.jit`会自动为我们添加这个！我们实际上可以用`jit_matmul.as_text()`打印HLO并看到以下HLO（大幅简化）：

```py
# 这个融合是分片输入和矩阵的实际matmul
%fusion = bf16[4,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[4,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# 我们在设备间约简部分求和的结果
ROOT %AllReduce = bf16[4,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[4,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

我们可以看到上面的matmul（融合）和AllReduce。特别注意形状。`bf16[4, 1024]`是激活的本地视图，因为我们的`batch_size=8`在2个设备上分割，我们的`d_model=2048`同样2路分割。

**这相当神奇！** 无论我们的程序多复杂，GSPMD和jit都会尝试为所有中间激活找到分片并根据需要添加通信。话虽如此，GSPMD有其缺陷。它可能犯错。有时你会看到一个配置文件并注意到出了问题。一个巨大的AllGather占用了80%的配置文件，而它不需要。当这种情况发生时，我们可以尝试通过用`jax.lax.with_sharding_constraint`明确标注中间张量来纠正编译器。例如，对于两个matmul，我可以强制中间激活沿`y`维度分片（不是说这是好主意）：

```py
import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X', 'Y'))

def matmul(x, Win, Wout):
  hidden = jnp.einsum('bd,df->bf', x, Win)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.sharding.NamedSharding(mesh, P('x', 'y')))
  return jnp.einsum('bf,df->bd', hidden, Wout)
```

这构成了jit世界中JAX并行编程的大约60%，因为这是我们干预编译器的唯一方式。值得在Colab中玩玩`with_sharding_constraint`并了解它如何工作。当我们使用`jax.jit`写LLM时，我们90%的控制分片工作是通过`in_shardings`和`out_shardings`更改输入和输出分片，并用`with_sharding_constraint`标注中间张量以确保正确的通信正在发生。更多jax.jit示例，[这是一个很好的文档](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)。

<h3 id="shard-map-explicit-parallelism-control-over-a-program">shard_map: 对程序的显式并行性控制</h3>

虽然GSPMD是"编译器接管"模式，但jax [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html)将一切交给你。你指定输入的分片，就像在jax.jit中一样，但然后你明确地写所有通信。`jax.jit`为你留下程序的全局跨设备视图，而`shard_map`给你本地每设备视图。

这里是一个例子。尝试推理这个函数做什么：<d-footnote>如果你想在colab中通过模拟网格自己玩这个，你可以使用以下单元格 `import os; os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'`</d-footnote>

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

from jax.experimental.shard_map import shard_map as shmap

P = shd.PartitionSpec
shd.set_mesh(jax.make_mesh(axis_shapes=(2, 4), axis_names=('x','y')))

x = jnp.arange(0, 512, dtype=jnp.int32, device=P(('x', 'y')))

# 这个函数将在数组的1/8上操作。
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = shmap(slice_and_average, shd.get_abstract_mesh(), in_specs=P(('x', 'y')), out_specs=P(None,))(x)
assert out.shape == (4,)
```

**这做什么？** `slice_and_average`在每个TPU上运行，处理数组的1/8，从中我们切片前4个元素并在完整网格上平均它们。这意味着我们有效地做`mean(x[:4], x[64:68], x[128:132], …)`。这相当酷，因为这不是一个容易在JAX中表达的操作。

**为什么这样做而不是jax.jit？** 如果我们使用了`jax.jit`，`slice_and_average`会看到数组的全局视图（完整的`[512,]`数组）。我们必须切出这个不均匀的切片然后执行一个XLA必须正确解释的平均。XLA可能添加了错误的通信或感到困惑。这里我们看到本地视图并只写我们需要的通信。

**示例[集合Matmul]：** 举一个更现实的例子，说我们要实现模型并行性，其中激活最初是模型分片的，即A[B<sub>X</sub>, D<sub>Y</sub>] \* W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]。天真地，我们会通过先AllGather A然后进行本地矩阵乘法来做到这一点：

1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>])
2. Out[B<sub>X</sub>, F<sub>Y</sub>] = A[B<sub>X</sub>, D] *<sub>D</sub> W[D, F<sub>Y</sub>]

遗憾的是，这不好，因为它不允许我们将通信与计算重叠。重叠它们可以通过"集合matmul"来完成，如[Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959)中所述。算法基本上如下：

* 对于每个Y分片，执行A的本地块与W的本地块的matmul，产生形状`[B / X, F / Y]`的结果。同时，置换A使你在本地得到下一个块，执行matmul，并对结果求和。

我们可以用shard\_map相当容易地实现：

```py
import functools

import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np

from jax.experimental.shard_map import shard_map

mesh = jax.make_mesh(axis_shapes=(2, 4), axis_names=('X', 'Y'))
def P(*args):
  return shd.NamedSharding(mesh, shd.PartitionSpec(*args))

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, P('X', 'Y'))
W = jax.device_put(W, P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  # lhs是循环操作数；rhs是本地操作数
  axis_size = jax.lax.psum(1, axis_name='Y')  # 对于这个例子axis_size = 4
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # 对一个块进行Matmul
    update = lhs @ rhs_chunk
    # 向左循环移位
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # 在最终置换后计算最后一个块，让lhs保持我们找到它的状态
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(shard_map(
  collective_matmul_allgather_lhs_contracting, mesh,
  in_specs=(shd.PartitionSpec('X', 'Y'), shd.PartitionSpec(None, 'Y')), out_specs=shd.PartitionSpec('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)
```

这相当棒！我们可以基准测试这个并看到它也快很多！[这里](https://imgur.com/a/e9I6SrM)是默认jit matmul的配置文件，用时311us，开始有一个大的阻塞AllGather：

{% include figure.liquid path="assets/img/not-overlapped.png" class="img-fluid" %}

[这里](https://imgur.com/a/21iy0Sv)是上面的版本，用时244us。你可以看到配置文件没有AllGather。全是有用的工作！我们的FLOPs利用率也高得多。

{% include figure.liquid path="assets/img/overlapped.png" class="img-fluid" %}

还值得注意的是，在收缩维度上没有分片的matmul时间是[224us](https://imgur.com/a/i3gNKfq)，所以我们在这里非常接近未分片基线。这是你可能最终做的性能工程的一个好例子，以改善TPU利用率。更多`shard_map`示例，[这个笔记很棒](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side)。

现在这里有一些有用的实践问题，尝试使用`jax.jit`或`shard_map`实现！

## 实践问题

这里有一些随机的JAX相关问题。我稍后会添加更多。对于所有这些，你需要在Colab中有一些TPU。你可以使用带TPUv2-8的公共Colab。从现在开始，我们假设你有N个可用设备。

**问题1：** 设**A**为形状float32[S<sub>X</sub>, D<sub>Y</sub>]的激活数组，其中`X * Y = N`。做以下事情：

1. 在JAX中写一个函数，计算每个`(X, Y)`分片内的平均值，即它返回一个大小为[X, Y]的数组，其中`arr[i, j]`是分片`(i, j)`的平均值。用`jax.jit`和`shard_map`都做。分析每个并看它们用了多长时间。有添加任何通信吗？*提示：不应该有，但有时XLA反正会添加。*

2. 在JAX中写一个函数，返回roll(x, shift, axis=0) - x对某个shift **在每个分片X内**。我不够虐待狂让你在jax.jit中做这个，所以只用`shard_map`做。

{% details 点这里看答案。 %}

第1部分：这里是第1部分的解决方案。注意我们必须为`jax.jit`解决方案做的相当复杂的reshape。

```py
import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import shard_map

P = jax.sharding.PartitionSpec

mesh = jax.make_mesh((4, 2), ('X','Y'))

average_shmap = shard_map.shard_map(
    lambda x: x.mean(keepdims=True), 
    mesh=mesh, 
    in_specs=P('X','Y'), out_specs=P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

第2部分：这里是第2部分的类似解决方案。

```py
import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import shard_map

import functools

P = jax.sharding.PartitionSpec

mesh = jax.make_mesh((4, 2), ('X','Y'))

def shift_shmap(x, shift: int):
  shmapped = shard_map.shard_map(
      lambda x: jnp.roll(x, shift, axis=0), 
      mesh=mesh, 
      in_specs=P('X','Y'), out_specs=P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return jnp.roll(reshaped, shift, axis=1).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```

{% enddetails %}

**问题2：** 这里我们将一起制作一个基本的"专家混合"模型。设**W**: float32[E<sub>X</sub>, D, F<sub>Y</sub>]为E个"专家"矩阵的集合。设**A**: float32[S<sub>X</sub>, D<sub>Y</sub>]（我们的激活），设**B**为一组"路由分配"，其中B[i]是范围`[0, E)`中的整数，告诉我们想要处理该激活的矩阵。我们想要在JAX中写一个返回`Out[i] = W[B[i]] @ A[i]`的函数。

1. 让我们开始完全忽略分片。使所有这些张量都小到足以装入一个设备。写一个这个函数的本地实现。*确保你不要具体化形状`[S, D, F]`的数组！提示：尝试将token排序到形状`[E, S, D]`的新缓冲区中，注意掩码（为什么我们需要第二维有大小S？）。*

2. 如果你只是`jax.jit`上述方法，会发生一些事情。分析这个并看它决定做什么通信。用了多长时间？

3. 你会注意到上述的一个问题是它可能在本地收集完整的激活集合**A**，即AllGather<sub>X</sub>([S<sub>X</sub>, D<sub>Y</sub>])，这不仅在通信方面昂贵，如果我们不能在本地装下完整激活集合，在内存方面也极其昂贵。使用`shard_map`和显式通信实现上述。

      1. 第一遍，使用`jax.lax.all_gather`并像(a)中那样重新排序可能是最容易的。

      2. 第二遍，尝试避免具体化任何大小`[E, S, D]`的数组，即尝试在`jax.lax.while_loop`内使用`jax.lax.all_to_all`以不规则方式执行计算。这样，你可以避免具体化完整激活并浪费填充上的计算。这比你的原始实现快多少？

4. 大多数MoE路由到多个(k)专家然后对结果求平均。重构上述以实现这个。在这种情况下设**B**: int32[S, k]为要路由到的k个专家。

**问题3：** 上面的集合matmul例子实际上与真实LLM超级相关。让我们调整例子来做完整的Transformer栈。

1. 作为练习，让我们从实现AllReduce集合matmul开始，即A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] -> Out[B<sub>X</sub>, F]。注意输出不是复制的。如上所述，天真算法基本上只是本地matmul后跟AllReduce。尝试制作这个操作的通信重叠"集合"版本。*提示：在输出维度上分片，随意使用`jax.lax.psum`（aka AllReduce）。* *注意：由于XLA处理这个的方式，它实际上可能不比基线快。*

2. 上面AllReduce集合matmul的补充是ReduceScatter集合matmul，如Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W2[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]。这发生在Transformer的下投影矩阵中。在JAX中实现这个的集合、重叠版本。小心只传递你需要的最小数据量。*提示：尝试在累积时置换结果。*

3. 将这两个组合成端到端Transformer块，执行In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]，带重叠通信。<d-footnote>如前所述，我们不能先做$W_{in} \cdot W_{out}$，因为我们这里省略了非线性。</d-footnote> 这比`jax.jit`实现快多少？

**问题4：** 上面实现的所有集合matmul都是单向的：它们只在一个方向上置换。重写集合AllReduce matmul和集合ReduceScatter matmul以使用双向通信。这些快多少？

### 第10部分结束。基本就是这样！关于最终结论和进一步阅读，点击[这里](../conclusion)。