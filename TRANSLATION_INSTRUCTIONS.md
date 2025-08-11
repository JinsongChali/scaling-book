# 《如何扩展你的模型》中文翻译指南

## 项目概述
本项目是将Google DeepMind出版的技术书籍《How To Scale Your Model》翻译成中文。该书详细介绍了在TPU和GPU上扩展大型语言模型/Transformers的技术细节。

## 翻译执行步骤

### 第一步：环境准备
1. 确保在 `/mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/` 目录下工作
2. 中文版本存储在 `Chinese version/` 子目录中
3. 已有4个章节部分翻译完成，需要完成剩余9个章节

### 第二步：待翻译文件清单（按章节顺序）

| 章节 | 英文文件 | 中文文件路径 | 状态 |
|------|---------|-------------|------|
| 0 | index.md | Chinese version/index.md | ✅ 已翻译 |
| 1 | roofline.md | Chinese version/roofline.md | ✅ 已翻译 |
| 2 | tpus.md | Chinese version/tpus.md | ❌ 待翻译 |
| 3 | sharding.md | Chinese version/sharding.md | ❌ 待翻译 |
| 4 | transformers.md | Chinese version/transformers.md | ✅ 已翻译 |
| 5 | training.md | Chinese version/training.md | ❌ 待翻译 |
| 6 | applied-training.md | Chinese version/applied-training.md | ❌ 待翻译 |
| 7 | inference.md | Chinese version/inference.md | ❌ 待翻译 |
| 8 | applied-inference.md | Chinese version/applied-inference.md | ❌ 待翻译 |
| 9 | profiling.md | Chinese version/profiling.md | ✅ 已翻译 |
| 10 | jax-stuff.md | Chinese version/jax-stuff.md | ❌ 待翻译 |
| 11 | conclusion.md | Chinese version/conclusion.md | ❌ 待翻译 |
| 12 | gpus.md | Chinese version/gpus.md | ❌ 待翻译 |

### 第三步：翻译规范

#### 3.1 必须保留的元素
1. **YAML Front Matter**：保持原有结构，仅翻译必要字段
   ```yaml
   layout: distill  # 保持不变
   title: "中文标题"  # 翻译
   description: "中文描述"  # 翻译
   section_number: X  # 保持不变
   date: YYYY-MM-DD  # 保持不变
   authors:  # 保持原作者信息
   toc:  # 翻译章节标题，保持结构
   ```

2. **特殊标记**：
   - `<d-cite key="reference"></d-cite>` - 保持原样
   - `<d-footnote>内容</d-footnote>` - 翻译内容部分
   - `{% include figure.liquid ... %}` - 仅翻译caption参数

3. **代码块**：保持原样，包括注释（可选择性翻译注释）

4. **数学公式**：LaTeX/MathJax格式保持原样

5. **链接**：
   - 外部链接：保持原样
   - 内部链接：更新为中文版本路径（如 `roofline.md` → `Chinese version/roofline.md`）

#### 3.2 术语翻译对照表

| 英文术语 | 中文翻译 | 备注 |
|---------|---------|------|
| Transformer | Transformer | 保持原文 |
| Attention | 注意力机制 | |
| Roofline | Roofline | 保持原文或"屋顶线模型" |
| TPU | TPU | 保持原文 |
| GPU | GPU | 保持原文 |
| Sharding | 分片 | |
| Parallelism | 并行 | |
| Data Parallelism | 数据并行 | |
| Model Parallelism | 模型并行 | |
| Pipeline Parallelism | 流水线并行 | |
| Tensor Parallelism | 张量并行 | |
| FSDP | FSDP | 保持原文 |
| ZeRO | ZeRO | 保持原文 |
| Megatron | Megatron | 保持原文 |
| JAX | JAX | 保持原文 |
| XLA | XLA | 保持原文 |
| FLOP/s | FLOP/s | 保持原文 |
| Batch Size | 批次大小 | |
| Sequence Length | 序列长度 | |
| Hidden Dimension | 隐藏维度 | |
| Embedding | 嵌入 | |
| Layer Norm | 层归一化 | |
| Softmax | Softmax | 保持原文 |
| Activation | 激活函数 | |
| Checkpoint | 检查点 | |
| Gradient | 梯度 | |
| Optimizer | 优化器 | |
| Learning Rate | 学习率 | |
| Loss | 损失 | |
| Forward Pass | 前向传播 | |
| Backward Pass | 反向传播 | |
| All-Reduce | All-Reduce | 保持原文或"全规约" |
| All-Gather | All-Gather | 保持原文或"全收集" |
| Reduce-Scatter | Reduce-Scatter | 保持原文或"规约分散" |
| All-to-All | All-to-All | 保持原文或"全对全" |

#### 3.3 翻译风格指南

1. **语言风格**：
   - 使用专业技术语言，避免口语化
   - 保持学术论文的严谨性
   - 对于首次出现的重要术语，使用格式：`中文翻译（English Term）`

2. **句子结构**：
   - 保持原文的逻辑结构
   - 长句可适当拆分，但不改变原意
   - 技术描述要准确，不能有歧义

3. **图表处理**：
   - 图片文件路径保持不变
   - 仅翻译图片说明文字（caption）
   - 图表中的英文标注暂时保持原样

### 第四步：翻译执行流程

对于每个待翻译文件，执行以下步骤：

1. **读取源文件**：
   ```bash
   # 使用Read工具读取英文原文
   Read file_path="/mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/[filename].md"
   ```

2. **创建中文版本**：
   ```bash
   # 使用Write工具创建中文文件
   Write file_path="/mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/Chinese version/[filename].md"
   ```

3. **翻译内容**：
   - 分段翻译，保持原有格式
   - 特别注意YAML头部信息的正确处理
   - 保留所有技术标记和特殊格式

4. **质量检查**：
   - 确保所有链接正确更新
   - 验证数学公式和代码块未被破坏
   - 检查术语一致性

### 第五步：参考已翻译章节

查看已完成的翻译以保持风格一致性：
- `Chinese version/index.md` - 介绍章节
- `Chinese version/roofline.md` - Roofline分析
- `Chinese version/transformers.md` - Transformer数学
- `Chinese version/profiling.md` - 性能分析

### 第六步：翻译优先级

建议按以下顺序进行翻译：
1. **基础章节**（按原书顺序）：
   - tpus.md（第2章）
   - sharding.md（第3章）
   
2. **核心技术章节**：
   - training.md（第5章）
   - inference.md（第7章）
   
3. **应用实践章节**：
   - applied-training.md（第6章）
   - applied-inference.md（第8章）
   
4. **补充章节**：
   - jax-stuff.md（第10章）
   - conclusion.md（第11章）
   - gpus.md（第12章）

### 第七步：验证清单

每完成一个章节的翻译后，检查：
- [ ] YAML前置信息正确且完整
- [ ] 所有内部链接已更新为中文路径
- [ ] 术语翻译与对照表一致
- [ ] 数学公式显示正常
- [ ] 代码块格式正确
- [ ] 图片引用路径正确
- [ ] 脚注和引用标记完整
- [ ] 章节目录（toc）已翻译

## 执行命令示例

```bash
# 1. 读取英文原文
Read file_path="/mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/tpus.md"

# 2. 创建并写入中文翻译
Write file_path="/mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/Chinese version/tpus.md" content="[翻译后的内容]"

# 3. 验证翻译结果
Read file_path="/mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/Chinese version/tpus.md"
```

## 注意事项

1. **版权信息**：保留所有原作者信息和版权声明
2. **技术准确性**：遇到不确定的技术术语，优先保持原文
3. **格式保持**：严格保持Jekyll/Distill框架的格式要求
4. **增量保存**：建议每完成一个主要章节就保存进度
5. **链接一致性**：所有章节间的交叉引用需要统一更新

## 完成标准

- 所有9个待翻译章节完成中文版本
- 术语使用一致
- 格式与原文保持一致
- 内部链接全部更新
- 可以正常通过Jekyll构建生成网站

---

*此文档为Claude Code执行翻译任务的指导手册，请严格按照步骤执行以确保翻译质量和一致性。*