# 《如何扩展你的模型》中文翻译执行计划

## 执行概要
本执行计划为Claude Code提供具体的翻译步骤，确保高质量完成全书中文翻译。

## 翻译风格总结（基于已完成章节）

### 从已翻译章节观察到的风格特点：
1. **标题翻译**：保持简洁专业，如 "All About Rooflines" → "关于Roofline分析的一切"
2. **术语处理**：核心技术术语（如Roofline, Transformer, TPU）保持英文原文
3. **链接处理**：内部链接需要更新路径（如 `../tpus` 指向上级目录）
4. **YAML前置信息**：仅翻译title、description、toc部分，其他保持原样
5. **语言风格**：学术化、专业化，避免口语表达

## 待翻译章节清单（9个文件）

### 第一批：基础概念章节（2个文件）
1. **tpus.md** - "How to Think About TPUs"（如何理解TPU）
2. **sharding.md** - "Sharded Matrices and How to Multiply Them"（分片矩阵及其乘法）

### 第二批：训练相关章节（2个文件）
3. **training.md** - "How to Parallelize a Transformer for Training"（如何并行化Transformer进行训练）
4. **applied-training.md** - "Training LLaMA 3 on TPUs"（在TPU上训练LLaMA 3）

### 第三批：推理相关章节（2个文件）
5. **inference.md** - "All About Transformer Inference"（关于Transformer推理的一切）
6. **applied-inference.md** - "Serving LLaMA 3-70B on TPUs"（在TPU上服务LLaMA 3-70B）

### 第四批：实践与总结章节（3个文件）
7. **jax-stuff.md** - "Programming TPUs in JAX"（使用JAX编程TPU）
8. **conclusion.md** - "Conclusions and Further Reading"（结论与延伸阅读）
9. **gpus.md** - "How to Think About GPUs"（如何理解GPU）

## 具体执行步骤

### 步骤1：翻译tpus.md
```
任务：翻译第2章 - 如何理解TPU
1. Read file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/tpus.md
2. 翻译内容，保持技术准确性
3. Write file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/Chinese version/tpus.md
4. 验证格式和链接
```

### 步骤2：翻译sharding.md
```
任务：翻译第3章 - 分片矩阵及其乘法
1. Read file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/sharding.md
2. 翻译内容，注意数学公式保持原样
3. Write file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/Chinese version/sharding.md
4. 验证数学公式显示正确
```

### 步骤3：翻译training.md
```
任务：翻译第5章 - 如何并行化Transformer进行训练
1. Read file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/training.md
2. 翻译内容，保持并行化术语的准确性
3. Write file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/Chinese version/training.md
4. 验证代码块格式正确
```

### 步骤4：翻译applied-training.md
```
任务：翻译第6章 - 在TPU上训练LLaMA 3
1. Read file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/applied-training.md
2. 翻译内容，保持模型名称原样
3. Write file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/Chinese version/applied-training.md
4. 验证实践代码示例完整
```

### 步骤5：翻译inference.md
```
任务：翻译第7章 - 关于Transformer推理的一切
1. Read file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/inference.md
2. 翻译内容，注意推理相关术语
3. Write file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/Chinese version/inference.md
4. 验证技术细节准确
```

### 步骤6：翻译applied-inference.md
```
任务：翻译第8章 - 在TPU上服务LLaMA 3-70B
1. Read file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/applied-inference.md
2. 翻译内容，保持模型规格准确
3. Write file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/Chinese version/applied-inference.md
4. 验证服务配置示例正确
```

### 步骤7：翻译jax-stuff.md
```
任务：翻译第10章 - 使用JAX编程TPU
1. Read file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/jax-stuff.md
2. 翻译内容，JAX API保持原样
3. Write file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/Chinese version/jax-stuff.md
4. 验证代码示例完整性
```

### 步骤8：翻译conclusion.md
```
任务：翻译第11章 - 结论与延伸阅读
1. Read file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/conclusion.md
2. 翻译内容，保持参考文献原样
3. Write file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/Chinese version/conclusion.md
4. 验证延伸阅读链接有效
```

### 步骤9：翻译gpus.md
```
任务：翻译第12章 - 如何理解GPU
1. Read file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/gpus.md
2. 翻译内容，注意GPU相关术语
3. Write file: /mnt/c/Users/Mola/OneDrive/查理科技/北京查理科技有限公司/Book/scaling-book/Chinese version/gpus.md
4. 验证GPU架构描述准确
```

## 质量控制检查清单

每完成一个章节后，执行以下检查：

### 格式检查
- [ ] YAML前置信息正确（title, description, toc已翻译）
- [ ] section_number保持原值
- [ ] 作者信息未改动
- [ ] layout: distill 保持不变

### 内容检查
- [ ] 术语翻译与对照表一致
- [ ] 数学公式（$...$和$$...$$）未被破坏
- [ ] 代码块格式正确（```语言标记正确）
- [ ] 脚注（<d-footnote>）内容已翻译
- [ ] 引用（<d-cite>）标记保持原样

### 链接检查
- [ ] 内部链接更新为中文路径
- [ ] 图片路径保持原样（assets/img/...）
- [ ] 外部链接保持原样
- [ ] 章节导航链接正确（previous/next）

### 特殊元素
- [ ] figure.liquid的caption参数已翻译
- [ ] 表格内容已翻译但格式保持
- [ ] 列表格式保持原样
- [ ] 特殊HTML标记未被破坏

## 翻译原则提醒

1. **准确性优先**：宁可保留英文术语，也不要错误翻译
2. **格式保持**：严格保持Jekyll/Distill框架要求
3. **一致性**：参考已翻译章节的风格和术语
4. **可读性**：中文表达要流畅自然，符合技术文档规范
5. **完整性**：不遗漏任何内容，包括图片说明、脚注等

## 预期完成时间

- 每个简单章节（如conclusion）：约30-45分钟
- 每个复杂章节（如training）：约60-90分钟
- 总计预估：9-12小时完成全部翻译

## 开始执行

现在可以按照上述步骤顺序执行翻译任务。建议：
1. 从tpus.md开始，因为它是基础概念章节
2. 每完成2-3个章节后进行整体一致性检查
3. 保持术语表更新，确保全书术语统一

---

*此执行计划为Claude Code的操作指南，请严格按照步骤执行，确保翻译质量。*