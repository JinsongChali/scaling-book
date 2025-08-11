# How To Scale Your Model | 如何扩展你的模型

A comprehensive guide to scaling Large Language Models (LLMs) on TPUs, now available in both English and Chinese.

这是一本关于在 TPU 上扩展大型语言模型（LLM）的综合指南，现已提供英文和中文版本。

🌐 **Live Site**: [https://JinsongChali.github.io/scaling-book](https://JinsongChali.github.io/scaling-book)

## About This Book | 关于本书

This book aims to demystify the science behind scaling language models. It covers:
- How TPUs (and GPUs) work and communicate with each other
- How LLMs run on real hardware
- How to parallelize models during training and inference for efficient scaling

本书旨在揭开语言模型扩展背后的科学原理，内容包括：
- TPU（和 GPU）的工作原理及相互通信方式
- LLM 在真实硬件上的运行机制
- 如何在训练和推理期间并行化模型以实现高效扩展

## Language Support | 语言支持

The book is available in two languages:
- **English**: Default version at the root URL
- **Chinese (中文)**: Available via language switcher or at `/chinese-version/`

本书提供两种语言版本：
- **英文**：默认版本，位于根 URL
- **中文**：可通过语言切换器访问，或直接访问 `/chinese-version/`

## Table of Contents | 目录

1. **Introduction** | 简介
2. **Roofline Analysis** | Roofline 分析
3. **TPUs** | TPU
4. **Sharding** | 分片
5. **Transformers** | Transformer
6. **Training** | 训练
7. **Applied Training** | 应用训练
8. **Inference** | 推理
9. **Applied Inference** | 应用推理
10. **Profiling** | 性能分析
11. **JAX Stuff** | JAX 相关
12. **Conclusion** | 结论
13. **GPUs** | GPU

## Running Locally | 本地运行

To build and run this book locally:

```bash
# Clone the repository
git clone https://github.com/JinsongChali/scaling-book.git
cd scaling-book

# Install dependencies
bundle install

# Run the Jekyll server
bundle exec jekyll serve
```

The book will be available at `http://localhost:4000/scaling-book`

### Mac OS Requirements
If you're on Mac OS, you may need to install additional dependencies:
```bash
brew install imagemagick
brew install ruby
pip install jupyter
git config safe.bareRepository all
```

## Deployment | 部署

This site is configured for GitHub Pages. To deploy:

1. Fork this repository
2. Go to Settings → Pages
3. Set Source to "Deploy from a branch"
4. Select `gh-pages` branch and `/ (root)` folder
5. Save and wait for deployment

For manual deployment (with repo write permission):
```bash
sh bin/deploy
```

## Contributing | 贡献

We welcome contributions in both English and Chinese! 

欢迎用英文和中文贡献内容！

### How to Contribute | 如何贡献

1. **Report Issues**: Leave a comment on the website (powered by Giscus) or open a GitHub issue
2. **Submit PRs**: Fork the repository and submit pull requests
3. **Contact**: Email jaaustin [at] google [dot] com

### Contributor License Agreement
To contribute on GitHub, you will need to sign a Google "Contributor License Agreement" (CLA):
https://cla.developers.google.com/clas

## Acknowledgments | 致谢

### Original Authors
This book was written by Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh and Albert Webson at Google DeepMind. Many of the ideas were first derived by James Bradbury and Reiner Pope.

### Chinese Translation
The Chinese version was translated and adapted to help more developers understand LLM scaling techniques.

中文版本经过翻译和改编，以帮助更多开发者理解 LLM 扩展技术。

### Website Theme
The website uses a Distill-style Jekyll theme created by [alshedivat/al-folio](https://github.com/alshedivat/al-folio) and the Distill team.

## Citation | 引用

For attribution in academic contexts, please cite this work as:

```
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.
```

BibTeX citation:

```bibtex
@article{scaling-book,
  title = {How to Scale Your Model},
  author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and Chen, Charlie and Vikram, Sharad and Lebron, Federico and Choy, Peter and Ramasesh, Vinay and Webson, Albert and Pope, Reiner},
  publisher = {Google DeepMind},
  howpublished = {Online},
  note = {Retrieved from https://JinsongChali.github.io/scaling-book/},
  year = {2025}
}
```

## License | 许可证

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

![dragon](assets/img/dragon.png)

*This book was originally called "How To Scale Your Dragon", after the Dreamworks film, hence the dragon imagery.*

*本书最初名为"How To Scale Your Dragon"，灵感来自梦工厂电影，因此使用了龙的图像。*