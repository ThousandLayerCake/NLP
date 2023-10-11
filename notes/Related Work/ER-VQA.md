# About Explanation / Reasoning VQA datasets

关于一类 解释/推理型VQA 的数据集







### CLEVR

论文地址：https://arxiv.org/pdf/1612.06890.pdf

**CLEVR 数据集的摘要：**[from:[Abstract](https://arxiv.org/pdf/1612.06890.pdf)]

> When building artificial intelligence systems that can reason and answer questions about visual data, we need diagnostic tests to analyze our progress and discover shortcomings. Existing benchmarks for visual question answering can help, but have strong biases that models can exploit to correctly answer questions without reasoning. They also conflate multiple sources of error, making it hard to pinpoint model weaknesses. We present a diagnostic dataset that tests a range of visual reasoning abilities. It contains minimal biases and has detailed annotations describing the kind of reasoning each question requires. We use this dataset to analyze a variety of modern visual reasoning systems, providing novel insights into their abilities and limitations.



**CLEVR 数据集的介绍：**[from:[A-OKVQA](https://arxiv.org/pdf/2206.01718v1.pdf)]

> In CLEVR [25], the image and question are automatically generated from templates and
> explicitly require models to go through multiple steps of reasoning to correctly answer. This dataset
> and similar datasets which rely on simulated images suffer from lack of visual realism and lack of
> richness in the images and questions and are thus prone to be overfit to with methods achieving
> nearly 100% accuracy [64]. Our dataset requires reasoning on real images and free-form language.



**小结**：虽然有推理，但是 **这个数据集和依赖于模拟图像的类似数据集** （这一类数据集）存在着缺乏 <u>视觉逼真度</u>、 <u>图像和问题的丰富性</u> 的问题。不够丰富，那就容易过拟合。





### 解释型数据集

- [数据集](https://arxiv.org/pdf/1801.09041.pdf) 在VQAv2上收集或提取了解释
- [数据集](https://arxiv.org/pdf/1802.08129.pdf) 在VQAv2上收集或提取了解释



**解释型数据集的介绍：**[from:[A-OKVQA](https://arxiv.org/pdf/2206.01718v1.pdf)]

> Other works [38, 28] have collected or extracted justifications on the VQAv2 [16] dataset. However,
> VQAv2 mostly focuses on questions about object attributes, counting and activities, which do not
> require reasoning on outside knowledge.