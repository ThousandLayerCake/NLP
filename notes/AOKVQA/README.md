# 关于AOKVQA的论文写作摘抄

论文地址：https://arxiv.org/pdf/2206.01718v1.pdf



# 介绍

**A-OKVQA数据集的介绍：**[from:Intro]

> We introduce A-OKVQA, a crowdsourced dataset composed of a diverse set of about 25K questions requiring a broad base of commonsense and world knowledge to answer. In contrast to the existing knowledge-based VQA datasets, the questions generally cannot be answered by simply querying a knowledge base, and instead require some form of commonsense reasoning about the scene depicted in the image.







## 背景

**VQA任务各个方面的研究背景：** [from:Intro]

> Since the VQA problem was formulated, many of these aspects have been studied. Early datasets mostly studied the perception and language understanding problem on natural image datasets [2, 34, 16]. Other datasets studied complex chains of reasoning about procedurally generated images [25]. More recently, datasets include questions which require factual [36, 56, 57] or commonsense knowledge [66].



**数据集提出背景：**[from:Intro]

> With the advent of large-scale pre-training of vision and language models [67, 62, 32, 33, 12, 43, 8] and other breakthroughs in multi-modal architectures, much of the low-hanging fruit in the field has been plucked and many of the benchmark datasets have seen saturated performance. Even performance on the newer knowledge-based datasets has been improved by such models [67]. 



**总结：**自从VQA任务被提出来后，许多人因为研究目的，提出了许多数据集[eg. 语言理解、复杂推理、基于事实、基于常识]。但是随着LLMs的突破，`the low-hanging fruit`（容易实现的目标）已经被摘取（实现），许多基准数据集的性能已经达到饱和。











