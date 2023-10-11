# About Knowledge-based VQA datasets

一类 基于外部知识 的VQA（KB-VQA）数据集





### KB-VQA

论文地址：https://arxiv.org/pdf/1511.02570.pdf

**KB-VQA数据集的介绍：**[from:[A-OKVQA](https://arxiv.org/pdf/2206.01718v1.pdf)]

> The earliest explicitly knowledge-based VQA datasets were KB-VQA [56] and FVQA [57]. While these benchmarks did specifically require knowledge for questions, the knowledge required for these benchmarks is completely “closed”.





### FVQA

**FVQA数据集的介绍：**[from:[A-OKVQA](https://arxiv.org/pdf/2206.01718v1.pdf)]

> FVQA [57] is annotated by selecting a triplet from a fixed knowledge graph. This forces the questions to require knowledge, but because the question is written based on this knowledge, these questions are fairly trivial once the knowledge is known and do not require much reasoning. In addition, the knowledge required is explicitly closed to the knowledge graphs used to generate the dataset, so these datasets can only test knowledge retrieval on those specific graphs. KVQA [48] is based on images in Wikipedia articles. Because of the source of the images, these questions tend to mostly test recognizing specific named entities (e.g., Barrack Obama) and then retrieving Wikipedia knowledge about that entity rather than commonsense knowledge.

**小结**：意思是由于问题都是根据知识来编写的，而知识是静态的，一旦掌握了这些知识，问题对于模型来说就是小菜一碟，不需要太多推理。





### KVQA

**KVQA数据集的介绍：**[from:[A-OKVQA](https://arxiv.org/pdf/2206.01718v1.pdf)]

> KVQA [48] is based on images in Wikipedia articles. Because of
> the source of the images, these questions tend to mostly test recognizing specific named entities (e.g.,
> Barrack Obama) and then retrieving Wikipedia knowledge about that entity rather than commonsense
> knowledge.

**小结：**KVQA将用于检索的数据从文本转为图像，根据问题抽取实体，然后在图片识别实体，用实体去检索维基百科知识。[动态]





### OK-VQA

**OK-VQA数据集的介绍：**[from:[A-OKVQA](https://arxiv.org/pdf/2206.01718v1.pdf)]

> Most similar to our work is OK-VQA [36]. This dataset was an improvement over prior work in terms
> of scale, and the quality of questions and images. It also has the property that the required knowledge
> was not “closed” or explicitly drawn from a particular source, and could be called “open”-domain
> knowledge. While this is an improvement over the previous works, it still suffers from problems
> which we address in this work. The knowledge required, while “open” is still biased towards simple
> lookup knowledge (e.g., what is the capital of this country?) and most questions do not require much
> reasoning. In contrast, our dataset is explicitly drawn to rely on more common-sense knowledge and
> to require more reasoning to solve. In addition, our dataset includes “rationale” annotations, which
> allow knowledge-based VQA systems to more densely annotate their knowledge acquisition and
> reasoning capabilities.

**小结**：OK-VQA所需要的知识不是“封闭”的或者静态的，但仍存在大多数问题不需要推理，只是一些简单的查找。



### S3VQA

**S3VQA数据集的介绍：**[from:[A-OKVQA](https://arxiv.org/pdf/2206.01718v1.pdf)]

> S3VQA [23] analyzes OK-VQA and creates a new dataset which includes
> questions that require detecting an object in the image, replacing the question with the word for that
> object and then querying the web to find the answer. Like OK-VQA, it even more explicitly has the
> problem of questions usually requiring a single retrieval rather than much commonsense knowledge
> or reasoning.





### Visual Commonsense Reasoning (VCR)

**VCR数据集的介绍：**[from:[A-OKVQA](https://arxiv.org/pdf/2206.01718v1.pdf)]

> Another related line of work is Visual Commonsense Reasoning (VCR) [66] and VisualCOMET [39].
> VCR is also a VQA dataset, but is collected from movie scenes and is quite focused on humans
> and their intentions (e.g. “why is [PERSON2] doing this”), whereas our dataset considers questions
> and knowledge about a variety of objects.





### Ads Dataset

广告数据集论文地址：https://openaccess.thecvf.com/content_cvpr_2017/papers/Hussain_Automatic_Understanding_of_CVPR_2017_paper.pdf



**Ads 数据集的介绍：**[from:[A-OKVQA](https://arxiv.org/pdf/2206.01718v1.pdf)]

> the Ads Dataset [22] is a dataset requiring knowledge
> about the topic and sentiments of the ads.







### KnowIT VQA

论文地址：https://arxiv.org/pdf/1910.10706.pdf

> considered knowledge-based question answering for a sitcom [14].



### WebQA

论文地址：https://arxiv.org/pdf/2109.00590.pdf

> considered by using web queries [9].