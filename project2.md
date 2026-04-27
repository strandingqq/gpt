项目 2：从零开始预训练 GPT 模型
项目说明
截止日期： 2026年5月10日 23:59（逾期提交每天扣除 1 分）
提交格式： 最终报告须为 PDF 格式。代码打包为：学号-姓名-project2.zip
提交链接：

代码上传至：https://epan.shanghaitech.edu.cn/l/b1HE3K

最终报告上传至：https://ecourse.shanghaitech.edu.cn/

数据集选项： hongloumeng.txt 或 TinyStories（二选一）。
GPU 资源： https://aistation2.shanghaitech.edu.cn:32206/

1. 引言
生成式预训练 Transformer (GPT) 模型在文本生成、少样本学习和推理方面展现出的卓越能力，彻底改变了自然语言处理领域。与将输入映射到标签的分类模型不同，GPT 是一种自回归语言模型，学习根据之前的上下文预测下一个 token。

在本项目中，你将使用 PyTorch 实现一个 GPT 模型，且不依赖高级 Transformer 库（例如 Hugging Face Transformers）。你将在文本语料库（TinyStories 或《红楼梦》）上进行预训练，尝试不同的模型大小，并分析生成文本的质量。最后，你可以完成一项进阶任务：架构优化（改进基础 Transformer 的某个方面）或 KV Cache 实现（加速推理）。

通过本项目，你将获得以下实践经验：

从零开始构建 Transformer 模块（如多头注意力机制、前馈网络）。

使用因果掩码实现自回归语言建模。

训练语言模型并生成连贯文本。

优化 Transformer 训练或推理的实用技术。

2. 基础任务
你的主要目标是使用 PyTorch 从零开始实现一个 GPT 模型，在选定的数据集上进行预训练，并比较不同的模型规模。

步骤：
2.1 数据集实现
选择一个数据集（hongloumeng.txt 或 TinyStories）。

对于 TinyStories：由于完整数据集非常大，你可以仅使用一个子集进行训练。子集的大小由你决定——选择适合你 GPU 显存和时间预算的数量。请在报告中明确说明你使用了多少样本。

对于《红楼梦》：该数据集相对较小。你可以单独使用它，也可以补充你选择的其他合适的中文文本数据。如果你添加了额外数据，请在报告中详细描述。

在 dataset.py 中实现自定义的 PyTorch Dataset 类。

创建训练集和验证集的切分。

2.2 模型实现
在 gpt.py 中实现你的 GPT 模型。禁止导入来自 torch.nn 或 Hugging Face 的预构建 Transformer 层。

所需组件：

Token 嵌入层 (Token embedding layer)

位置编码 (Positional encoding)

多个 Transformer 解码器块，每个块包含：

多头因果自注意力机制（带掩码注意力）

前馈网络

残差连接和归一化

最后的层归一化 (Layer Norm) 和线性输出层 (Linear head)（权重绑定可选）

实现不同的模型规模。

2.3 训练
在选定数据集上训练模型。

监控训练和验证的交叉熵损失、验证集困惑度 (perplexity)，并在验证过程中生成样本文本。

保存训练日志（损失曲线、验证困惑度、样本文本）。

2.4 生成与评估
实现一个具有可调温度 (temperature) 和 top-k/top-p 采样的文本生成函数。

对每个模型使用相同的提示词 (prompt) 至少生成 3 个文本样本。

对模型进行定性评估（连贯性、创造性、重复度）和定量评估（验证集上的困惑度）。

2.5 分析
从以下方面比较不同模型规模：

验证集困惑度（越低越好）

训练时间和内存占用

生成文本的质量（提供示例）

3. 进阶任务（二选一）
选择以下两项进阶任务之一。

3.1 任务 A – 架构优化
在以下至少一个维度改进基础 Transformer 架构：生成质量、训练速度、推理速度、GPU 显存占用或其他适用的指标。

要求：
选择你最大的模型作为基准 (baseline)。

模型优化必须是 Transformer 架构的结构性修改（不仅仅是超参数调整）。

在相同的验证集上训练优化后的模型，并与基准进行对比。

报告指标：困惑度、每个 epoch 的训练时间、每个 token 的推理时间、峰值显存占用（根据你的优化目标选择相关指标）。

在报告中解释该修改为何有效（或无效）。

3.2 任务 B – KV Cache 实现
为自回归生成实现 KV Cache，并测量其对推理速度的影响。

要求：
修改你最大的模型以支持 KV Cache。

KV Cache 存储来自之前 token 的 Key 和 Value 张量，这样在生成每个新 token 时，你只需计算新 token 的注意力，而无需对整个序列重新计算。

对比： 测量使用和不使用 KV Cache 的推理时间（每生成一个 token 所需的秒数或生成固定长度所需的总时间）。使用相同的提示词和生成参数。

4. 报告要求
你的 PDF 报告必须包含以下章节：

引言

数据集与预处理

描述你选择的数据集以及添加的任何额外数据。

解释你的分词方法和词汇表大小。

模型架构

描述你从零实现的 GPT 架构。

包含每种模型规模的超参数表。

对于进阶任务 A：清晰展示基准架构和修改后的版本，突出精确的结构变化。

训练设置

列出训练超参数：学习率、批大小、epoch 数等。

结果

每个模型的训练曲线（损失和困惑度）。

每个模型生成的样本文本（至少 3 个提示词）。

对于进阶任务：基准模型与优化/缓存模型之间相关指标（困惑度、训练速度、推理速度、显存占用）的对比表。

对比分析与讨论

基础任务：比较不同模型规模，讨论增加容量如何影响损失、困惑度、生成质量和训练速度。

进阶任务：分析你的修改带来的影响——是否达到了预期的改进？为什么？

结论

总结关键发现。

参考文献（如有）

5. 代码要求
所有代码必须结构清晰，并包含注释。

代码打包为：学号-姓名-project2.zip

不要包含数据集和模型检查点 (checkpoints)。

6. 资源
Radford 等人, 2018. 通过生成式预训练改进语言理解 (GPT-1)

Vaswani 等人, 2017. 注意力就是一切 (Attention Is All You Need)

# Project 2: Pre-training a GPT Model from Scratch

**Project Instruction**  
**Deadline:** 2026-5-10 23:59 (Late submissions will be penalized by 1 point per day)  
**Submission Format:** Final report must be in PDF. Code zipped as: studentID-name-project2.zip  
**Submission Link:**  
- upload code to https://epan.shanghaitech.edu.cn/l/b1HE3K
- upload final report to https://ecourse.shanghaitech.edu.cn/  

**Dataset Options:** `hongloumeng.txt` or [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) (choose one).
**GPU Resources:** https://aistation2.shanghaitech.edu.cn:32206/

## 1. Introduction

Generative Pre-trained Transformer (GPT) models have revolutionized natural language processing by demonstrating remarkable capabilities in text generation, few-shot learning, and reasoning. Unlike classification models that map input to a label, GPT is an autoregressive language model that learns to predict the next token given the previous context. 

In this project, you will implement a GPT model using PyTorch, without relying on high-level transformer libraries (e.g., Hugging Face Transformers). You will pre-train it on a text corpus (TinyStories or hongloumeng), experiment with different model sizes, and analyze the generated text quality. Finally, you may complete an **advanced task**: either **architectural optimization** (improving some aspect of the base Transformer) or **KV cache implementation** (accelerating inference).

Through this project, you will gain hands-on experience with:
- Building transformer blocks from scratch (e.g., multi-head attention, feed-forward networks).
- Implementing autoregressive language modeling with causal masking.
- Training language models and generating coherent text.
- Practical techniques for optimizing Transformer training or inference.

## 2. Basic Task

Your primary goal is to implement a GPT model from scratch in PyTorch, pre-train it on a chosen dataset, and compare different model sizes.

### Steps:

#### 2.1 Dataset Implementation
- Choose a dataset (hongloumeng.txt or TinyStories). 
   - For TinyStories: Because the complete dataset is very large, you may use only a subset for training. The size of the subset is up to you – choose an amount that fits your GPU memory and time budget. Please clearly state in your report how many samples you used.
   - For hongloumeng: This dataset is relatively small. You may either use it alone or supplement it with other suitable Chinese text data of your choice. If you add extra data, describe it clearly in your report.
- Implement a custom PyTorch `Dataset` class in `dataset.py`.
- Create training and validation splits.

#### 2.2 Model Implementation
- Implement your GPT model in `gpt.py`. **Do not** import pre-built transformer layers from `torch.nn` or Hugging Face.
- Required components:
  - Token embedding layer
  - Positional encoding
  - Multiple transformer decoder blocks, each containing:
    - Multi-head causal self-attention (with masked attention)
    - Feed-forward network
    - Residual connections and normalization
  - Final layer norm and linear head (tied weights optional)
- Implement different model sizes. 

#### 2.3 Training
- Train models on the chosen dataset.
- Monitor the training and validation cross-entropy loss, validation perplexity, and generate sample text during validation.
- Save training logs (loss curves, validation perplexity, sample text).

#### 2.4 Generation and Evaluation
- Implement a text generation function with adjustable temperature and top‑k/top‑p sampling.
- Generate at least 3 text samples from each model using the same prompt.
- Evaluate models qualitatively (coherence, creativity, repetition) and quantitatively (perplexity on validation set).

#### 2.5 Analysis
- Compare different model sizes in terms of:
  - Validation perplexity (lower is better)
  - Training time and memory usage
  - Quality of generated text (provide examples)

## 3. Advanced Tasks (Choose ONE)

Select one of the following two advanced tasks. 

### 3.1 Task A – Architecture Optimization

Improve the **base Transformer architecture** in at least one of the following dimensions: **generation quality, training speed, inference speed, GPU memory usage**, or **other applicable metrics**. 

#### Requirements:

- Choose your largest model as the baseline. 
- The model optimization must be a **structural modification** to the Transformer architecture (not just hyperparameter tuning). 
- Train the optimized model and compare against the baseline on the same validation set. 
- Report metrics: perplexity, training time per epoch, inference time per token, peak memory usage (whichever is relevant to your optimization goal).
- Analyze why the modification helped (or did not help) and provide an explanation in your report.

### 3.2 Task B – KV Cache Implementation

Implement **KV cache** for autoregressive generation and measure its impact on inference speed.

#### Requirements:

- **Modify your largest model** to support KV caching. 
   - The KV cache stores the key and value tensors from previous tokens, so that at each new token, you only compute attention for the new token instead of re‑computing over the entire sequence.
- **Comparison:** Measure the inference time (seconds per generated token or total time to generate a fixed length) **with** and **without** KV cache. Use the same prompt and generation parameters.

## 4. Report Requirements

Your PDF report must include the following sections:

1. **Introduction**  

2. **Dataset & Preprocessing**  
   - Describe the dataset you chose and any additional data you added.
   - Explain your tokenization method and vocabulary size.

3. **Model Architectures**  
   - Describe the GPT architecture you implemented from scratch. 
   - Include the hyperparameter tables for each model sizes.
   - For the Advanced Task A: Clearly present the baseline architecture and the modified version. Highlight the exact structural change.

4. **Training Setup**  
   - List the training hyperparameters: learning rate, batch size, number of epochs, etc.

5. **Results**  
   - Training curves (loss and perplexity) for each model.
   - Sample generated text (at least 3 prompts) from each model.
   - For the Advanced Task: comparison table of relevant metrics (perplexity, training speed, inference speed, memory usage) between baseline and optimized/cached model.

6. **Comparative Analysis and Discussion**  
   - For the Basic Task: compare different model sizes, discuss how increasing capacity affects loss, perplexity, generation quality, and training speed.
   - For the Advanced Task: analyze the impact of your modification – did it achieve the intended improvement? Why or why not?

7. **Conclusion**  
   - Summarize key findings.

8. **References** (if any)

## 5. Code Requirements

- All code must be well-structured, and commented.
- Zip your code as: `studentID-name-project2.zip`
- Do **not** include the dataset and the model checkpoints.

## 6. Resources

- Radford et al., 2018. [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT-1)
- Vaswani et al., 2017. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)