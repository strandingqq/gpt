# GPT 项目实施计划

## 项目目标
使用 PyTorch 从零实现 GPT 模型，在红楼梦文本上进行预训练，并实现文本生成。

---

## 任务清单

### 第一阶段：修复已有组件 Bug

| 步骤 | 内容 | 优先级 |
|------|------|--------|
| 1.1 | 修复 `CasualSelfAttention`：`self.num_heads` 未保存为实例变量 | P0 |
| 1.2 | 修复 standalone `scaled_dot_product_attention`：`softmax` 未定义 | P0 |
| 1.3 | 修复 `sotfmax` 函数：函数名拼写错误 + `torch,sum` 逗号 | P1 |
| 1.4 | 修复 `GPT` 类：`nn.Embedding()` 缺少参数、`context_length` 未定义 | P0 |
| 1.5 | 修复 `GPT` 类：position_embedding 的相加逻辑错误 | P0 |

### 第二阶段：完成 GPT 主模型

| 步骤 | 内容 | 优先级 |
|------|------|--------|
| 2.1 | 正确实现 Token Embedding 和 Position Embedding | P0 |
| 2.2 | 完成 `GPT.forward()` 并正确相加 token_emb + pos_emb | P0 |
| 2.3 | 考虑权重绑定（lm_head 与 token_embedding 共享） | P2 |

### 第三阶段：数据处理

| 步骤 | 内容 | 优先级 |
|------|------|--------|
| 3.1 | 选择分词方式（字符级，汉字 → token id） | P0 |
| 3.2 | 实现 `TextDataset` 类（滑动窗口采样） | P0 |
| 3.3 | 划分训练集 / 验证集 | P0 |
| 3.4 | 编写 `dataloader.py` 或在 notebook 中构建 DataLoader | P0 |

### 第四阶段：训练循环

| 步骤 | 内容 | 优先级 |
|------|------|--------|
| 4.1 | 设置 AdamW 优化器、学习率调度 | P0 |
| 4.2 | 实现训练循环（forward → loss → backward → step） | P0 |
| 4.3 | 添加验证步骤（计算验证 loss 和困惑度） | P0 |
| 4.4 | 保存训练日志（loss 曲线、样本文本） | P1 |

### 第五阶段：文本生成

| 步骤 | 内容 | 优先级 |
|------|------|--------|
| 5.1 | 实现 `generate()` 函数（自回归采样） | P0 |
| 5.2 | 支持 temperature 采样 | P0 |
| 5.3 | 支持 top-k 和 top-p（nucleus）采样 | P0 |
| 5.4 | 用相同 prompt 生成至少 3 个样本 | P0 |

### 第六阶段：分析评估

| 步骤 | 内容 | 优先级 |
|------|------|--------|
| 6.1 | 对比不同模型规模的验证困惑度 | P1 |
| 6.2 | 定性评估生成文本质量（连贯性、重复度） | P1 |
| 6.3 | 分析困惑度、训练时间、生成质量的关系 | P1 |

---

## 文件结构

```
e:\2026\DL\project2\
├── gpt.ipynb          # 所有模型代码（GPT + TransformerBlock + Attention + FFN）
├── dataset.py         # 数据集处理（分词 + Dataset 类）
├── train.py           # 训练脚本（可选）
├── schedule.md        # 本计划文件
├── hongloumeng.txt    # 数据集
└── project2.md        # 项目说明
```

---

## 推荐实现顺序

1. **先修 Bug**（第一阶段）—— 确保组件能正常运行
2. **完成 GPT 主模型**（第二阶段）—— 核心框架搭起来
3. **数据处理**（第三阶段）—— 字符级分词最简单
4. **训练循环**（第四阶段）—— 能跑起来看 loss 下降
5. **文本生成**（第五阶段）—— 看实际效果
6. **分析评估**（第六阶段）—— 汇报结果

---

## 当前进度

- [x] TransformerBlock 实现
- [x] CasualSelfAttention 实现（Bug 待修）
- [x] FeedForward / SwiGLUFeedForward 实现
- [x] LayerNorm 实现
- [x] GPT 类骨架（Bug 待修）
- [ ] 数据集处理
- [ ] 训练循环
- [ ] 文本生成
