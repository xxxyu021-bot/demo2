# Demo2 中文命名实体识别（NER）
## 项目介绍
分别使用bert预训练模型bert-base-chinese和chinese-bert-wwm，完成weibo和msra二个数据集中文实体识别任务，实现同时支持两种bert预训练模型和数据集，数据加载与预处理，模型设计与训练流程和模型评估与结果可视化。
## 数据集介绍

本项目使用两个中文命名实体识别数据集：

- **MSRA NER**：经典中文命名实体识别数据集，主要包含人名（PER）、地名（LOC）和组织机构名（ORG）三类实体。
- **Weibo NER**：基于中文微博文本构建的命名实体识别数据集，包含人名、地名、组织机构等多种实体类型。

数据集均划分为训练集、验证集和测试集，采用 BIO 标注格式
## 项目结构
```
├── config/
│   ├── msra_bert.json
│   ├── msra_wwm.json
│   ├── weibo_bert.json
│   └── weibo_wwm.json
├── data/
│   ├── msra/
│   │   ├── train.txt
│   │   ├── dev.txt
│   │   └── test.txt
│   └── weibo/
│       ├── train.txt
│       ├── dev.txt
│       └── test.txt
├── Remote/
├── swanlog/
├── config.py
├── dataset.py
├── metrics.py
├── model.py
├── train.py
└── utils.py
## 超参数设置
```

| 实验 | 数据集 | 预训练模型 | Batch Size | Epochs | 学习率 | 最大序列长度 | Dropout | Weight Decay | Warmup Ratio | Early Stopping | Seed |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | MSRA | bert-base-chinese | 8 | 6 | 2e-5 | 128 | 0.3 | 0.01 | 0.1 | 3 | 42 |
| 2 | MSRA | hfl/chinese-bert-wwm | 8 | 6 | 2e-5 | 128 | 0.3 | 0.01 | 0.1 | 3 | 42 |
| 3 | Weibo | bert-base-chinese | 8 | 6 | 2e-5 | 128 | 0.3 | 0.01 | 0.1 | 3 | 42 |
| 4 | Weibo | hfl/chinese-bert-wwm | 8 | 6 | 2e-5 | 128 | 0.3 | 0.01 | 0.1 | 3 | 42 |

训练采用 **AdamW** 优化器，并使用带 Warmup 的线性学习率调度策略；模型根据验证集 **F1** 保存最佳权重，并采用 Early Stopping 防止过拟合。
## 实验结果
