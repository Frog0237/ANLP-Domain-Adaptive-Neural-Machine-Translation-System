# Technical Notes: Domain-Adaptive NMT Pipeline

> 本文档记录项目的核心技术决策、设计理由，以及面试中高概率被追问的问题与参考答案。
> 适合面试前快速复习，也适合在 README 之外理解“为什么这么做”。

---

## 目录

1. [项目定位与整体架构](#1-项目定位与整体架构)
2. [数据处理模块](#2-数据处理模块)
3. [模型选型与训练策略](#3-模型选型与训练策略)
4. [推理引擎与解码策略](#4-推理引擎与解码策略)
5. [评估体系设计](#5-评估体系设计)
6. [Attention Coverage 诊断](#6-attention-coverage-诊断)
7. [错误分类体系](#7-错误分类体系)
8. [工程化实践](#8-工程化实践)
9. [面试高频问题集](#9-面试高频问题集)
10. [可扩展方向](#10-可扩展方向)

---

## 1. 项目定位与整体架构

### 做了什么

构建一套端到端的领域自适应神经机器翻译系统，覆盖：数据清洗与分词 → 模型训练（可从零训练或微调）→ 批量推理 → 多指标评估 → 注意力诊断 → 解码策略优化。

### 为什么不直接调用翻译 API

| 对比维度 | 调用 API | 自建 Pipeline |
|---------|---------|--------------|
| 领域适配 | 通用模型，领域文本质量不可控 | 可在领域语料上训练/微调 |
| 可解释性 | 黑盒 | 可提取 attention、定位问题 |
| 成本控制 | 按 token 计费 | 一次训练，无限推理 |
| 面试价值 | 展示“会用” | 展示“会做” |

### 为什么选 Encoder-Decoder（不用 Decoder-only LLM）

- **任务适配**：机器翻译是 seq2seq 任务，encoder-decoder 天然匹配
- **效率**：MarianMT 体量适中，单 GPU 即可训练/推理
- **可控性**：可直接读取 cross-attention 权重进行诊断

---

## 2. 数据处理模块

### 设计要点

```
src/data.py
├── load_parallel_corpus()      # 数据加载（支持多 split）
├── create_preprocess_fn()      # 闭包式 tokenization
└── prepare_datasets()          # 端到端：加载 + 分词
```

### 为什么用闭包（closure）而不是全局变量

```python
# 全局变量方案：线程不安全、难以测试
tokenizer = global_tokenizer
def preprocess(examples):
    return tokenizer(...)

# 闭包方案：每次调用返回独立函数，可组合、可测试
def create_preprocess_fn(tokenizer, **kwargs):
    def preprocess(examples):
        return tokenizer(...)
    return preprocess
```

闭包方案可以对不同 split 设置不同的 max_length，互不干扰。

---

## 3. 模型选型与训练策略

### MarianMT 架构

- 6 层 Encoder + 6 层 Decoder
- 约 74M 参数
- 使用 Helsinki-NLP/opus-mt-de-en 的 config/tokenizer

### 相对位置编码（T5 方案）

- T5 使用 learned relative position bias（相对位置偏置）而非绝对位置嵌入
- 适合探索更长序列的泛化能力
- 本项目提供 `configs/relative_t5_config.yaml` 作为可选实验配置
- 使用 T5 时需要设置 `source_prefix`（例如 `translate German to English: `）

### 相对位置编码（ALiBi 方案）

- ALiBi 通过对注意力分数添加线性偏置实现相对位置建模
- 对长序列的外推更稳定，且无需显式位置嵌入
- 本项目提供 `configs/alibi_config.yaml` 作为 Marian 的实验配置

### FP16 混合精度训练

**为什么用 FP16**：
- 降低显存占用，提高吞吐
- 同等显存下可增大 batch size，梯度更稳定

**为什么不用 BF16**：
- 需要较新的 GPU 架构支持

### Dynamic Batching（group_by_length）

**原理**：将长度相近的序列分到同一 batch，减少 padding 计算浪费。

### 训练超参选择

| 超参 | 值 | 理由 |
|------|---|------|
| lr | 5e-5 | Transformer 常用学习率 |
| weight_decay | 0.01 | 轻度正则，降低过拟合风险 |
| epochs | 5 | 与数据规模和训练预算匹配 |
| batch_size | 128 | 平衡吞吐与稳定性 |

---

## 4. 推理引擎与解码策略

### Beam Search 与 Length Penalty

- **Beam Search**：保留多个候选序列，降低贪心解码的局部最优风险。
- **Length Penalty**：缓解“偏好短输出”的问题，尤其在长句上更重要。

### 解码策略搜索

通过网格搜索（beam size × length penalty）系统性比较不同解码配置，避免手工调参的偏差。

---

## 5. 评估体系设计

### 多指标互补

```
BLEU:        n-gram 精确率 + 简短惩罚
ChrF:        字符级 F-score，适合形态变化
LengthRatio: 预测长度 / 参考长度
Repetition:  bigram 重复比例
BERTScore:   语义相似度（基于预训练表示）
```

### 为什么加入 BERTScore

- BLEU 关注表面重叠，语义层面不足
- BERTScore 更贴近语义一致性，常用于补充传统指标

---

## 6. Attention Coverage 诊断

### 核心思路

对于每个源端 token j，计算其在所有解码步骤中收到的最大注意力权重：

```
coverage_j = max_i attn[i, j]
```

如果 coverage_j < threshold，则该 token 被视为“低覆盖”，可能存在漏译风险。

### 单句诊断与可视化

- `src/diagnose.py` 支持单句诊断：输出 heatmap 图、低覆盖 token 高亮、自动标签
- 适合面试展示“能定位到哪个词没被翻译”

---

## 7. 错误分类体系

### 6 类错误定义

| 错误类型 | 定义 | 典型表现 |
|---------|------|---------|
| Omission | 源端内容在译文中缺失 | 长句中后半部分丢失 |
| Hallucination | 译文包含源端不存在的内容 | 生成与源无关的短句 |
| Grammar | 译文语法错误 | 词序不自然、时态错误 |
| Truncation | 翻译在完整表达前中断 | 句子到一半突然结束 |
| Entity | 命名实体翻译错误 | 人名、机构名被替换或丢失 |
| Register | 语域不匹配 | 议会文本翻译成口语风格 |

`diagnose.py` 中的自动标签是启发式规则，适合快速定位问题，并不替代人工评估。

---

## 8. 工程化实践

### 项目结构原则

- 配置与代码分离（YAML 驱动）
- 模块化设计（data/train/inference/evaluate/diagnose）
- 可测试（tests/ 覆盖核心指标与数据逻辑）
- 可复现（固定随机种子、版本化配置）

### 工程化配套

- GitHub Actions 运行单元测试
- Dockerfile 提供容器化运行环境

---

## 9. 面试高频问题集

### Q1: Transformer 的 self-attention 和 cross-attention 有什么区别？

**Self-attention**：Q、K、V 来自同一序列，建模序列内部依赖。  
**Cross-attention**：Q 来自 decoder，K、V 来自 encoder，用于对齐源/目标。

### Q2: 为什么模型在长句上容易欠翻译？

常见原因：训练长度分布偏短、位置编码外推能力有限、注意力在长序列上被稀释。

### Q3: 如何解释 BLEU 和 BERTScore 的差异？

BLEU 更偏表面重叠；BERTScore 更关注语义一致性。面试中强调“两个指标互补”。

### Q4: 如果要部署到生产环境，你会怎么做？

1. 预训练模型 + 领域微调
2. FastAPI + 推理服务（vLLM/TGI）
3. 自动评估流水线与质量阈值告警
4. 反馈闭环，持续优化

---

## 10. 可扩展方向

- 引入 COMET/BLEURT 等更强语义指标
- 加入 UI Demo（Gradio/Streamlit）
- 提供 Docker Compose 或多语言扩展配置

---

## 附录：项目与简历对应关系

| 简历描述 | 对应模块 | 文件 |
|---------|---------|------|
| MarianMT 端到端 NMT Pipeline | 整体架构 | src/train.py, src/data.py |
| 多指标评估（BLEU/ChrF/BERTScore） | 评估流水线 | src/metrics.py, src/evaluate.py |
| Attention coverage 诊断与可视化 | 诊断模块 | src/diagnose.py |
| Beam Search 与长度惩罚搜索 | 解码搜索 | src/decode_search.py |
| 错误分类标准 | 错误体系 | src/error_taxonomy.py |
