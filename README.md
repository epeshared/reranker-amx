# Reranker 服务说明

## 概述

本项目基于 Hugging Face `transformers` 和 Intel Extension for PyTorch (IPEX) 实现了一个句子对重排序（Reranker）示例，通过预训练的序列分类模型为输入的句子对生成相似度打分，并在 LCQMC 数据集上进行评测。

## 特性

* 支持 CPU 和 GPU 推理
* 可选 IPEX 加速（仅在 CPU 模式且已安装 `intel_extension_for_pytorch` 时生效）
* 使用 TorchScript Trace + Freeze 实现图优化
* 内置 LCQMC 验证集评测，可输出 Accuracy、F1、总耗时和平均单样本耗时

## 环境依赖

* Python 3.11+
* `torch`, `transformers`, `datasets`, `scikit-learn`
* 可选：`intel_extension_for_pytorch` （用于 CPU 上的 IPEX 优化）

> **提示**：若使用 GPU 推理，无需安装 IPEX。

```bash
pip install torch torchvision torchaudio    # 或者根据 CUDA 版本选择对应的 PyTorch 发行版
pip install transformers datasets scikit-learn
# 可选：CPU 推理时加速
pip install intel_extension_for_pytorch
```

## 代码结构

```text
├── reranker.py       # 主脚本，包含 Reranker 类和评测流程\ n└── README.md        # 本说明文档
```

## 使用说明

1. 修改模型路径

   ```python
   # 在 reranker.py 顶部，将 model 变量替换为你的本地模型路径或 Hugging Face repo
   model = "/path/to/your/reranker-model"
   ```

2. 运行脚本

   ```bash
   python reranker.py --rerank-Dev <device> --use-ipex <True|False>
   ```

   * `--rerank-Dev`: 指定推理设备，示例：`cpu`, `cuda`, `cuda:0`
   * `--use-ipex`: 是否在 CPU 上启用 IPEX 优化，默认 `True`

### 示例

```bash
# CPU 推理并启用 IPEX（需要安装 intel_extension_for_pytorch）
python reranker.py --rerank-Dev cpu --use-ipex True

# GPU 推理（IPEX 会被跳过）
python reranker.py --rerank-Dev cuda:0 --use-ipex False
```

脚本将加载 LCQMC 验证集并输出类似：

```
Loading LCQMC dataset (validation split)...
Evaluating on LCQMC dataset...
Evaluation on LCQMC - Accuracy: 0.9123, F1: 0.9034
Total time: 12.3456 seconds, Samples: 15600, Avg time per sample: 0.000792 seconds
```

## 代码说明

* \`\`\*\* 类\*\*

  * 初始化：加载模型、Tokenizer、可选 IPEX + TorchScript 优化
  * `predict_score(text_a, text_b)`: 对单个句子对打分，返回浮点分数
* \`\`\*\* 函数\*\*

  * 加载 `C-MTEB/LCQMC` 验证集
  * 遍历所有样本，调用 `predict_score`，收集分数和标签
  * 以 0.5 为阈值计算 Accuracy 和 F1，并输出耗时统计

## 常见问题

* **IPEX 未安装**：若 `--use-ipex True` 但环境中未检测到 `intel_extension_for_pytorch`，脚本会给出警告并跳过 IPEX 优化。
* **GPU 模式**：`--rerank-Dev cuda` 时会忽略 IPEX 优化，仅在 CPU 上生效。

## 扩展

* 批量推理：可将 `Reranker.predict_score` 修改为支持批量输入以提升效率。
* 自定义评测：替换 `evaluate_on_lcqmc` 中的数据集为其他任务。

