import torch
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    ipex = None
    IPEX_AVAILABLE = False

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
from pprint import pprint

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# model = "/mnt/nvme0/home/xtang/models/BAAI/bge-reranker-large"
# model = "/home/xtang/models/BAAI/bge-reranker-v2-m3"
model = "/home/xtang/models/maidalun1020/bce-reranker-base_v1"

class Reranker():
    def __init__(self, device, use_ipex):
        self.device = device
        # 注意这里传入的 use_ipex 是字符串，比如 "True"/"False"，可以先转成 bool
        self.use_ipex = (use_ipex.lower() == "true")
        print(f"Loading BGE-Reranker-Large model... on device={self.device} with use_ipex={self.use_ipex}")
        
        # 加载模型和 tokenizer（可替换为你的本地路径或 huggingface repo）
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.model.eval().to(self.device)

        # 在 CPU 且用户指定 use_ipex=True 且环境中可用 IPEX 才执行优化
        if self.device == "cpu" and self.use_ipex and IPEX_AVAILABLE:
            print("Using IPEX for CPU optimization...")
            # 这里以 dtype=torch.int8 为例，但实际可能需要先做量化或校准流程
            self.model = ipex.optimize(self.model, dtype=torch.bfloat16)

            vocab_size = self.model.config.vocab_size
            batch_size = 16
            seq_length = 128
            d = torch.randint(vocab_size, size=[batch_size, seq_length]).to(self.device)
            m = torch.ones_like(d).to(self.device)

            # 使用 torch.jit.trace 进行图优化
            self.model = torch.jit.trace(self.model, (d, m), check_trace=False, strict=False)
            self.model = torch.jit.freeze(self.model)
        else:
            if not IPEX_AVAILABLE and self.use_ipex:
                print("Warning: use_ipex=True but intel_extension_for_pytorch not installed. Skipping IPEX optimization.")
            elif self.device != "cpu" and self.use_ipex:
                print("Warning: use_ipex=True but device is not CPU. Skipping IPEX optimization.")

        print("Reranker created.")

    def predict_score(self, text_a, text_b):
        """
        对 (text_a, text_b) 做一次前向计算，返回一个打分。
        你也可以改成一次性处理批量数据，提升推理效率。
        """
        inputs = self.tokenizer(text_a, text_b, return_tensors="pt",
                                truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)  # outputs是一个dict
            score_tensor = outputs["logits"]  # 取scores
            # 如果是形如 [batch_size, 1] 的张量，则 squeeze(-1)
            score = score_tensor.squeeze(-1).item()
        return score


import time

def evaluate_on_lcqmc(reranker):
    """
    使用 LCQMC 数据集进行测试。LCQMC 的标签是二分类(0/1)，表示句子对是否相似。
    我们以简单的阈值=0.5 来判断大于0.5 即为相似，反之不相似。
    """
    print("Loading LCQMC dataset (validation split)...")
    dataset = load_dataset("C-MTEB/LCQMC", split="validation")

    all_scores = []
    all_labels = []

    print("Evaluating on LCQMC dataset...")

    start_time = time.time()  # 记录起始时间
    for data in dataset:
        sentence1 = data["sentence1"]
        sentence2 = data["sentence2"]
        label = data["score"]  # 0 or 1

        score = reranker.predict_score(sentence1, sentence2)
        all_scores.append(score)
        all_labels.append(label)

    end_time = time.time()  # 记录结束时间

    # 计算总用时
    total_time = end_time - start_time
    # 计算处理的样本数
    data_count = len(dataset)
    # 计算平均每条处理时间
    avg_time_per_sample = total_time / data_count if data_count > 0 else 0

    # 简单阈值判断：score > 0.5 即预测为 1，否则为 0
    threshold = 0.5
    preds = [1 if s > threshold else 0 for s in all_scores]

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)

    print(f"Evaluation on LCQMC - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f'Total time: {total_time:.4f} seconds, Samples: {data_count}, '
          f'Avg time per sample: {avg_time_per_sample:.6f} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reranker Service")
    parser.add_argument("--rerank-Dev", type=str, default="cpu",
                        help="Specify the device for reranking, e.g., 'cpu', 'cuda', 'cuda:0'")
    parser.add_argument("--use-ipex", type=str, default="True",
                        help="Use IPEX to optimize model when using CPU")

    args, extra = parser.parse_known_args()

    # 打印已解析的参数和未知参数
    print("Known arguments:", args)
    print("extra arguments:", extra)

    # 打印传入的参数
    print("Starting Reranker Service with the following options:")
    pprint(vars(args))

    device = args.rerank_Dev
    use_ipex = args.use_ipex

    reranker = Reranker(device, use_ipex)

    # 在此处进行 LCQMC 数据集上评测
    evaluate_on_lcqmc(reranker)
