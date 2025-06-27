import json
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
import time

model = "/home/xtang/models/maidalun1020/bce-reranker-base_v1"

class Reranker():
    def __init__(self, device, use_ipex):
        self.device = device
        self.use_ipex = (use_ipex.lower() == "true")
        print(f"Loading reranker model on device={self.device}, use_ipex={self.use_ipex}")

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.model.eval().to(self.device)

        if self.device == "cpu" and self.use_ipex and IPEX_AVAILABLE:
            print("Applying IPEX optimization (bfloat16 + TorchScript)...")
            self.model = ipex.optimize(self.model, dtype=torch.bfloat16)
            vocab_size = self.model.config.vocab_size
            dummy_input = torch.randint(vocab_size, (16, 128), device=self.device)
            dummy_mask = torch.ones_like(dummy_input)
            self.model = torch.jit.trace(self.model, (dummy_input, dummy_mask), check_trace=False, strict=False)
            self.model = torch.jit.freeze(self.model)
        else:
            if self.use_ipex and not IPEX_AVAILABLE:
                print("Warning: IPEX not installed, skipping optimization.")
            elif self.use_ipex and self.device != "cpu":
                print("Warning: IPEX only supported on CPU, skipping optimization.")

        print("Reranker ready.")

    def predict_score(self, text_a, text_b):
        inputs = self.tokenizer(text_a, text_b,
                                truncation=True, padding=True,
                                return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs)["logits"].squeeze(-1)
        return logits.item() if logits.dim() == 0 else logits.cpu().tolist()[0]


def evaluate_on_lcqmc(reranker):
    print("=== LCQMC 验评 ===")
    ds = load_dataset("C-MTEB/LCQMC", split="validation")
    scores, labels = [], []

    start_time = time.time()
    for ex in ds:
        s = reranker.predict_score(ex["sentence1"], ex["sentence2"])
        scores.append(s)
        labels.append(ex["score"])
    end_time = time.time()

    total_time = end_time - start_time
    data_count = len(ds)
    avg_time_per_sample = total_time / data_count if data_count > 0 else 0

    preds = [1 if s > 0.5 else 0 for s in scores]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"LCQMC  Accuracy={acc:.4f}, F1={f1:.4f}")
    print(f'Total time: {total_time:.4f} seconds, Samples: {data_count}, '
          f'Avg time per sample: {avg_time_per_sample:.6f} seconds')


def evaluate_on_ms_marco_doc(reranker, max_queries=None):
    print("=== MS MARCO Document Ranking 验评 (MRR) ===")
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
    mrrs = []
    total_queries = len(ds) if max_queries is None else min(len(ds), max_queries)

    start_time = time.time()
    for i, ex in enumerate(ds):
        if max_queries and i >= max_queries:
            break
        query = ex.get("query")
        raw_passages = ex.get("passages")
        # parse JSON if loaded as string
        if isinstance(raw_passages, str):
            passages = json.loads(raw_passages)
        else:
            passages = raw_passages

        scores = []
        labels = []
        for p in passages:
            # p should be a dict with keys 'passage_text' and 'is_selected'
            text = p.get("passage_text") if isinstance(p, dict) else str(p)
            label = p.get("is_selected", 0) if isinstance(p, dict) else 0
            scores.append(reranker.predict_score(query, text))
            labels.append(label)

        ranked_idxs = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        rr = 0.0
        for rank, idx in enumerate(ranked_idxs, start=1):
            if labels[idx] == 0.8:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
    end_time = time.time()

    total_time = end_time - start_time
    data_count = total_queries
    avg_time_per_sample = total_time / data_count if data_count > 0 else 0

    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
    print(f"MS MARCO Validation MRR@all = {avg_mrr:.4f} over {len(mrrs)} queries")
    print(f'Total time: {total_time:.4f} seconds, Samples: {data_count}, '
          f'Avg time per sample: {avg_time_per_sample:.6f} seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank-Dev", type=str, default="cpu")
    parser.add_argument("--use-ipex", type=str, default="True")
    args, _ = parser.parse_known_args()

    print("Options:", vars(args))
    reranker = Reranker(args.rerank_Dev, args.use_ipex)

    # evaluate_on_lcqmc(reranker)
    evaluate_on_ms_marco_doc(reranker, max_queries=None)
