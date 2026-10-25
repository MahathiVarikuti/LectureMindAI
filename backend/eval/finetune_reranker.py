"""Fine-tune the cross-encoder reranker on QASPER train (question, paragraph) pairs.

  python -m eval.finetune_reranker --pairs 3000 --epochs 2
  RERANKER_MODEL=models/reranker-lecturemind python -m eval.run retrieval --dataset qasper

Positives are gold evidence paragraphs; negatives are other paragraphs of the
same paper (the hard case the reranker sees at query time). Trained on the
*train* split; evaluation uses *validation* papers, so there is no leakage.
The reranker's score scale changes after fine-tuning, so re-run
`eval.run threshold` before trusting RERANKER_RELEVANCE_THRESHOLD.
"""
import argparse
import os
import random
import time

import torch
from sentence_transformers import CrossEncoder

from app import config
from eval.datasets import load_parquet

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "reranker-lecturemind")


def build_pairs(n_pairs, negatives_per_positive=3, seed=0):
    df = load_parquet("qasper", "train")
    rng = random.Random(seed)
    pairs = []
    for _, row in df.iterrows():
        paragraphs = [p for paras in row["full_text"]["paragraphs"] for p in paras if p and len(p.split()) > 15]
        qas = row["qas"]
        for question, annotations in zip(qas["question"], qas["answers"]):
            evidence = {e for ann in annotations["answer"] for e in ann["evidence"]
                        if e and not e.startswith("FLOAT SELECTED") and len(e.split()) > 10}
            negatives = [p for p in paragraphs if p not in evidence]
            if not evidence or len(negatives) < negatives_per_positive:
                continue
            pairs.append((question, rng.choice(sorted(evidence)), 1.0))
            pairs += [(question, n, 0.0) for n in rng.sample(negatives, negatives_per_positive)]
    # keep each positive with its negatives together, then truncate
    groups = [pairs[i:i + negatives_per_positive + 1] for i in range(0, len(pairs), negatives_per_positive + 1)]
    rng.shuffle(groups)
    flat = [x for g in groups for x in g]
    return flat[:n_pairs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=config.RERANKER_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--pairs", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    args = p.parse_args()
    torch.manual_seed(0)

    pairs = build_pairs(args.pairs)
    print(f"{len(pairs)} pairs ({sum(int(x[2]) for x in pairs)} positive)")
    model = CrossEncoder(args.base, num_labels=1, max_length=args.max_length)
    tok, net = model.tokenizer, model.model
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(pairs) + args.batch_size - 1) // args.batch_size)
    warmup = max(1, steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / max(1, steps - warmup)))
    )
    rng = random.Random(0)
    net.train()
    step, t0 = 0, time.perf_counter()
    for epoch in range(args.epochs):
        rng.shuffle(pairs)
        for s in range(0, len(pairs), args.batch_size):
            batch = pairs[s:s + args.batch_size]
            enc = tok([b[0] for b in batch], [b[1] for b in batch], truncation="only_second",
                      max_length=args.max_length, padding=True, return_tensors="pt")
            logits = net(**enc).logits.view(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.tensor([b[2] for b in batch]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 20 == 0 or step == steps:
                print(f"epoch {epoch + 1} step {step}/{steps} loss {loss.item():.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    net.eval()
    os.makedirs(args.out, exist_ok=True)
    model.save(args.out)
    print(f"saved fine-tuned reranker to {args.out}")


if __name__ == "__main__":
    main()
