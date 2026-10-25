"""Fine-tune the retrieval embedding model on (question, evidence paragraph) pairs.

  python -m eval.finetune_embedder --pairs 3000 --epochs 1
  python -m eval.run retrieval --dataset qasper --embed-model models/bge-small-lecturemind

Trains on QASPER *train* papers (evaluation uses the *validation* papers, so
there is no leakage). Loss: MultipleNegativesRankingLoss with in-batch negatives
plus one mined same-paper hard negative per question:
the loss takes (anchor, positive, hard negative) features.
Runs on CPU in a few minutes for ~3k triples.
"""
import argparse
import os
import random
import time

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss

from app import config
from eval.datasets import load_parquet

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bge-small-lecturemind")


def build_pairs(n_pairs, seed=0):
    df = load_parquet("qasper", "train")
    rng = random.Random(seed)
    triples = []
    for _, row in df.iterrows():
        paragraphs = [p for paras in row["full_text"]["paragraphs"] for p in paras if p and len(p.split()) > 15]
        qas = row["qas"]
        for question, annotations in zip(qas["question"], qas["answers"]):
            evidence = {e for ann in annotations["answer"] for e in ann["evidence"]
                        if e and not e.startswith("FLOAT SELECTED") and len(e.split()) > 10}
            if not evidence:
                continue
            positive = rng.choice(sorted(evidence))
            negatives = [p for p in paragraphs if p not in evidence]
            if negatives:
                triples.append((question, positive, rng.choice(negatives)))
    rng.shuffle(triples)
    return triples[:n_pairs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=config.EMBEDDING_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--pairs", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-seq-length", type=int, default=256)
    args = p.parse_args()

    triples = build_pairs(args.pairs)
    print(f"{len(triples)} (question, positive, hard negative) triples")

    model = SentenceTransformer(args.base)
    model.max_seq_length = args.max_seq_length
    loss_fn = MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = args.epochs * ((len(triples) + args.batch_size - 1) // args.batch_size)
    warmup = max(1, steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / max(1, steps - warmup)))
    )
    device = model.device
    prefix = config.QUERY_INSTRUCTION if "bge" in args.base.lower() else ""

    def features(texts):
        # Recent transformers tokenizers add string metadata such as
        # ``modality`` to BatchEncoding; the loss only accepts tensor features.
        return {
            k: v.to(device)
            for k, v in model.tokenize(texts).items()
            if torch.is_tensor(v)
        }

    model.train()
    step, t0 = 0, time.perf_counter()
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(triples)
        for s in range(0, len(triples), args.batch_size):
            batch = triples[s:s + args.batch_size]
            q = features([prefix + b[0] for b in batch])
            pos = features([b[1] for b in batch])
            neg = features([b[2] for b in batch])
            # anchor, positive, hard negative: other positives in the batch act as extra negatives
            loss = loss_fn([q, pos, neg], labels=None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 10 == 0 or step == steps:
                print(f"epoch {epoch + 1} step {step}/{steps} loss {loss.item():.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    model.eval()
    os.makedirs(args.out, exist_ok=True)
    model.save(args.out)
    print(f"saved fine-tuned model to {args.out}")
    print(f"use it with:  EMBEDDING_MODEL={args.out}  (then re-upload documents)")


if __name__ == "__main__":
    main()
