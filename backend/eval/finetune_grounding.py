"""Fine-tune the NLI grounding checker on sentence-level RAG adherence labels.

  python -m eval.finetune_grounding --pairs 3000 --epochs 2
  python -m eval.run hallucination --source ragtruth --nli-model models/nli-deberta-lecturemind

Training data: RAGBench *train* (rungalileo/ragbench), where every response
sentence is labelled supported / unsupported against the retrieved documents.
Each example mirrors inference in app.services.grounding_service: the premise
is the app chunk with the most lexical overlap with the sentence, the
hypothesis is the sentence. Supported -> "entailment", unsupported -> "neutral",
so the model drops into the existing pipeline unchanged. Evaluation uses
RAGTruth (a different dataset, never trained on) and RAGBench *test*.

Only the transformer layers are trained (the 128k-token embedding matrix is
frozen) so it fits in about 1.5 GB of RAM on a laptop CPU.
"""
import argparse
import os
import random
import re
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app import config
from app.utils.chunker import chunk_text
from eval.datasets import RAGBENCH_SUBSETS, load_parquet

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "nli-deberta-lecturemind")
_WORD = re.compile(r"[a-z0-9]+")


def _overlap(sentence_tokens, chunk):
    chunk_tokens = set(_WORD.findall(chunk.lower()))
    return len(sentence_tokens & chunk_tokens) / max(1, len(sentence_tokens))


def build_examples(per_subset, seed=0):
    rng = random.Random(seed)
    supported, unsupported = [], []
    for subset in RAGBENCH_SUBSETS:
        url = f"https://huggingface.co/api/datasets/galileo-ai/ragbench/parquet/{subset}/train/0.parquet"
        try:
            df = load_parquet(f"ragbench_{subset}", "train", url=url)
        except Exception as e:  # a missing subset should not abort training
            print(f"  skipping {subset}: {e}")
            continue
        idx = list(range(len(df)))
        rng.shuffle(idx)
        for i in idx[:per_subset]:
            row = df.iloc[i]
            if row["response_sentences"] is None or row["documents"] is None:
                continue
            bad = {k.strip(". ") for k in (row["unsupported_response_sentence_keys"] if row["unsupported_response_sentence_keys"] is not None else [])}
            chunks = [c for c in chunk_text("\n\n".join(d for d in row["documents"] if d)) if c.strip()]
            if not chunks:
                continue
            for key, sentence in (list(s) for s in row["response_sentences"]):
                if len(sentence.split()) < 4:
                    continue
                tokens = set(_WORD.findall(sentence.lower()))
                premise = max(chunks, key=lambda c: _overlap(tokens, c))
                (unsupported if key.strip(". ") in bad else supported).append((premise, sentence))
        print(f"  {subset}: total so far {len(supported)} supported / {len(unsupported)} unsupported")
    return supported, unsupported


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=config.NLI_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--pairs", type=int, default=3000, help="total training pairs (balanced)")
    p.add_argument("--per-subset", type=int, default=300, help="responses sampled per RAGBench subset")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=320)
    args = p.parse_args()
    torch.manual_seed(0)

    supported, unsupported = build_examples(args.per_subset)
    rng = random.Random(0)
    half = args.pairs // 2
    rng.shuffle(supported), rng.shuffle(unsupported)
    n = min(half, len(supported), len(unsupported))
    examples = [(p_, h, 1) for p_, h in supported[:n]] + [(p_, h, 0) for p_, h in unsupported[:n]]
    rng.shuffle(examples)
    print(f"{len(examples)} balanced training pairs ({n} per class)")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base)
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    ent, neu = labels["entailment"], labels["neutral"]
    for name, param in model.named_parameters():
        if "embeddings.word_embeddings" in name:
            param.requires_grad = False
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    print(f"trainable parameters: {sum(p_.numel() for p_ in trainable) / 1e6:.1f}M")

    steps = args.epochs * ((len(examples) + args.batch_size - 1) // args.batch_size)
    warmup = max(1, steps // 10)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / max(1, steps - warmup)))
    )
    model.train()
    step, t0 = 0, time.perf_counter()
    for epoch in range(args.epochs):
        rng.shuffle(examples)
        for s in range(0, len(examples), args.batch_size):
            batch = examples[s:s + args.batch_size]
            enc = tok([b[0] for b in batch], [b[1] for b in batch], truncation="only_first",
                      max_length=args.max_length, padding=True, return_tensors="pt")
            y = torch.tensor([ent if b[2] else neu for b in batch])
            loss = torch.nn.functional.cross_entropy(model(**enc).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 10 == 0 or step == steps:
                print(f"epoch {epoch + 1} step {step}/{steps} loss {loss.item():.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    model.eval()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved fine-tuned grounding model to {args.out}")
    print(f"use it with:  NLI_MODEL={args.out}  or  --nli-model {args.out}")


if __name__ == "__main__":
    main()
