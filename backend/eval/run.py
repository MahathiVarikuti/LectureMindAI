"""LectureMind evaluation harness. Run from the backend/ directory:

  python -m eval.run retrieval     --dataset qasper            # ablation: v1 vs new retrieval
  python -m eval.run threshold     --dataset qasper            # calibrate RELEVANCE_THRESHOLD
  python -m eval.run generation    --dataset squad_v2 --models qwen2.5:1.5b qwen2.5:7b
  python -m eval.run hallucination                             # grounding checker vs RAGTruth labels
  python -m eval.run conversation                              # query rewriting on CoQA

Results are written to eval/results/ as JSON (+ JSONL of every prediction).
"""
import argparse
import json
import os
import random
import re
import statistics
import time
import uuid

import chromadb

from app import config
from app.services.embedding_service import embed_documents
from app.services.grounding_service import check_grounding
from app.services.llm_service import rewrite_query
from app.services.rag_pipeline import answer_question
from app.services.retrieval_service import Retriever, select_context
from app.utils.chunker import chunk_text, chunk_text_fixed
from eval import metrics
from eval.datasets import LOADERS, load_ragbench, load_ragtruth

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
_client = chromadb.EphemeralClient()

# name -> how the index is built and searched
RETRIEVAL_CONFIGS = {
    "v1_baseline":   dict(chunker="fixed", embed="sentence-transformers/all-MiniLM-L6-v2", hybrid=False, rerank=False),
    "shipped_minilm": dict(chunker="smart", embed="all-MiniLM-L6-v2", hybrid=False, rerank=False),
    "dense_bge":     dict(chunker="smart", embed="BAAI/bge-small-en-v1.5", hybrid=False, rerank=False),
    "hybrid_bge":    dict(chunker="smart", embed="BAAI/bge-small-en-v1.5", hybrid=True, rerank=False),
    "hybrid_rerank": dict(chunker="smart", embed="BAAI/bge-small-en-v1.5", hybrid=True, rerank=True),
    "minilm_hybrid_rerank": dict(chunker="smart", embed="all-MiniLM-L6-v2", hybrid=True, rerank=True, fuse=False),
    "minilm_hybrid_rerank_fused": dict(chunker="smart", embed="all-MiniLM-L6-v2", hybrid=True, rerank=True, fuse=True, weight=1.0),
    "minilm_fused_w2": dict(chunker="smart", embed="all-MiniLM-L6-v2", hybrid=True, rerank=True, fuse=True, weight=2.0),
    "minilm_fused_w3": dict(chunker="smart", embed="all-MiniLM-L6-v2", hybrid=True, rerank=True, fuse=True, weight=3.0),
    "minilm_fused_w5": dict(chunker="smart", embed="all-MiniLM-L6-v2", hybrid=True, rerank=True, fuse=True, weight=5.0),
}
_index_cache = {}


def build_index(docs, chunker, embed_model, tag):
    key = (tag, chunker, embed_model)
    if key in _index_cache:
        return _index_cache[key]
    chunk_fn = chunk_text_fixed if chunker == "fixed" else chunk_text
    ids, texts, metas = [], [], []
    for doc in docs:
        for page in doc["pages"]:
            for chunk in chunk_fn(page["text"]):
                if not chunk.strip():
                    continue
                ids.append(f"{doc['doc_id']}_{len(ids)}")
                texts.append(chunk)
                metas.append({"document_id": doc["doc_id"], "filename": doc["title"][:80], "page": page["page"]})
    t = time.perf_counter()
    embeddings = embed_documents(texts, model_name=embed_model)
    space = "l2" if chunker == "fixed" else "cosine"  # v1 used Chroma's default L2
    col = _client.create_collection(f"eval_{uuid.uuid4().hex[:12]}", metadata={"hnsw:space": space})
    for s in range(0, len(ids), 1000):
        col.add(ids=ids[s:s + 1000], documents=texts[s:s + 1000],
                embeddings=embeddings[s:s + 1000].tolist(), metadatas=metas[s:s + 1000])
    print(f"  indexed {len(docs)} docs -> {len(ids)} chunks ({chunker}, {embed_model}) in {time.perf_counter() - t:.0f}s")
    _index_cache[key] = col
    return col


def make_retriever(docs, cfg, tag):
    col = build_index(docs, cfg["chunker"], cfg["embed"], tag)
    return Retriever(col, embedding_model=cfg["embed"], use_hybrid=cfg["hybrid"], use_reranker=cfg["rerank"],
                     fuse_rerank=cfg.get("fuse"), fuse_weight=cfg.get("weight"))


def relevance_flags(hits, q):
    return [int(h["metadata"].get("document_id") == q["doc_id"] and metrics.is_relevant(h["text"], q["evidence"]))
            for h in hits]


def save(name, payload, rows=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{name}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if rows:
        with open(path.replace(".json", ".jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
    print(f"\nsaved {path}")


def table(rows, cols):
    widths = [max(len(c), *(len(f"{r.get(c, ''):.3f}" if isinstance(r.get(c), float) else str(r.get(c, ''))) for r in rows)) for c in cols]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    print(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(fmt(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))


def load(args):
    kwargs = {"n_docs": args.docs} if args.docs else {}
    docs, questions = LOADERS[args.dataset](**kwargs)
    return docs, questions


# --- 1. retrieval ablation ---------------------------------------------------
def cmd_retrieval(args):
    docs, questions = load(args)
    qs = [q for q in questions if q["scope"] == "in_scope" and q["answerable"] and q["evidence"]]
    random.Random(0).shuffle(qs)
    qs = qs[:args.n]
    print(f"{args.dataset}: {len(docs)} docs, {len(qs)} answerable questions")
    results = []
    for name in args.configs:
        cfg = RETRIEVAL_CONFIGS[name]
        retriever = make_retriever(docs, cfg, args.dataset)
        rels, t = [], time.perf_counter()
        for q in qs:
            rels.append(relevance_flags(retriever.retrieve(q["question"], top_k=10), q))
        row = {"config": name, "latency_ms": (time.perf_counter() - t) / max(len(qs), 1) * 1000}
        for k in (1, 3, 5, 10):
            row.update(metrics.retrieval_metrics(rels, k))
        results.append(row)
        print(f"  {name}: hit@5={row['hit@5']:.3f} mrr@10={row['mrr@10']:.3f}")
    print()
    table(results, ["config", "hit@1", "hit@3", "hit@5", "mrr@10", "ndcg@5", "latency_ms"])
    save(f"retrieval_{args.dataset}", {"dataset": args.dataset, "n_questions": len(qs), "results": results})


# --- 2. threshold calibration ------------------------------------------------
def cmd_threshold(args):
    docs, questions = load(args)
    rng = random.Random(0)
    pos = [q for q in questions if q["scope"] == "in_scope" and q["answerable"]]
    neg = [q for q in questions if q["scope"] == "out_of_scope"]
    # "Hard" negatives: the document IS indexed but does not answer the question
    # (SQuAD v2 / QASPER unanswerable). Retrieval finds on-topic chunks, so the
    # relevance score is much less separable than for out-of-corpus questions.
    hard = [q for q in questions if q["scope"] == "in_scope" and not q["answerable"]]
    rng.shuffle(pos), rng.shuffle(neg), rng.shuffle(hard)
    pos, neg, hard = pos[:args.n], neg[:args.n], hard[:args.n]
    print(f"{args.dataset}: {len(pos)} answerable vs {len(neg)} out-of-corpus vs {len(hard)} in-corpus unanswerable")
    report = {"dataset": args.dataset, "configs": {}}
    for name in args.configs:
        retriever = make_retriever(docs, RETRIEVAL_CONFIGS[name], args.dataset)
        top = lambda q: (retriever.retrieve(q["question"], top_k=1) or [{"score": 0.0}])[0]["score"]
        ps, ns, hs = [top(q) for q in pos], [top(q) for q in neg], [top(q) for q in hard]
        entry = {}
        for label, negatives in (("out_of_corpus", ns), ("in_corpus_unanswerable", hs), ("combined", ns + hs)):
            if not negatives:
                continue
            rows, best = metrics.sweep_threshold(ps, negatives)
            auc = metrics.auroc(ps, negatives)
            entry[label] = {"auroc": auc, "best": best, "sweep": rows,
                            "negative_median": statistics.median(negatives)}
            print(f"\n[{name} | negatives={label}] AUROC = {auc:.3f} "
                  f"(median answerable {statistics.median(ps):.3f}, negatives {statistics.median(negatives):.3f})")
            table([r for r in rows if r["threshold"] in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7) or r is best],
                  ["threshold", "answer_rate_in_scope", "abstain_rate_out_of_scope", "balanced_accuracy"])
            print(f"best threshold = {best['threshold']} (balanced accuracy {best['balanced_accuracy']:.3f})")
        report["configs"][name] = {"answerable_median": statistics.median(ps), **entry}
    print("\nNote: abstaining on out-of-scope questions only matters if it does not cost too many in-scope answers;"
          " prefer a threshold that keeps answer_rate_in_scope >= ~0.95.")
    save(f"threshold_{args.dataset}", report)


# --- 3. end-to-end generation / model comparison ----------------------------
def _strip_citations(text):
    return re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", text)


def cmd_generation(args):
    docs, questions = load(args)
    rng = random.Random(1)
    groups = {
        "answerable": [q for q in questions if q["scope"] == "in_scope" and q["answerable"]],
        "unanswerable": [q for q in questions if q["scope"] == "in_scope" and not q["answerable"]],
        "out_of_scope": [q for q in questions if q["scope"] == "out_of_scope"],
    }
    n_each = {"answerable": args.n // 2, "unanswerable": args.n // 4, "out_of_scope": args.n // 4}
    sample = []
    for g, qs in groups.items():
        rng.shuffle(qs)
        sample += [{**q, "group": g} for q in qs[:n_each[g]]]
    retriever = make_retriever(docs, RETRIEVAL_CONFIGS["hybrid_rerank"], args.dataset)
    threshold = args.threshold if args.threshold is not None else retriever.relevance_threshold
    print(f"{args.dataset}: {len(sample)} questions "
          f"({', '.join(f'{g}={sum(s['group'] == g for s in sample)}' for g in groups)}), threshold={threshold}")

    summary, all_rows = [], []
    for model in args.models:
        rows = []
        for i, q in enumerate(sample):
            r = answer_question(q["question"], retriever, model=model, rewrite=False,
                                relevance_threshold=threshold, grounding=not args.no_grounding)
            ans = _strip_citations(r["answer"])
            row = {"model": model, "qid": q["qid"], "group": q["group"], "question": q["question"],
                   "gold": q["answers"], "answer": r["answer"], "abstained": r["abstained"],
                   "abstain_reason": r["abstain_reason"], "top_score": r["top_score"],
                   "groundedness": (r["grounding"] or {}).get("groundedness"), "flagged": r["flagged"],
                   "latency_s": r["timings"]["total"]}
            if q["group"] == "answerable":
                row.update(em=metrics.exact_match(ans, q["answers"]), f1=metrics.token_f1(ans, q["answers"]),
                           answer_recall=metrics.answer_recall(ans, q["answers"]))
            rows.append(row)
            print(f"\r  {model}: {i + 1}/{len(sample)}", end="", flush=True)
        print()
        ans_rows = [r for r in rows if r["group"] == "answerable"]
        answered = [r for r in ans_rows if not r["abstained"]]
        grounded = [r["groundedness"] for r in rows if r["groundedness"] is not None]
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        summary.append({
            "model": model,
            "EM": mean([r["em"] for r in ans_rows]),
            "F1": mean([r["f1"] for r in ans_rows]),
            "ans_recall": mean([r["answer_recall"] for r in ans_rows]),
            "false_refusal": mean([float(r["abstained"]) for r in ans_rows]),
            "refusal_unans": mean([float(r["abstained"]) for r in rows if r["group"] == "unanswerable"]),
            "refusal_oos": mean([float(r["abstained"]) for r in rows if r["group"] == "out_of_scope"]),
            "groundedness": mean(grounded),
            "halluc_rate": mean([float(r["flagged"]) for r in rows if not r["abstained"]]),
            "F1_if_answered": mean([r["f1"] for r in answered]),
            "p50_s": metrics.percentile([r["latency_s"] for r in rows], 50),
            "p95_s": metrics.percentile([r["latency_s"] for r in rows], 95),
        })
        all_rows += rows
    print()
    table(summary, ["model", "EM", "F1", "ans_recall", "false_refusal", "refusal_unans", "refusal_oos",
                    "groundedness", "halluc_rate", "p50_s", "p95_s"])
    save(f"generation_{args.dataset}", {"dataset": args.dataset, "threshold": threshold,
                                        "n": len(sample), "summary": summary}, all_rows)


# --- 4. hallucination detector vs human labels (RAGTruth) -------------------
def cmd_hallucination(args):
    rows = load_ragbench(n=args.n) if args.source == "ragbench" else load_ragtruth(n=args.n)
    print(f"{args.source}: {len(rows)} responses, {sum(r['hallucinated'] for r in rows)} labelled hallucinated")
    scores, labels = [], []
    for i, r in enumerate(rows):
        contexts = chunk_text(r["context"])
        report = check_grounding(r["output"], contexts)
        r["groundedness"] = report["groundedness"]
        r["min_support"] = min((s["support"] for s in report["sentences"]), default=1.0)
        scores.append(r)
        labels.append(r["hallucinated"])
        print(f"\r  {i + 1}/{len(rows)}", end="", flush=True)
    print()
    hall = [r for r in scores if r["hallucinated"]]
    clean = [r for r in scores if not r["hallucinated"]]
    out = {"n": len(rows), "positives": len(hall)}
    for key in ("groundedness", "min_support"):
        auc = metrics.auroc([1 - r[key] for r in hall], [1 - r[key] for r in clean])
        best = None
        sweep = []
        for t in [x / 20 for x in range(1, 20)]:
            m = metrics.binary_classification(labels, [r[key] < t for r in scores])
            sweep.append({"threshold": t, **m})
            if best is None or m["f1"] > best["f1"]:
                best = {"threshold": t, **m}
        out[key] = {"auroc": auc, "best": best, "sweep": sweep}
        print(f"\n[{key}] AUROC = {auc:.3f}; best F1 {best['f1']:.3f} at threshold {best['threshold']}"
              f" (precision {best['precision']:.3f}, recall {best['recall']:.3f})")
    current = metrics.binary_classification(labels, [r["groundedness"] < config.GROUNDEDNESS_THRESHOLD for r in scores])
    out["current_config"] = {"GROUNDEDNESS_THRESHOLD": config.GROUNDEDNESS_THRESHOLD, **current}
    print(f"\ncurrent GROUNDEDNESS_THRESHOLD={config.GROUNDEDNESS_THRESHOLD}: "
          f"precision {current['precision']:.3f}, recall {current['recall']:.3f}, F1 {current['f1']:.3f}")
    save(f"hallucination_{args.source}", out, scores)


# --- 5. conversational query rewriting (CoQA) --------------------------------
def cmd_conversation(args):
    docs, questions = LOADERS["coqa"](n_docs=args.docs or 60)
    qs = [q for q in questions if q["turn"] > 0 and q["evidence"]]
    random.Random(0).shuffle(qs)
    qs = qs[:args.n]
    retriever = make_retriever(docs, RETRIEVAL_CONFIGS["hybrid_rerank"], "coqa")
    print(f"CoQA: {len(docs)} stories, {len(qs)} follow-up questions")
    variants = {"raw question": [], "prev question + question": [], "LLM rewrite": []}
    rows = []
    for i, q in enumerate(qs):
        prev_q = q["history"][-2]["content"]
        rewritten = rewrite_query(q["history"], q["question"], model=args.models[0])
        for name, text in (("raw question", q["question"]),
                           ("prev question + question", f"{prev_q} {q['question']}"),
                           ("LLM rewrite", rewritten)):
            variants[name].append(relevance_flags(retriever.retrieve(text, top_k=10), q))
        rows.append({"qid": q["qid"], "question": q["question"], "rewritten": rewritten})
        print(f"\r  {i + 1}/{len(qs)}", end="", flush=True)
    print()
    results = []
    for name, rels in variants.items():
        results.append({"query": name, **metrics.retrieval_metrics(rels, 1), **metrics.retrieval_metrics(rels, 5)})
    table(results, ["query", "hit@1", "hit@5", "mrr@5"])
    save("conversation_coqa", {"n": len(qs), "rewrite_model": args.models[0], "results": results}, rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["retrieval", "threshold", "generation", "hallucination", "conversation"])
    p.add_argument("--dataset", default="qasper", choices=list(LOADERS))
    p.add_argument("--docs", type=int, help="number of documents to index")
    p.add_argument("-n", type=int, default=200, help="number of questions / samples")
    p.add_argument("--source", default="ragtruth", choices=["ragtruth", "ragbench"],
                   help="labelled hallucination data for the `hallucination` command")
    p.add_argument("--nli-model", help="override NLI_MODEL for the grounding check (e.g. cross-encoder/nli-deberta-v3-base)")
    p.add_argument("--configs", nargs="+", default=list(RETRIEVAL_CONFIGS), choices=list(RETRIEVAL_CONFIGS))
    p.add_argument("--models", nargs="+", default=[config.LLM_MODEL])
    p.add_argument("--threshold", type=float, help="override RELEVANCE_THRESHOLD")
    p.add_argument("--embed-model", help="also evaluate this (e.g. fine-tuned) embedding model")
    p.add_argument("--reranker-model", help="override RERANKER_MODEL (e.g. models/reranker-lecturemind)")
    p.add_argument("--candidate-k", type=int, help="override CANDIDATE_K, the pool the reranker sees")
    p.add_argument("--no-grounding", action="store_true", help="skip the NLI grounding check (faster)")
    args = p.parse_args()
    if args.nli_model:
        config.NLI_MODEL = args.nli_model
    if args.reranker_model:
        config.RERANKER_MODEL = args.reranker_model
    if args.candidate_k:
        config.CANDIDATE_K = args.candidate_k
    if args.embed_model:
        RETRIEVAL_CONFIGS["custom_dense"] = dict(chunker="smart", embed=args.embed_model, hybrid=False, rerank=False)
        RETRIEVAL_CONFIGS["custom_hybrid_rerank"] = dict(chunker="smart", embed=args.embed_model, hybrid=True, rerank=True)
        args.configs += ["custom_dense", "custom_hybrid_rerank"]
    {"retrieval": cmd_retrieval, "threshold": cmd_threshold, "generation": cmd_generation,
     "hallucination": cmd_hallucination, "conversation": cmd_conversation}[args.command](args)


if __name__ == "__main__":
    main()
