# Evaluation plan: datasets, hallucination control, thresholds, model size

Everything here is run from `backend/` with `python -m eval.run <command>` (see `eval/run.py`).
Numbers below come from the committed runs in `backend/eval/results/` (2026-10-19).

## 1. Datasets (what each one is for)

| Dataset | Used for | Why | Status |
|---|---|---|---|
| **QASPER** (allenai/qasper) | retrieval, fine-tuning | Long structured papers, closest to lecture notes; has unanswerable questions | in use |
| **SQuAD v2** | abstention threshold, generation | 1/3 of questions unanswerable | in use |
| **CoQA** | conversational query rewriting | follow-ups with pronouns | in use |
| **RAGTruth** | hallucination detector | human hallucination labels on RAG answers | in use |
| **RAGBench** (rungalileo/ragbench) | hallucination detector | 12 domains incl. techqa, emanual, pubmedqa; adherence labels at response and sentence level; larger and more varied than RAGTruth | loader added, **untested** (`--source ragbench`) |
| MuSiQue / HotpotQA | multi-hop retrieval | questions needing several chunks; MuSiQue includes unanswerable variants | suggested |
| MTRAG | multi-turn RAG | human-written multi-turn RAG conversations, replaces/extends CoQA | suggested |
| ChatRAG Bench / ConfRAG | abstention | unanswerable questions in RAG form | suggested |

Source catalogue: https://github.com/jinbo0906/Awesome-RAG-Datasets. It has no educational dataset;
QASPER (scientific papers) is the nearest proxy for lecture material. For the final report, also
hand-label 30-50 questions on your own lecture PDFs: that is the only in-domain number.

## 2. What the existing results say

**Retrieval (QASPER, 112 questions)**

| config | hit@1 | hit@5 | hit@10 | MRR@10 |
|---|---|---|---|---|
| v1 baseline (MiniLM, fixed chunks) | 0.06 | 0.20 | 0.33 | 0.12 |
| dense BGE | 0.09 | 0.26 | 0.35 | 0.16 |
| hybrid BGE | 0.09 | 0.28 | 0.36 | 0.17 |
| hybrid + rerank | 0.14 | 0.37 | 0.46 | 0.24 |

Retrieval is the weakest part: even the best config finds a relevant chunk in the top 5 only 37% of the time.
The reranker doubles hit@1 but costs about 1.1 s per query. Likely causes and fixes, in order of expected payoff:
1. **Bigger candidate pool for the reranker** (`CANDIDATE_K` 20 -> 50): hit@10 (0.46) is far above hit@5, so the right chunk is often reachable but ranked low.
2. **Stronger reranker** (`BAAI/bge-reranker-base`, `RERANKER_MODEL` env var) and **bge-base** embeddings.
3. **Contextual chunk headers**: prepend document title and section name to each chunk before embedding.
4. **Fine-tuned embedder** (`eval/finetune_embedder.py`) on QASPER train, then `--embed-model`.
5. Check the metric before trusting it: relevance is judged by 4-gram overlap with the gold evidence paragraph (`metrics.is_relevant`), which under-counts chunks that answer the question but cite different text.

**Abstention threshold**: the reranker threshold (0.125) separates *out-of-corpus* questions almost perfectly
(AUROC 0.994) but that is the easy case. The hard case is a question about an *indexed* document that the
document does not answer. The generation run shows it: qwen2.5:1.5b refuses only 20% of those. `eval.run threshold`
now reports AUROC and best threshold separately for out-of-corpus, in-corpus unanswerable, and combined negatives.
Choose the threshold from the combined sweep, not the out-of-corpus one.

**Hallucination detector**: the NLI grounding check is weak. Against RAGTruth, AUROC is 0.65 (groundedness) and
0.74 (min per-sentence support); at the current `GROUNDEDNESS_THRESHOLD=0.6`, precision is 0.21 and recall 0.92
(F1 0.34), so it flags almost everything. In the SQuAD run 58% of answers were flagged. Options:
- try `cross-encoder/nli-deberta-v3-base` (`--nli-model`) or a purpose-built detector (Vectara HHEM, MiniCheck) and compare AUROC on RAGTruth **and** RAGBench;
- use `min_support` instead of the mean;
- treat the flag as a UI warning, not an abstention, until precision is above about 0.5;
- reduce hallucination at the source: prompt for quotes, `temperature=0`, and keep the abstention threshold.

**Small vs large LLM (SQuAD v2, n=120)**

| | qwen2.5:1.5b | qwen2.5:3b |
|---|---|---|
| answer recall | 0.90 | 0.82 |
| false refusal | 5% | 13% |
| refuses in-corpus unanswerable | 20% | 60% |
| p95 latency | 17 s | 27 s |

Exact match is about 0 for both because answers are full sentences; report answer recall and F1 (or an LLM judge), not EM.
Larger models refuse unanswerable questions more reliably but refuse more answerable ones and are slower. Run
`qwen2.5:7b` or `llama3.1:8b` for the comparison the mentor asked about. With n=120 the differences are within noise: use n >= 500
and report confidence intervals.

## 3. Commands

```
python -m eval.run retrieval     --dataset qasper -n 200
python -m eval.run threshold     --dataset squad_v2 -n 150      # now includes in-corpus unanswerable
python -m eval.run threshold     --dataset qasper   -n 150
python -m eval.run generation    --dataset squad_v2 -n 500 --models qwen2.5:1.5b qwen2.5:3b qwen2.5:7b
python -m eval.run hallucination --source ragtruth
python -m eval.run hallucination --source ragbench --nli-model cross-encoder/nli-deberta-v3-base
python -m eval.run conversation
```

## 4. Defaults vs. the evaluation settings
The evaluation harness measured the reranked, hybrid, BGE configuration. The application defaults follow the project spec instead
(`all-MiniLM-L6-v2`, dense retrieval only) because it is the lightest on RAM. To reproduce the measured quality in the app, set
`EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 USE_HYBRID=1 USE_RERANKER=1` (and optionally `USE_GROUNDING_CHECK=1`); the relevance threshold then
defaults to 0.125 (reranker scale). With the default MiniLM dense mode the threshold is 0.25 cosine: calibrate it with `python -m eval.run threshold`.
Re-upload documents after changing the embedding model.

## 5. Running safely on a laptop

The Windows freezes were memory exhaustion: up to three models (embedder, reranker, NLI) plus Ollama plus torch.
- Uploads are streamed to disk in 1 MB blocks and capped (`MAX_PDF_SIZE_MB`, `MAX_MEDIA_SIZE_MB`); only one ingestion runs at a time.
- Run one process stack at a time (not Docker and local uvicorn together).
- For a first run set `USE_RERANKER=0 USE_GROUNDING_CHECK=0`, then enable each and watch RAM.
- Eval runs: start with `-n 50`; close other apps; run one model per command.
