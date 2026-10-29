# Evaluation plan: datasets, hallucination control, thresholds, model size

Everything here is run from `backend/` with `python -m eval.run <command>` (see `eval/run.py`).
Numbers below come from the committed runs in `backend/eval/results/` (2026-10-19)
and the additional CPU runs on 2026-10-23/24.

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

**Re-run (QASPER validation, n=100, 40 indexed papers).** These numbers use the
current harness and its 4-gram evidence metric:

| config | hit@1 | hit@5 | hit@10 | MRR@10 |
|---|---:|---:|---:|---:|
| MiniLM fixed baseline | 0.060 | 0.200 | 0.300 | 0.123 |
| BGE dense | 0.130 | 0.250 | 0.350 | 0.178 |
| BGE hybrid | 0.080 | 0.260 | 0.360 | 0.164 |
| BGE hybrid + reranker | 0.130 | 0.340 | 0.430 | 0.217 |
| fine-tuned BGE dense | 0.150 | 0.280 | 0.390 | 0.210 |
| fine-tuned BGE hybrid + reranker | 0.130 | 0.340 | 0.430 | 0.218 |

The fine-tune used 1,000 QASPER pairs and one CPU epoch. It improved dense
retrieval but did not improve the reranked hit@5 over the unfine-tuned model.
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

## 4. Threshold calibration and defaults

The shipped MiniLM dense configuration was calibrated with the combined
out-of-corpus and in-corpus-unanswerable negative sweep on SQuAD v2 and QASPER
(100 answerable questions per run). A threshold of **0.30** was selected because
it keeps the measured answer rate at approximately 0.99 on both datasets; higher
thresholds reached only about 0.51-0.81 answer rate. This satisfies the
answerable-rate floor but leaves hard in-corpus negatives difficult to separate
(combined-negative AUROC 0.793 on SQuAD v2 and 0.614 on QASPER).

The shipped default is `all-MiniLM-L6-v2`, dense-only, with a **0.15** cosine threshold
(`DENSE_RELEVANCE_THRESHOLD`). 0.30 is the benchmark-optimal value, but real lecture and resume
PDFs score the right chunk only 0.15-0.3, so 0.15 was chosen to avoid false refusals; in dense mode
the threshold gates only the best chunk (`per_chunk_threshold=False`). Raise it via
`RELEVANCE_THRESHOLD=0.30` for stricter abstention. The reranker threshold remains 0.125 and is a
separate score scale; re-calibrate it if you switch to the fine-tuned reranker.

## 5. Defaults vs. the evaluation settings
The evaluation harness measured the reranked, hybrid, BGE configuration. The application defaults follow the project spec instead
(`all-MiniLM-L6-v2`, dense retrieval only) because it is the lightest on RAM. To reproduce the measured quality in the app, set
`EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 USE_HYBRID=1 USE_RERANKER=1` (and optionally `USE_GROUNDING_CHECK=1`); the relevance threshold then
defaults to 0.125 (reranker scale). With the default MiniLM dense mode the threshold is 0.15 cosine: calibrate it with `python -m eval.run threshold`.
Re-upload documents after changing the embedding model.

## 6. Multi-hop and multi-turn coverage

`eval.datasets.load_hotpotqa` now loads HotpotQA distractor validation rows into
the common document/question format, including supporting-fact evidence and
unanswerable `noanswer` rows. A few real validation rows were verified locally.
`load_coqa` remains the multi-turn loader and was already exercised by the
conversation command; it provides turn history in the same format.

## 7. Running safely on a laptop

The Windows freezes were memory exhaustion: up to three models (embedder, reranker, NLI) plus Ollama plus torch.
- Uploads are streamed to disk in 1 MB blocks and capped (`MAX_PDF_SIZE_MB`, `MAX_MEDIA_SIZE_MB`); only one ingestion runs at a time.
- Run one process stack at a time (not Docker and local uvicorn together).
- For a first run set `USE_RERANKER=0 USE_GROUNDING_CHECK=0`, then enable each and watch RAM.
- Eval runs: start with `-n 50`; close other apps; run one model per command.

## 8. Fine-tuning (what was trained, on what, and the results)

All training runs on a 12-core laptop CPU with ~2.6 GB free RAM; trained weights go to the git-ignored
`backend/models/`. Held-out data is never used for training.

| Model | Script | Training data | Status |
|---|---|---|---|
| NLI grounding checker (`cross-encoder/nli-deberta-v3-small`) | `eval/finetune_grounding.py` | RAGBench **train**, 7 domains, 2,000 balanced sentence pairs (supported / unsupported), 2 epochs, word embeddings frozen (43.5M trainable params), about 45 min | trained, evaluated |
| Embedder (`bge-small` / MiniLM) | `eval/finetune_embedder.py` | QASPER train questions + hard same-paper negatives | script upgraded to hard negatives; earlier 1k-pair run in section 2 |
| Reranker (`ms-marco-MiniLM-L-6-v2`) | `eval/finetune_reranker.py` | QASPER train, 1 positive + 3 same-paper negatives per question | script written, **not yet run** |

**Grounding checker, evaluated on RAGTruth (300 QA responses, 49 hallucinated; never trained on):**

| | AUROC groundedness | AUROC min-support | best F1 (threshold) |
|---|---|---|---|
| original NLI model | 0.651 | 0.741 | 0.36 (groundedness 0.35) |
| fine-tuned | 0.742 | 0.785 | 0.52 (min-support 0.45) |

**Same comparison on RAGBench test (n=300, 91 hallucinated; same source as the training data):**

| | AUROC groundedness | AUROC min-support | best F1 | F1 at current threshold 0.6 |
|---|---|---|---|---|
| original NLI model | 0.607 | 0.652 | 0.51 | 0.48 |
| fine-tuned | 0.603 | 0.665 | 0.52 | 0.24 |

**Verdict: mixed.** The fine-tuned checker helps on RAGTruth (AUROC +0.09 groundedness, +0.04 min-support, best F1 0.36 to 0.52)
but gives **no gain on RAGBench test**, even though it trained on RAGBench train, and at the current threshold 0.6 it is
worse on both sets (it needs a much higher or different threshold). Best-F1 thresholds were chosen on the evaluated sample itself,
and each run is a single n=300 sample, so differences of a few points are noise. It is therefore **not made the default**.
Likely causes: only 2,000 pairs for 2 epochs, premises picked by lexical overlap, and response-level labels
that are noisy. Next steps: more data and epochs on a GPU, pick the threshold on one dataset and test on another, and try
MiniCheck or HHEM. Keep the flag as a UI warning, not a hard refusal. To try it: `NLI_MODEL=backend/models/nli-deberta-lecturemind`.

Datasets reviewed and rejected: the Civitai multimodal-RAG set (AI-generated image prompts, no questions or labels).
Lecture-specific candidates not yet used: LPM (slides + speech), LecSlides-370K, MMLongBench-Doc, SPIQA, HaluBench, LLM-AggreFact.

## 9. Retrieval fine-tuning results and the recommended profile

QASPER validation, 100 questions, 40 indexed papers, hybrid (dense + BM25) retrieval, 20 candidates unless noted.

| Configuration | hit@1 | hit@3 | hit@5 | MRR@10 | latency |
|---|---:|---:|---:|---:|---:|
| BGE + original reranker | 0.13 | 0.26 | 0.34 | 0.217 | 1.4 s |
| BGE + **fine-tuned reranker** | 0.18 | 0.30 | 0.42 | 0.265 | 1.1 s |
| BGE (fine-tuned) + fine-tuned reranker | 0.18 | 0.30 | 0.43 | 0.269 | 1.2 s |
| BGE + fine-tuned reranker, 50 candidates | 0.19 | 0.29 | 0.40 | 0.277 | 2.6 s |
| BGE + original reranker, 50 candidates | 0.13 | 0.28 | 0.34 | 0.220 | 2.9 s |
| **MiniLM + fine-tuned reranker (recommended)** | **0.19** | **0.35** | **0.45** | **0.289** | 1.3 s |

- The reranker was fine-tuned on QASPER train (`eval/finetune_reranker.py`: 3,000 pairs, 1 positive + 3 same-paper negatives per question, 2 epochs, about 21 min on CPU). It raised hit@5 from 0.34 to 0.42 with the same candidates.
- A bigger candidate pool (50) did not help and doubled latency, so `CANDIDATE_K` stays 20.
- Fine-tuning the BGE embedder on top added nothing measurable (0.43 vs 0.42, within noise).
- The best result keeps the shipped MiniLM embeddings, so existing documents need no re-upload.
- Caveats: the reranker was trained and evaluated on scientific papers (QASPER); n=100 means differences of a few points are noise, and the gain on real lecture material is unverified. The MiniLM-vs-BGE ordering differs from section 2 because the reranker, not the embedder, now decides the final order.

**Abstention threshold for the fine-tuned reranker.** Fine-tuning changes the score scale: good matches now score near 1.0 (median 0.97 on SQuAD v2), so the old 0.125 threshold rejects almost nothing.

| Threshold | answer rate (SQuAD v2 / QASPER) | refuses unrelated questions (SQuAD v2 / QASPER) |
|---|---|---|
| 0.30 | 0.99 / 1.00 | 0.79 / 0.06 |
| 0.50 | 0.94 / 0.95 | 0.86 / 0.08 |
| 0.70 | 0.88 / 0.95 | 0.94 / 0.18 |

Out-of-corpus AUROC is 0.979 on SQuAD v2 but only 0.63 on QASPER, where unrelated papers are on similar topics. Questions about an indexed document that it does not answer are barely separable (AUROC 0.60 on SQuAD v2, 0.70 on QASPER). The recommended `RELEVANCE_THRESHOLD=0.30` keeps about 99% of answerable questions; raise it for stricter abstention. Do not use 0.30 with the dense-only default, whose cosine scale needs 0.15.

**Recommended profile** (set in `backend/.env`; weights are git-ignored, train them with `python -m eval.finetune_reranker`):
```
USE_HYBRID=1
USE_RERANKER=1
RERANKER_MODEL=models/reranker-lecturemind
RELEVANCE_THRESHOLD=0.30
```

**Citations.** The answer's `[n]` markers are now reconciled with the source list (`align_citations` in `rag_pipeline.py`): only the sources the answer cites are listed, renumbered in order of first use, and video frames are always kept. With the reranker on, the "match" percentage on each source is the reranker's relevance probability instead of a raw MiniLM cosine, so a correct source shows about 85-90% rather than 22-52%.

**Multi-part questions.** The system prompt now has a rule to answer every part of a multi-part question separately with exact figures (rule 7 in `llm_service.py`). Before it, a question such as "stipend for internship, full time?" got only one of the two answers. On a copy of the SalesRaga job description (2 questions, so anecdotal, not a benchmark), both `qwen2.5:1.5b` and `qwen2.5:3b` then answered both parts; `3b` quoted the document exactly with citations, while `1.5b` added an invented sentence on the requirements question. The local `backend/.env` therefore uses `LLM_MODEL=qwen2.5:3b`; the code default stays `qwen2.5:1.5b` for low-RAM machines. This is not a measured accuracy gain: run `eval.run generation` before relying on it.

## 10. Reranker robustness on short documents

**Problem.** On a real six-chunk job-description PDF, "what are their requirements" was answered "not found" in the chat, because after query rewriting the fine-tuned reranker scored the chunk containing the requirements 0.04 (it scored 0.73 for the original phrasing). The old per-chunk cutoff (0.30) then dropped that chunk before the LLM saw it. The reranker was tuned and benchmarked on QASPER, which has long papers with many chunks; on a short document it is much less reliable.

**Test.** 12 hand-written questions about that PDF (requirements, stipend, how to apply, "tell me about the company", with several phrasings each), counting how often the chunk containing the answer is among the chunks sent to the LLM (top 4, after the relevance gate). This is one document and 12 questions, so it is indicative, not a benchmark.

| Setup | Answer chunk reaches the LLM (of 12) | QASPER hit@5 |
|---|---:|---:|
| Dense only | 6 | n/a |
| Hybrid (dense + BM25), no reranker | 9 | n/a |
| Hybrid + original reranker, 0.30 per-chunk cutoff | 4 | 0.34 |
| Hybrid + fine-tuned reranker, 0.30 per-chunk cutoff (before) | 8 | 0.45 |
| Fine-tuned reranker alone, best-chunk gate only | 9 | 0.45 |
| Blend, reranker weight 1 | 11 | 0.32 |
| **Blend, reranker weight 2 (shipped)** | **11** | **0.39** |
| Blend, reranker weight 3 | 9 | 0.42 |
| Blend, reranker weight 5 | 9 | 0.43 |

**Change.** The final order is a weighted reciprocal-rank fusion of the hybrid ranking and the reranker ranking (`RERANK_FUSION`, `RERANK_FUSION_WEIGHT=2`), and the relevance threshold gates only the *best* chunk instead of every chunk (`per_chunk_threshold=False` in the pipeline). The displayed score is still the reranker's probability. This trades benchmark score on long papers (0.45 to 0.39 hit@5) for robustness on short documents; both numbers are in the table so the trade-off is visible. Weights 3 and 5 recover more of the QASPER score but lose the short-document gain.

**Caveats.** The abstention threshold was calibrated on the reranker's raw probability, which is unchanged. The blend only affects ordering. Which weight is best for real lecture material is unknown until there is an in-domain labelled set (see TASKS item 29).
