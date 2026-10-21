"""Evaluation metrics: retrieval, answer quality, abstention, and threshold calibration."""
import math
import re
import string
from collections import Counter

import numpy as np


# --- text normalisation (SQuAD official style) -------------------------------
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction, golds):
    p = normalize_answer(prediction)
    return float(any(p == normalize_answer(g) for g in golds))


def token_f1(prediction, golds):
    def f1(pred, gold):
        p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
        common = Counter(p) & Counter(g)
        same = sum(common.values())
        if not p or not g:
            return float(p == g)
        if same == 0:
            return 0.0
        precision, recall = same / len(p), same / len(g)
        return 2 * precision * recall / (precision + recall)

    return max((f1(prediction, g) for g in golds), default=0.0)


def answer_recall(prediction, golds):
    """Fraction of gold-answer tokens that appear in the prediction. LLM answers are
    full sentences, so EM/F1 under-reward correct answers; this complements them."""
    p = set(normalize_answer(prediction).split())
    best = 0.0
    for g in golds:
        g_tokens = normalize_answer(g).split()
        if g_tokens:
            best = max(best, sum(t in p for t in g_tokens) / len(g_tokens))
    return best


# --- retrieval relevance -----------------------------------------------------
def _ngrams(text, n=4):
    tokens = normalize_answer(text).split()
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def is_relevant(chunk_text, evidence, threshold=0.5):
    """A chunk is relevant if it covers >= threshold of an evidence passage's 4-grams,
    or if the chunk lies (mostly) inside the evidence passage."""
    c = _ngrams(chunk_text)
    for ev in evidence:
        e = _ngrams(ev)
        if not e or not c:
            continue
        inter = len(c & e)
        if inter / len(e) >= threshold or inter / len(c) >= threshold:
            return True
    return False


def retrieval_metrics(relevance_lists, k):
    """relevance_lists: one list of 0/1 flags per query, in ranked order."""
    hits, mrr, ndcg = [], [], []
    for rel in relevance_lists:
        top = rel[:k]
        hits.append(float(any(top)))
        first = next((i for i, r in enumerate(rel) if r), None)
        mrr.append(1.0 / (first + 1) if first is not None and first < k else 0.0)
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(top))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(sum(rel), k)))
        ndcg.append(dcg / ideal if ideal else 0.0)
    n = max(len(relevance_lists), 1)
    return {f"hit@{k}": sum(hits) / n, f"mrr@{k}": sum(mrr) / n, f"ndcg@{k}": sum(ndcg) / n}


# --- threshold calibration ---------------------------------------------------
def auroc(pos_scores, neg_scores):
    """Probability a random positive scores higher than a random negative."""
    if not pos_scores or not neg_scores:
        return float("nan")
    pos, neg = np.asarray(pos_scores), np.asarray(neg_scores)
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def sweep_threshold(pos_scores, neg_scores, grid=None):
    """Answer if score >= t. pos = should answer, neg = should abstain.
    Returns one row per threshold with answer-rate on positives and
    abstain-rate on negatives, plus the threshold maximising balanced accuracy."""
    grid = grid if grid is not None else [round(x, 3) for x in np.linspace(0, 1, 41)]
    rows = []
    for t in grid:
        answered = sum(s >= t for s in pos_scores) / max(len(pos_scores), 1)
        rejected = sum(s < t for s in neg_scores) / max(len(neg_scores), 1)
        rows.append({"threshold": t, "answer_rate_in_scope": answered,
                     "abstain_rate_out_of_scope": rejected,
                     "balanced_accuracy": (answered + rejected) / 2})
    best = max(rows, key=lambda r: (r["balanced_accuracy"], -r["threshold"]))
    return rows, best


def binary_classification(y_true, y_pred):
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def percentile(values, q):
    return float(np.percentile(values, q)) if values else float("nan")
