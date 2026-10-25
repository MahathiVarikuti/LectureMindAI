"""Loaders that turn public benchmarks into one common format.

Document: {"doc_id", "title", "pages": [{"page", "text"}]}
Question: {"qid", "question", "answers": [str], "answerable": bool,
           "doc_id", "evidence": [str], "history": [{"role", "content"}]}

Datasets (all downloaded as parquet from the Hugging Face Hub):
  qasper    - questions about full NLP research papers; long, technical documents
              with section structure, closest to lecture notes/slides. Has
              unanswerable questions.  (allenai/qasper)
  squad_v2  - Wikipedia articles; 1/3 of questions are deliberately unanswerable
              from the text, ideal for calibrating abstention.  (rajpurkar/squad_v2)
  coqa      - multi-turn conversations over a passage; follow-up questions use
              pronouns, which tests query rewriting.  (stanfordnlp/coqa)
  ragtruth  - LLM responses with human word-level hallucination labels; used to
              evaluate the hallucination detector.  (wandb/RAGTruth-processed)
"""
import os
import random
import urllib.request

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SOURCES = {
    "qasper": "https://huggingface.co/api/datasets/allenai/qasper/parquet/qasper/{split}/0.parquet",
    "squad_v2": "https://huggingface.co/api/datasets/rajpurkar/squad_v2/parquet/squad_v2/{split}/0.parquet",
    "coqa": "https://huggingface.co/api/datasets/stanfordnlp/coqa/parquet/default/{split}/0.parquet",
    "ragtruth": "https://huggingface.co/api/datasets/wandb/RAGTruth-processed/parquet/default/{split}/0.parquet",
}


def load_parquet(name, split, url=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}_{split}.parquet")
    if not os.path.exists(path):
        print(f"downloading {name}/{split} ...")
        urllib.request.urlretrieve(url or SOURCES[name].format(split=split), path)
    return pd.read_parquet(path)


def _split_docs(docs, questions, n_docs, n_out_of_scope_docs, seed):
    """Keep n_docs documents in the corpus. Answerable questions about another
    n_out_of_scope_docs documents (NOT indexed) become out-of-scope questions:
    the system should refuse them."""
    rng = random.Random(seed)
    order = list(docs)
    rng.shuffle(order)
    indexed = set(order[:n_docs])
    held_out = set(order[n_docs:n_docs + n_out_of_scope_docs])
    kept_q = []
    for q in questions:
        if q["doc_id"] in indexed:
            q["scope"] = "in_scope"
            kept_q.append(q)
        elif q["doc_id"] in held_out and q["answerable"]:
            q["scope"] = "out_of_scope"
            q["answerable"] = False
            kept_q.append(q)
    return [docs[d] for d in order[:n_docs]], kept_q


# --- QASPER ------------------------------------------------------------------
def _qasper_answer(ans):
    if ans["unanswerable"]:
        return None
    if ans["yes_no"] is not None:
        return "Yes" if ans["yes_no"] else "No"
    if ans["free_form_answer"]:
        return ans["free_form_answer"]
    spans = list(ans["extractive_spans"])
    return ", ".join(spans) if spans else None


def load_qasper(split="validation", n_docs=40, n_out_of_scope_docs=15, seed=0):
    df = load_parquet("qasper", split)
    docs, questions = {}, []
    for _, row in df.iterrows():
        ft = row["full_text"]
        pages = [{"page": 1, "text": f"{row['title']}\n\nAbstract\n{row['abstract']}"}]
        for i, (name, paras) in enumerate(zip(ft["section_name"], ft["paragraphs"]), start=2):
            text = "\n\n".join(p for p in paras if p)
            if text.strip():
                pages.append({"page": i, "text": f"{name or ''}\n{text}".strip()})
        docs[row["id"]] = {"doc_id": row["id"], "title": row["title"], "pages": pages}

        qas = row["qas"]
        for question, qid, annotations in zip(qas["question"], qas["question_id"], qas["answers"]):
            answers, evidence, unanswerable_votes = [], [], 0
            for ann in annotations["answer"]:
                a = _qasper_answer(ann)
                if a is None:
                    unanswerable_votes += 1
                else:
                    answers.append(a)
                evidence += [e for e in ann["evidence"] if e and not e.startswith("FLOAT SELECTED")]
            answerable = unanswerable_votes * 2 < len(annotations["answer"]) and bool(answers)
            # answerable questions whose evidence is only a table/figure can't be judged
            if answerable and not evidence:
                continue
            questions.append({
                "qid": qid, "question": question, "answers": answers, "answerable": answerable,
                "doc_id": row["id"], "evidence": sorted(set(evidence)), "history": [],
            })
    return _split_docs(docs, questions, n_docs, n_out_of_scope_docs, seed)


# --- SQuAD 2.0 ---------------------------------------------------------------
def load_squad_v2(split="validation", n_docs=15, n_out_of_scope_docs=10, seed=0):
    df = load_parquet("squad_v2", split)
    docs, questions = {}, []
    for title, group in df.groupby("title", sort=False):
        contexts = list(dict.fromkeys(group["context"]))
        docs[title] = {
            "doc_id": title, "title": title,
            "pages": [{"page": i + 1, "text": c} for i, c in enumerate(contexts)],
        }
        for _, row in group.iterrows():
            answers = list(dict.fromkeys(row["answers"]["text"]))
            questions.append({
                "qid": row["id"], "question": row["question"], "answers": answers,
                "answerable": bool(answers), "doc_id": title, "evidence": [row["context"]],
                "history": [],
            })
    return _split_docs(docs, questions, n_docs, n_out_of_scope_docs, seed)


# --- CoQA --------------------------------------------------------------------
def load_coqa(split="validation", n_docs=60, seed=0):
    df = load_parquet("coqa", split)
    docs, questions = {}, []
    for idx, row in df.iterrows():
        doc_id = f"coqa_{idx}"
        story = row["story"]
        docs[doc_id] = {"doc_id": doc_id, "title": doc_id, "pages": [{"page": 1, "text": story}]}
        history = []
        ans = row["answers"]
        for turn, (q, a, s, e) in enumerate(
            zip(row["questions"], ans["input_text"], ans["answer_start"], ans["answer_end"])
        ):
            rationale = story[s:e] if 0 <= s < e else ""
            questions.append({
                "qid": f"{doc_id}_{turn}", "question": q, "answers": [a], "answerable": True,
                "doc_id": doc_id, "evidence": [rationale] if len(rationale.split()) >= 4 else [],
                "history": list(history), "turn": turn,
            })
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return _split_docs(docs, questions, n_docs, 0, seed)


def load_hotpotqa(split="validation", n_docs=60, n_out_of_scope_docs=15, seed=0):
    """Load HotpotQA's multi-hop questions into the shared document format."""
    df = load_parquet(
        "hotpotqa",
        split,
        url=f"https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/{split}/0.parquet",
    )
    docs, questions = {}, []
    for _, row in df.iterrows():
        doc_id = row["id"]
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        pages = [{"page": i + 1, "text": f"{title}\n" + " ".join(text)}
                 for i, (title, text) in enumerate(zip(titles, sentences))]
        docs[doc_id] = {"doc_id": doc_id, "title": doc_id, "pages": pages}
        supporting = set(row["supporting_facts"]["title"])
        evidence = [p["text"] for p in pages if p["text"].split("\n", 1)[0] in supporting]
        answer = str(row["answer"])
        questions.append({
            "qid": doc_id, "question": row["question"], "answers": [] if answer == "noanswer" else [answer],
            "answerable": answer != "noanswer", "doc_id": doc_id, "evidence": evidence, "history": [],
        })
    return _split_docs(docs, questions, n_docs, n_out_of_scope_docs, seed)


# --- RAGTruth ----------------------------------------------------------------
def load_ragtruth(split="test", task_types=("QA",), n=300, seed=0):
    df = load_parquet("ragtruth", split)
    df = df[df["task_type"].isin(task_types)]
    rows = []
    for _, r in df.iterrows():
        labels = r["hallucination_labels_processed"]
        rows.append({
            "id": r["id"], "query": r["query"], "context": r["context"], "output": r["output"],
            "model": r["model"],
            "hallucinated": (labels["evident_conflict"] + labels["baseless_info"]) > 0,
        })
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


# --- RAGBench ----------------------------------------------------------------
# rungalileo/ragbench: 12 domain subsets with RAG responses labelled for
# adherence (faithfulness) at response and sentence level. Domains relevant to
# lecture material: techqa, emanual, pubmedqa, covidqa, hotpotqa.
# NOTE: column names (`question`, `documents`, `response`, `adherence_score`)
# follow the dataset card and have not been verified against a download yet.
RAGBENCH_SUBSETS = ("techqa", "emanual", "pubmedqa", "covidqa", "hotpotqa", "delucionqa", "expertqa")


def load_ragbench(split="test", subsets=("techqa", "emanual", "pubmedqa", "hotpotqa"), n=300, seed=0):
    rows = []
    for subset in subsets:
        url = ("https://huggingface.co/api/datasets/rungalileo/ragbench/parquet/"
               f"{subset}/{split}/0.parquet")
        df = load_parquet(f"ragbench_{subset}", split, url=url)
        for _, r in df.iterrows():
            docs = [d for d in r["documents"] if d]
            rows.append({
                "id": f"{subset}_{r['id']}", "query": r["question"], "context": "\n\n".join(docs),
                "documents": docs, "output": r["response"], "model": subset,
                "hallucinated": not bool(r["adherence_score"]),
            })
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


LOADERS = {
    "qasper": load_qasper, "squad_v2": load_squad_v2, "coqa": load_coqa,
    "hotpotqa": load_hotpotqa,
}
