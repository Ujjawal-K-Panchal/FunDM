"""CLM baselines: released CLM evaluation and fine-tuning data.

python scripts/baselines.py <command> [args]; commands: eval_mmlu_pro, eval_sets, choice_data.
"""

import argparse
import collections
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset

from data import DATASETS
import data

# ---------------------------------------------------------------- eval_mmlu_pro
ECE_BINS = 15


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error of the top pick, equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def summarize(probs: list[np.ndarray], gold: list[int], cats: list[str]) -> dict:
    pred = np.array([int(p.argmax()) for p in probs])
    g = np.array(gold)
    conf = np.array([float(p.max()) for p in probs])
    correct = (pred == g).astype(float)
    nll = float(np.mean([-math.log(max(float(p[i]), 1e-12)) for p, i in zip(probs, gold)]))
    per = collections.defaultdict(list)
    for c, ok in zip(cats, correct):
        per[c].append(ok)
    return {"accuracy": float(correct.mean()), "nll": nll, "ece": ece(conf, correct),
            "mean_confidence": float(conf.mean()),
            "per_category": {c: {"accuracy": float(np.mean(v)), "n": len(v)} for c, v in sorted(per.items())}}


def git_head(path: str) -> str:
    return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def eval_mmlu_pro(argv=None) -> None:
    """CLM's released head on MMLU-Pro (gold + up to 9 wrong options), with the clm-raw ablation.
    
    State = the question through CLM's state_text (stripped); candidates = the options verbatim, as Engine.rank sends them.
    Writes results/clm_mmlu_pro.json and per-question predictions for paired comparisons.
    """
    from clm.embedder import Embedder
    from clm.heads import HeadPair
    from clm.engine import RAW_SCALE
    from clm.schema import state_text
    ap = argparse.ArgumentParser(prog="baselines.py eval_mmlu_pro", description=eval_mmlu_pro.__doc__)
    ap.add_argument("--emb-url", default="http://127.0.0.1:8090/v1/embeddings")
    ap.add_argument("--limit", type=int, default=0, help="first N questions; 0 = all")
    ap.add_argument("--chunk", type=int, default=2048, help="texts per progress step")
    ap.add_argument("--out", default="results/clm_mmlu_pro.json")
    a = ap.parse_args(argv)

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", revision=DATASETS["TIGER-Lab/MMLU-Pro"])
    if a.limit:
        ds = ds.select(range(a.limit))
    qs = [state_text(r["question"], None) for r in ds]   # Engine strips the state; 711 questions have edge whitespace
    opts = [list(r["options"]) for r in ds]
    gold = [int(r["answer_index"]) for r in ds]
    cats = [r["category"] for r in ds]
    texts = list(dict.fromkeys(qs + [o for os_ in opts for o in os_]))
    print(f"[eval] {len(ds)} questions, {len(texts)} unique texts", flush=True)

    emb = Embedder(url=a.emb_url, model="qwen3-8b", batch=64, cache_size=len(texts) + 1)
    vec: dict[str, np.ndarray] = {}
    tokens, t0 = 0, time.time()
    for i in range(0, len(texts), a.chunk):
        part = texts[i:i + a.chunk]
        v, tk = emb.embed(part)
        tokens += tk
        vec.update(zip(part, v))
        done, el = i + len(part), time.time() - t0
        print(f"[eval] embedded {done}/{len(texts)} texts, {tokens} tokens, {el:.0f}s, "
              f"ETA {el / done * (len(texts) - done) / 60:.1f} min", flush=True)
    embed_s = time.time() - t0

    head = HeadPair("clm-latest", os.environ["CLM_CKPT"])
    head.ensure()
    Q = np.stack([vec[q] for q in qs])
    flat = [o for os_ in opts for o in os_]
    A = np.stack([vec[o] for o in flat])
    zq = torch.cat([head.project_states(Q[i:i + 4096]) for i in range(0, len(Q), 4096)]).float()
    za = torch.cat([head.project_actions(A[i:i + 4096]) for i in range(0, len(A), 4096)]).float()
    Qr, Ar = torch.from_numpy(Q).to(zq.device), torch.from_numpy(A).to(zq.device)

    res, k = {"clm": [], "clm_raw": []}, 0
    for i, os_ in enumerate(opts):
        n = len(os_)
        res["clm"].append(torch.softmax(head.scale * (za[k:k + n] @ zq[i]), -1).cpu().numpy())
        res["clm_raw"].append(torch.softmax(RAW_SCALE * (Ar[k:k + n] @ Qr[i]), -1).cpu().numpy())
        k += n

    out = {"dataset": "TIGER-Lab/MMLU-Pro test", "n_questions": len(ds),
           "chance": float(np.mean([1 / len(o) for o in opts])),
           "options_per_question": dict(sorted(collections.Counter(len(o) for o in opts).items())),
           "clm": summarize(res["clm"], gold, cats), "clm_raw": summarize(res["clm_raw"], gold, cats),
           "head_scale": head.scale, "encoder_tokens": tokens, "embed_seconds": embed_s,
           "ms_per_question": 1000 * embed_s / len(ds),
           "commits": {"repo": git_head("."), "CLM": git_head("third_party/CLM")},
           "checkpoint": os.environ["CLM_CKPT"]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    with open(a.out.replace(".json", "_preds.jsonl"), "w") as f:
        for r, g, pc, pr in zip(ds, gold, res["clm"], res["clm_raw"]):
            f.write(json.dumps({"question_id": r["question_id"], "gold": g, "p_clm": pc.round(6).tolist(),
                                "p_clm_raw": pr.round(6).tolist()}) + "\n")
    print("[eval] clm acc {:.4f} ece {:.4f} | clm_raw acc {:.4f} | chance {:.4f}".format(
        out["clm"]["accuracy"], out["clm"]["ece"], out["clm_raw"]["accuracy"], out["chance"]), flush=True)


# ---------------------------------------------------------------- eval_sets
def pairs_for(name: str, it: dict) -> tuple[str, list[str]]:
    """-> (state text, candidate texts in option order), via CLM's own build_pairs."""
    from clm.schema import build_pairs
    if name == "boolq":
        state, keys, texts = build_pairs(it["passage"], {"q": {"type": "noul", "instructions": it["query"]}})["q"]
        return state, [texts[keys.index("false")], texts[keys.index("true")]]   # options are ["no", "yes"]
    q = {"type": "choice", "instructions": None, "criteria": {str(i): o for i, o in enumerate(it["options"])}}
    state, _, texts = build_pairs(it["question"], {"q": q})["q"]
    return state, texts


def run(name: str, emb, head, out_dir: str) -> None:
    items = data.EVAL_SETS[name]()
    built = [pairs_for(name, it) for it in items]
    texts = list(dict.fromkeys([s for s, _ in built] + [c for _, cs in built for c in cs]))
    t0 = time.time()
    vec, tokens = {}, 0
    for i in range(0, len(texts), 2048):
        part = texts[i:i + 2048]
        v, tk = emb.embed(part)
        vec.update(zip(part, v)); tokens += tk
    zs = head.project_states(np.stack([vec[s] for s, _ in built])).float()
    flat = [c for _, cs in built for c in cs]
    za = torch.cat([head.project_actions(np.stack([vec[c] for c in flat[i:i + 4096]]))
                    for i in range(0, len(flat), 4096)]).float()
    probs, k = [], 0
    for i, (_, cs) in enumerate(built):
        probs.append(torch.softmax(head.scale * (za[k:k + len(cs)] @ zs[i]), -1).cpu().numpy())
        k += len(cs)
    gold = [it["gold"] for it in items]
    res = {"dataset": name, "n_questions": len(items), "chance": float(np.mean([1 / len(it["options"]) for it in items])),
           "clm": summarize(probs, gold, [it["category"] for it in items]),
           "encoder_tokens": tokens, "embed_seconds": time.time() - t0, "head_scale": head.scale,
           "question_type": "noul" if name == "boolq" else "choice"}
    out = os.path.join(out_dir, f"clm_{name}.json")
    json.dump(res, open(out, "w"), indent=1)
    with open(out.replace(".json", "_preds.jsonl"), "w") as f:
        for it, p in zip(items, probs):
            f.write(json.dumps({"question_id": it["id"], "gold": it["gold"], "p_clm": p.round(6).tolist()}) + "\n")
    print(f"[clm] {name}: n {len(items)} acc {res['clm']['accuracy']:.4f} ece {res['clm']['ece']:.4f} "
          f"chance {res['chance']:.4f} ({res['embed_seconds'] / 60:.1f} min)", flush=True)


def eval_sets(argv=None) -> None:
    """CLM's released head on the lm-eval-style MCQ sets, formatted exactly as CLM's Engine formats them.
    
    Choice sets: state = the question, candidates = the options (a "choice" question). BoolQ uses CLM's
    yes/no type: state = the passage, instructions = the question. Writes results/clm_<set>.json and preds.
    """
    from clm.embedder import Embedder
    from clm.heads import HeadPair
    ap = argparse.ArgumentParser(prog="baselines.py eval_sets", description=eval_sets.__doc__)
    ap.add_argument("--sets", nargs="+", default=list(data.EVAL_SETS))
    ap.add_argument("--emb-url", default="http://127.0.0.1:8090/v1/embeddings")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args(argv)
    emb = Embedder(url=a.emb_url, model="qwen3-8b", batch=64, cache_size=2_000_000)
    head = HeadPair("clm-latest", os.environ["CLM_CKPT"]).ensure()
    for name in a.sets:
        run(name, emb, head, a.out_dir)


# ---------------------------------------------------------------- choice_data
KEYS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OUT = "data/clm_choice/mcq"


def rows(items: list[dict], qid: str, prefix: str) -> list[dict]:
    out = []
    for n, it in enumerate(items):
        crit = {KEYS[i]: o.strip() for i, o in enumerate(it["options"])}
        out.append({"id": f"{prefix}-{it['id']}-{n}", "state": it["question"].strip(),
                    "questions": json.dumps({qid: {"type": "choice", "criteria": crit, "instructions": ""}}),
                    "gold": json.dumps({qid: {"label": KEYS[it["gold"]]}})})
    return out


def choice_data(argv=None) -> None:
    """Our MCQ data in CLM's typed-decision format, so CLM's own finetune.py (--task choice) can train on it.
    
    train: the head's training items (ARC, SciQ, MedMCQA, padded as in mcq.py extract_features).
    test: the 8 evaluation sets; each set is its own question id, so CLM's per-question accuracy is per-set accuracy.
    Each row: state = the question (CLM's state_text(question, None)), one choice question over the option texts.
    """
    os.makedirs(OUT, exist_ok=True)
    train = data.arc() + data.sciq() + data.medmcqa(35000, 0)
    data.pad_with_borrowed(train, 0.5, 10, 0)
    pq.write_table(pa.Table.from_pylist(rows(train, "mcq", "train")), os.path.join(OUT, "train.parquet"))
    test = rows(data.mmlu_pro(), "mmlu_pro", "mmlu_pro")
    for name, fn in data.EVAL_SETS.items():
        test += rows(fn(), name, name)
    pq.write_table(pa.Table.from_pylist(test), os.path.join(OUT, "test.parquet"))
    print(f"[clm-choice] train {len(train)} rows, test {len(test)} rows -> {OUT}")


COMMANDS = {"eval_mmlu_pro": eval_mmlu_pro, "eval_sets": eval_sets, "choice_data": choice_data}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit("usage: python scripts/baselines.py {eval_mmlu_pro|eval_sets|choice_data} [args]")
    COMMANDS[sys.argv[1]](sys.argv[2:])
