"""CoderForge agent selection: data, judge, KD states, summaries, action-trained baselines, released CLM.

python scripts/coderforge.py <command> [args]; commands: fetch, judge, cf_extract, evaluate, summary, embed, together, joint,
released_clm, adp_overlap.
"""

import argparse
import collections
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import statistics
import sys
import time
import urllib.request

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from data import EXIT_LAYER, revision_for
from heads import ANSWER, SEP, HEADS, soft_ce
from mcq import all_logits, question_split

# ---------------------------------------------------------------- fetch
REPO, SHARDS = "togethercomputer/CoderForge-Preview", 224
CONSTANT = {"agent_name": "openhands", "model": "coderforge", "n_steps": 0, "domain": "SWE_Rebench"}


def line_of(row: dict) -> str:
    out = {"trajectory_id": row["trajectory_id"], "task_id": row["trajectory_id"].rsplit("_run", 1)[0],
           "reward": row["reward"], "messages": json.loads(row["messages"]), **CONSTANT}
    return json.dumps(out) + "\n"


def fetch(argv=None) -> None:
    """Rebuild data/coderforge/{train,val}.jsonl from togethercomputer/CoderForge-Preview (trajectories, SWE_Rebench).
    
    The runs and their order are listed in results/coderforge/runs.tsv. Each line adds task_id (the trajectory id without
    its _runN suffix) and four constant fields to the dataset's row, as in the files the results used; the script checks
    the rebuilt files against results/coderforge/data_sha256.txt. Downloads the split's 224 shards (18.8 GB).
    
    python scripts/coderforge.py fetch [--out data/coderforge]
    """
    ap = argparse.ArgumentParser(prog="coderforge.py fetch", description=fetch.__doc__)
    ap.add_argument("--out", default="data/coderforge")
    ap.add_argument("--manifest", default="results/coderforge/runs.tsv")
    a = ap.parse_args(argv)
    want = [l.rstrip("\n").split("\t") for l in open(a.manifest)][1:]
    split_of = {tid: split for split, tid in want}
    lines = {}
    for k in range(SHARDS):
        path = hf_hub_download(REPO, f"trajectories/SWE_Rebench-{k:05d}-of-{SHARDS:05d}.parquet", repo_type="dataset",
                               revision=revision_for(REPO))
        for row in pq.read_table(path, columns=["trajectory_id", "reward", "messages"]).to_pylist():
            if row["trajectory_id"] in split_of:
                lines[row["trajectory_id"]] = line_of(row)
        print(f"[fetch] shard {k + 1}/{SHARDS}: {len(lines)}/{len(want)} runs found", flush=True)
    missing = [tid for _, tid in want if tid not in lines]
    if missing:
        sys.exit(f"[fetch] {len(missing)} runs missing, e.g. {missing[:3]}")
    os.makedirs(a.out, exist_ok=True)
    sums = {l.split()[-1]: l.split()[0] for l in open("results/coderforge/data_sha256.txt") if not l.startswith("#")}
    for split in ("train", "val"):
        path = os.path.join(a.out, f"{split}.jsonl")
        with open(path, "w") as f:
            f.writelines(lines[tid] for s, tid in want if s == split)
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        print(f"[fetch] {path}: sha256 {'matches' if digest == sums[f'{split}.jsonl'] else 'DIFFERS: ' + digest}")


# ---------------------------------------------------------------- judge
MODEL, CAP, TASK_CAP = "Qwen/Qwen3-8B", 8191, 2048
JUDGE_QUESTION = ("Above is an AI agent's attempt to resolve the issue. Will the repository's hidden tests pass with the "
            "agent's final changes? Answer with yes or no.")


def ids_of(tok, msgs, **kw) -> list[int]:
    x = tok.apply_chat_template(msgs, tokenize=True, **kw)
    return x["input_ids"] if hasattr(x, "keys") else x


def judge_ids(tok, msgs: list[dict]) -> list[int]:
    full = ids_of(tok, msgs, add_generation_prompt=False)
    n1, n2 = len(ids_of(tok, msgs[:1], add_generation_prompt=False)), len(ids_of(tok, msgs[:2], add_generation_prompt=False))
    task = full[n1:n2][:TASK_CAP]
    q = ids_of(tok, [{"role": "user", "content": JUDGE_QUESTION}], add_generation_prompt=True, enable_thinking=False)
    rest = full[n2:][-(CAP - len(task) - len(q)):]
    return task + rest + q


def judge(argv=None) -> None:
    """Qwen3-8B's own verdict on each CoderForge run, at the head's cost: one forward pass of at most 8191 tokens.
    
    Input: the task statement (first 2048 tokens) plus the run's latest tokens, then a yes/no question.
    Score = log p(yes) - log p(no) at the answer position, thinking disabled.
    """
    ap = argparse.ArgumentParser(prog="coderforge.py judge", description=judge.__doc__)
    ap.add_argument("--data", default="data/coderforge")
    ap.add_argument("--tasks", default="results/coderforge/tasks.json")
    ap.add_argument("--out", default="results/coderforge/judge.jsonl")
    a = ap.parse_args(argv)
    tasks = json.load(open(a.tasks))
    want = {"val.jsonl": set(tasks["holdout"]), "train.jsonl": set(tasks["train_pool"])}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = {json.loads(l)["trajectory_id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision_for(MODEL))
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision_for(MODEL), dtype=torch.bfloat16).cuda().eval()
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]
    n = 0
    with open(a.out, "a") as f, torch.no_grad():
        for name, keep in want.items():
            for line in open(os.path.join(a.data, name)):
                r = json.loads(line)
                if r["task_id"] not in keep or r["trajectory_id"] in done:
                    continue
                msgs = json.loads(r["messages"]) if isinstance(r["messages"], str) else r["messages"]
                x = torch.tensor([judge_ids(tok, msgs)], device="cuda")
                lp = torch.log_softmax(model(input_ids=x, logits_to_keep=1).logits[0, -1].float(), -1)
                score = (torch.logsumexp(lp[yes], 0) - torch.logsumexp(lp[no], 0)).item()
                f.write(json.dumps({"trajectory_id": r["trajectory_id"], "task_id": r["task_id"], "reward": r["reward"],
                                    "split": "test" if name == "val.jsonl" else "train_pool", "score": score,
                                    "p_yes_no_mass": torch.exp(lp[yes + no]).sum().item()}) + "\n")
                f.flush(); n += 1
                if n % 50 == 0:
                    print(f"[judge] {n} runs scored", flush=True)
    print("[judge] done", flush=True)


# ---------------------------------------------------------------- cf_extract
MODEL, CAP, TASK_CAP = "Qwen/Qwen3-8B", 8191, 2048
OPTIONS_QUESTION = ("Above is an AI agent's attempt to resolve the issue. Will the repository's hidden tests pass with the "
            "agent's final changes? Answer with the letter of the correct option.\nOptions:\n{}\n{}")
OPTS = ["yes", "no"]  # original index 0 is "yes"
JUDGE = "results/coderforge/judge.jsonl"  # scripts/coderforge.py judge; run from the repo root
TASKS = "results/coderforge/tasks.json"  # task split written by scripts/coderforge.py embed
DATA = "data/coderforge"


def question(tok, order: list[int]):
    """-> question ids (a user turn plus the empty-thinking assistant prefix), option-end positions within them."""
    lines = [f"{'AB'[s]}. {OPTS[i]}" for s, i in enumerate(order)]
    msg = [{"role": "user", "content": OPTIONS_QUESTION.format(*lines)}]
    text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == ids_of(tok, msg, add_generation_prompt=True, enable_thinking=False)
    ends = []
    for line in lines:
        e = text.index(line) + len(line) - 1  # char index of the option's last character
        ends.append(next(j for j, (_, s1) in enumerate(enc["offset_mapping"]) if s1 > e))
    return enc["input_ids"], ends


def student(tok, msgs: list[dict], q: list[int], q_ends: list[int]):
    """Same truncation as scripts/coderforge.py judge, with the options question."""
    full = ids_of(tok, msgs, add_generation_prompt=False)
    n1 = len(ids_of(tok, msgs[:1], add_generation_prompt=False))
    n2 = len(ids_of(tok, msgs[:2], add_generation_prompt=False))
    task = full[n1:n2][:TASK_CAP]
    rest = full[n2:][-(CAP - len(task) - len(q)):]
    ids = task + rest + q
    return ids, [len(task) + len(rest) + e for e in q_ends], len(ids) - 1


def cf_extract(argv=None) -> None:
    """CoderForge judge distillation: one student pass per run; the teacher is the judge's existing score.
    
    Student input: the judge's input (task statement's first 2048 tokens, then the run's latest tokens; 8191 tokens in all)
    with the question asked as options "A. yes / B. no", order shuffled per run. Saves, per layer L, the states at the end
    of each option and at the last input token, and the teacher logits per slot from results/coderforge/judge.jsonl
    (log p(yes) - log p(no) for yes, 0 for no). One file per layer: <out>_L<L>.pt
    """
    ap = argparse.ArgumentParser(prog="coderforge.py cf_extract", description=cf_extract.__doc__)
    ap.add_argument("--set", choices=["test", "train"], required=True)
    ap.add_argument("--layers", default="16,20,24,28,32")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="prefix; writes <out>_L<layer>.pt")
    ap.add_argument("--groups", default="", help="if set, write trajectory id -> task id here")
    a = ap.parse_args(argv)
    layers = [int(x) for x in a.layers.split(",")]

    split = "test" if a.set == "test" else "train_pool"
    judge = {r["trajectory_id"]: r for r in map(json.loads, open(JUDGE)) if r["split"] == split}
    keep = set(json.load(open(TASKS))["holdout" if a.set == "test" else "train_pool"])
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision_for(MODEL))
    qs = {tuple(o): question(tok, o) for o in ([0, 1], [1, 0])}
    letters = [[tok.encode(w, add_special_tokens=False)[0] for w in (c, " " + c)] for c in "AB"]
    rng = random.Random(a.seed)

    rows = []
    for line in open(os.path.join(DATA, "val.jsonl" if a.set == "test" else "train.jsonl")):
        r = json.loads(line)
        if r["task_id"] not in keep or r["trajectory_id"] not in judge:
            continue
        msgs = json.loads(r["messages"]) if isinstance(r["messages"], str) else r["messages"]
        order = [0, 1]
        rng.shuffle(order)
        ids, ends, fin = student(tok, msgs, *qs[tuple(order)])
        rows.append(dict(ids=ids, ends=ends, fin=fin, order=order, id=r["trajectory_id"], task=r["task_id"],
                         reward=float(r["reward"]), score=judge[r["trajectory_id"]]["score"]))
        if a.limit and len(rows) >= a.limit:
            break
    print(f"[cf] {a.set}: {len(rows)} runs, {len({r['task'] for r in rows})} tasks, "
          f"max {max(len(r['ids']) for r in rows)} tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision_for(MODEL), dtype=torch.bfloat16, device_map="cuda").eval()
    states, mc, t0 = [], [], time.time()
    with torch.no_grad():
        for n, r in enumerate(rows, 1):
            out = model.model(input_ids=torch.tensor([r["ids"]], device="cuda"), output_hidden_states=True)
            pos = torch.tensor(r["ends"] + [r["fin"]], device="cuda")
            states.append(torch.stack([out.hidden_states[L][0, pos] for L in layers]).to(torch.bfloat16).cpu())
            lp = torch.log_softmax(model.lm_head(out.last_hidden_state[0, r["fin"]]).float(), -1)
            mc.append(torch.stack([torch.logsumexp(lp[ids], 0) for ids in letters]).cpu())  # slot A, slot B
            if n % 100 == 0 or n == len(rows):
                el = time.time() - t0
                print(f"[cf] {n}/{len(rows)} runs, {el / n:.2f} s/run, ETA {el / n * (len(rows) - n) / 60:.1f} min", flush=True)

    tl = torch.tensor([[r["score"] if OPTS[i] == "yes" else 0.0 for i in r["order"]] for r in rows])
    mcl = torch.stack(mc)
    meta = {"order": [r["order"] for r in rows], "gold": [0 if r["reward"] > 0 else 1 for r in rows],
            "category": [r["task"] for r in rows], "id": [r["id"] for r in rows], "task": [r["task"] for r in rows],
            "reward": [r["reward"] for r in rows], "set": f"cf_{a.set}", "layers": layers,
            "letter_mass_mean": float(mcl.exp().sum(-1).mean()), "mc": mcl.flatten(), "tl": tl.flatten()}
    off = torch.arange(len(rows) + 1) * 2
    for k, L in enumerate(layers):
        hq = torch.stack([s[k, -1] for s in states])
        hk = torch.cat([s[k, :-1] for s in states])
        torch.save({**meta, "hq": hq, "hk": hk, "off": off, "layer": L}, f"{a.out}_L{L}.pt")
    if a.groups:
        json.dump({r["id"]: r["task"] for r in rows}, open(a.groups, "w"))
    print(f"[cf] saved {a.out}_L*.pt in {(time.time() - t0) / 60:.1f} min; mean A/B letter mass {meta['letter_mass_mean']:.3f}",
          flush=True)


# ---------------------------------------------------------------- evaluate
def picks_auc(score, task, reward):
    by = collections.defaultdict(list)
    for s, t, r in zip(score, task, reward):
        by[t].append((s, r > 0))
    picks = sum(max(v, key=lambda x: x[0])[1] for v in by.values())
    aucs = []
    for v in by.values():
        pos, neg = [s for s, ok in v if ok], [s for s, ok in v if not ok]
        if pos and neg:
            aucs.append(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)))
    rand = sum(sum(ok for _, ok in v) / len(v) for v in by.values())
    return picks, statistics.mean(aucs), rand, len(by)


def yes_minus_no(logits: torch.Tensor, order) -> list[float]:
    """Per-run score: logit of the "yes" slot minus the "no" slot."""
    return [float(logits[i, o.index(0)] - logits[i, o.index(1)]) for i, o in enumerate(order)]


def evaluate(argv=None) -> None:
    """CoderForge best-of-runs selection: tasks whose picked run passes, and within-task AUC (mean over mixed tasks).
    
    python scripts/coderforge.py evaluate --feats feats_cf/cf_test_L24.pt [--preds results/kd_cf/linear_L24_s0_cf_test_preds.pt]
    Without --preds it scores the judge (the teacher), the options-prompt letter readout, and random choice.
    """
    ap = argparse.ArgumentParser(prog="coderforge.py evaluate", description=evaluate.__doc__)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--preds", default="")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    d = torch.load(a.feats, weights_only=False)
    n = len(d["id"])
    rows = {}
    if a.preds:
        p = torch.load(a.preds, weights_only=False)
        assert p["id"] == d["id"]
        rows[a.label or "student"] = yes_minus_no(p["student"], d["order"])
    else:
        rows["teacher (judge)"] = yes_minus_no(d["tl"].view(n, 2), d["order"])
        rows["options-prompt letters"] = yes_minus_no(d["mc"].view(n, 2), d["order"])
    for name, score in rows.items():
        picks, auc, rand, tasks = picks_auc(score, d["task"], d["reward"])
        print(json.dumps({"label": name, "tasks": tasks, "picks": picks, "auc": round(auc, 4), "random": round(rand, 1)}),
              flush=True)


# ---------------------------------------------------------------- summary
SEEDS, LAYERS, MAIN, N_BOOT = range(5), (16, 20, 24, 28, 32), EXIT_LAYER, 10000


def facts(name: str) -> dict:
    d = torch.load(f"feats_cf/{name}_L{MAIN}.pt", weights_only=False)
    score = yes_minus_no(d["tl"].view(-1, 2), d["order"])
    return {"runs": len(d["id"]), "tasks": len(set(d["task"])), "pass": sum(r > 0 for r in d["reward"]) / len(d["id"]),
            "judge_says_yes": sum(s > 0 for s in score) / len(score), "letter_mass": d["letter_mass_mean"]}


def ci(per_task: list[float], stat) -> list[float]:
    """95% paired bootstrap interval over tasks."""
    rng = random.Random(0)
    boots = sorted(stat([rng.choice(per_task) for _ in per_task]) for _ in range(N_BOOT))
    return [boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]]


def vs_teacher(arch: str) -> dict:
    """Head at MAIN layers (five-seed mean per task) minus the judge on the same tasks."""
    d = torch.load(f"feats_cf/cf_test_L{MAIN}.pt", weights_only=False)
    by = collections.defaultdict(list)
    for i, tk in enumerate(d["task"]):
        by[tk].append(i)
    tasks, ok = sorted(by), [r > 0 for r in d["reward"]]

    def pick(sc):
        return {tk: max(by[tk], key=lambda i: sc[i]) for tk in tasks}

    def auc(sc, tk):
        pos, neg = [sc[i] for i in by[tk] if ok[i]], [sc[i] for i in by[tk] if not ok[i]]
        return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

    t = yes_minus_no(d["tl"].view(-1, 2), d["order"])
    tp = pick(t)
    dk, da, swaps, right, wrong = [], [], [], [], []
    for s in SEEDS:
        p = torch.load(f"results/kd_cf/{arch}_L{MAIN}_s{s}_cf_test_preds.pt", weights_only=False)
        sc = yes_minus_no(p["student"], d["order"])
        sp = pick(sc)
        dk.append([int(ok[sp[tk]]) - int(ok[tp[tk]]) for tk in tasks])
        da.append([auc(sc, tk) - auc(t, tk) for tk in tasks])
        swaps.append(sum(sp[tk] != tp[tk] for tk in tasks))
        right.append(sum(ok[sp[tk]] and not ok[tp[tk]] for tk in tasks))
        wrong.append(sum(ok[tp[tk]] and not ok[sp[tk]] for tk in tasks))
    k = [statistics.mean(c) for c in zip(*dk)]
    a = [statistics.mean(c) for c in zip(*da)]
    gaps = []
    for tk in tasks:
        top = sorted((t[i] for i in by[tk]), reverse=True)
        gaps.append(top[0] - top[1])
    return {"picks_diff": sum(k), "picks_ci": ci(k, sum), "auc_diff": statistics.mean(a), "auc_ci": ci(a, statistics.mean),
            "swaps": swaps, "head_right_judge_wrong": right, "judge_right_head_wrong": wrong,
            "judge_top_p_over_0999": sum(t[tp[tk]] > 6.9 for tk in tasks), "judge_top2_gap_under_05": sum(g < 0.5 for g in gaps)}


def summary(argv=None) -> None:
    """CoderForge KD summary for the paper: data facts, judge and full-model rows, five-seed head rows, and the 24-layer
    heads against the judge (paired bootstrap over tasks, pick swaps) -> results/kd_cf/summary.json.
    
    python scripts/coderforge.py summary   (from the repo root, after run_kd_cf.sh and run_kd_cf_seeds.sh)
    """
    d = torch.load(f"feats_cf/cf_test_L{MAIN}.pt", weights_only=False)
    n = len(d["id"])
    out = {"test": facts("cf_test"), "train": facts("cf_train"), "heads": {}}
    rows = [json.loads(line) for line in open(JUDGE)]
    out["judge_yes_no_mass"] = sum(r["p_yes_no_mass"] for r in rows) / len(rows)
    for key, logits in (("teacher", d["tl"].view(n, 2)), ("options_prompt", d["mc"].view(n, 2))):
        picks, auc, rand, _ = picks_auc(yes_minus_no(logits, d["order"]), d["task"], d["reward"])
        out[key] = {"picks": picks, "auc": auc}
        out["random"] = rand
    for arch in ("linear", "comparator"):
        out["heads"][arch] = {}
        for L in LAYERS:
            dl = torch.load(f"feats_cf/cf_test_L{L}.pt", weights_only=False)
            picks, aucs = [], []
            for s in SEEDS:
                p = torch.load(f"results/kd_cf/{arch}_L{L}_s{s}_cf_test_preds.pt", weights_only=False)
                assert p["id"] == dl["id"]
                k, auc, _, _ = picks_auc(yes_minus_no(p["student"], dl["order"]), dl["task"], dl["reward"])
                picks.append(k)
                aucs.append(auc)
            out["heads"][arch][str(L)] = {"picks": picks, "auc": aucs}
    out["vs_teacher"] = {arch: vs_teacher(arch) for arch in ("linear", "comparator")}
    json.dump(out, open("results/kd_cf/summary.json", "w"), indent=1)
    print(json.dumps(out["vs_teacher"], indent=1))


# ---------------------------------------------------------------- embed
MODEL, CAP, STRIDE, CONFIG = "Qwen/Qwen3-8B", 8191, 4096, "openhands_coderforge"
_tok = None


def action_text(m: dict) -> str:
    calls = [json.dumps({"name": c["function"]["name"], "arguments": c["function"]["arguments"]})
             for c in m.get("tool_calls") or []]
    return "\n".join([(m.get("content") or "").strip(), *calls]).strip() or " "


def prepare(row: dict):
    """-> (row id fields, full ids, state end positions, action ids) for one trajectory."""
    global _tok
    _tok = _tok or AutoTokenizer.from_pretrained(MODEL, revision=revision_for(MODEL))
    ids = lambda x: x["input_ids"] if hasattr(x, "keys") else x
    msgs = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
    full = ids(_tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False))
    ends, acts = [], []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        p = ids(_tok.apply_chat_template(msgs[:i], tokenize=True, add_generation_prompt=False))
        assert full[:len(p)] == p, f"{row['trajectory_id']}: chat template is not prefix-consistent at {i}"
        ends.append(len(p))
        acts.append(ids(_tok(action_text(m), add_special_tokens=False))[:CAP])
    return {k: row[k] for k in ("trajectory_id", "task_id", "reward")}, full, ends, acts


def rows_of(path: str, tasks: set[str], done: set[str]):
    for line in open(path):
        r = json.loads(line)
        if r["task_id"] in tasks and r["trajectory_id"] not in done:
            yield r


@torch.no_grad()
def embed_states(model, full: list[int], ends: list[int]) -> torch.Tensor:
    by_start: dict[int, list[int]] = {}
    for j, e in enumerate(ends):
        start = 0 if e <= CAP else math.ceil((e - CAP) / STRIDE) * STRIDE
        by_start.setdefault(start, []).append(j)
    out = torch.empty(len(ends), model.config.hidden_size)
    for start, js in by_start.items():
        stop = max(ends[j] for j in js)
        h = model(input_ids=torch.tensor([full[start:stop]], device="cuda")).last_hidden_state[0]
        out[js] = h[torch.tensor([ends[j] - 1 - start for j in js], device="cuda")].float().cpu()
    return F.normalize(out, dim=-1)


@torch.no_grad()
def embed_texts(model, seqs: list[list[int]], budget: int = 32768) -> torch.Tensor:
    out = torch.empty(len(seqs), model.config.hidden_size)
    order = sorted(range(len(seqs)), key=lambda j: len(seqs[j]))
    i = 0
    while i < len(order):
        n = 1
        while i + n < len(order) and (n + 1) * len(seqs[order[i + n]]) <= budget:
            n += 1
        js = order[i:i + n]; i += len(js)
        L = max(len(seqs[j]) for j in js)
        x = torch.zeros(len(js), L, dtype=torch.long); mask = torch.zeros(len(js), L, dtype=torch.long)
        for r, j in enumerate(js):
            x[r, :len(seqs[j])] = torch.tensor(seqs[j]); mask[r, :len(seqs[j])] = 1
        h = model(input_ids=x.cuda(), attention_mask=mask.cuda()).last_hidden_state
        last = torch.tensor([len(seqs[j]) - 1 for j in js], device="cuda")
        out[js] = h[torch.arange(len(js), device="cuda"), last].float().cpu()
    return F.normalize(out, dim=-1)


def merge(out: str, order: list[str]) -> None:
    samples, st, ac = [], [], []
    for tid in order:
        p = torch.load(os.path.join(out, "parts", f"{tid}.pt"))
        st.append(p["state"]); ac.append(p["action"])
        samples += [{**p["meta"], "step_idx": k, "model": "coderforge", "config": CONFIG} for k in range(len(p["state"]))]
    torch.save(torch.cat(st).half(), os.path.join(out, "state_embeddings.pt"))
    torch.save(torch.cat(ac).half(), os.path.join(out, "action_embeddings.pt"))
    json.dump({"num_samples": len(samples), "hidden_size": int(st[0].shape[1]), "model_name": MODEL,
               "max_model_len": CAP + 1, "state_window_stride": STRIDE, "samples": samples},
              open(os.path.join(out, "metadata.json"), "w"))
    print(f"[merge] {len(order)} trajectories, {len(samples)} steps", flush=True)


def embed(argv=None) -> None:
    """Embed CoderForge SWE-Rebench step traces with Qwen3-8B into a CLM-style embedding dir.
    
    State of step k = chat messages before assistant turn k; action = that turn's text and tool calls.
    Last-token, final-layer vectors, L2-normalised, as in CLM's DeepSWE recipe with one deviation:
    a state longer than 8191 tokens sees its last 4096-8191 tokens (shared 8191-token windows at
    stride 4096), not exactly the last 8191. Resumable: one .pt per trajectory in OUT/parts.
    """
    ap = argparse.ArgumentParser(prog="coderforge.py embed", description=embed.__doc__)
    ap.add_argument("--data", default="data/coderforge")
    ap.add_argument("--out", default="data/clm_emb/coderforge-swerebench-8k")
    ap.add_argument("--n-train-tasks", type=int, default=200)
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
    tasks_of = lambda f: sorted({json.loads(l)["task_id"] for l in open(os.path.join(a.data, f))})
    test = tasks_of("val.jsonl")
    train = sorted(random.Random(0).sample(tasks_of("train.jsonl"), a.n_train_tasks))
    json.dump({"holdout": test, "train_pool": train}, open(os.path.join(a.out, "tasks.json"), "w"), indent=1)
    model = AutoModel.from_pretrained(MODEL, revision=revision_for(MODEL), dtype=torch.bfloat16).cuda().eval()
    done = {f[:-3] for f in os.listdir(os.path.join(a.out, "parts"))}
    for f, tasks in (("val.jsonl", set(test)), ("train.jsonl", set(train))):
        with mp.get_context("spawn").Pool(6) as pool:
            for n, (meta, full, ends, acts) in enumerate(pool.imap(prepare, rows_of(os.path.join(a.data, f), tasks, done), chunksize=2)):
                part = {"meta": meta, "state": embed_states(model, full, ends).half(), "action": embed_texts(model, acts).half()}
                torch.save(part, os.path.join(a.out, "parts", f"{meta['trajectory_id']}.pt"))
                done.add(meta["trajectory_id"])
                if n % 20 == 0:
                    print(f"[{f}] {len(done)} trajectories done, last {len(ends)} steps / {len(full)} tokens", flush=True)
    order = sorted(done)
    merge(a.out, order)


# ---------------------------------------------------------------- together
MODEL, CAP, A_CAP, K = "Qwen/Qwen3-8B", 8192, 320, 8
N_SAME, N_TRAIN, WINDOW = 3, 10000, 12


def make_spec(meta: list[dict], tasks: dict, test_like_train: bool = False) -> dict:
    ok = [float(m["reward"] or 0) > 0.5 for m in meta]
    pool_t, test_t = set(tasks["train_pool"]), set(tasks["holdout"])
    pool = [i for i, m in enumerate(meta) if m["task_id"] in pool_t and ok[i]]
    by_task: dict[str, list[int]] = {}
    for i in pool:
        by_task.setdefault(meta[i]["task_id"], []).append(i)
    rng = random.Random(0)
    rows = []
    for i in rng.sample(pool, N_TRAIN):
        same = [j for j in rng.sample(by_task[meta[i]["task_id"]], min(N_SAME + 1, len(by_task[meta[i]["task_id"]]))) if j != i][:N_SAME]
        rand = []
        while len(rand) < K - 1 - len(same):
            j = rng.choice(pool)
            if j != i and j not in same and j not in rand:
                rand.append(j)
        rows.append({"split": "train", "state": i, "cands": [i] + same + rand})
    refs = random.Random(4321).sample(pool, K - 1)
    runs: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        if m["task_id"] in test_t:
            runs.setdefault(m["trajectory_id"], []).append(i)
    test_ok = [i for i, m in enumerate(meta) if m["task_id"] in test_t and ok[i]]
    t_rng = random.Random(4321)
    for rid in sorted(runs):
        for i in sorted(runs[rid], key=lambda j: meta[j]["step_idx"])[-WINDOW:]:
            cands = [i] + refs
            if test_like_train:  # training-style distractors: 3 passing steps of the same task, other runs, + 4 random
                own = [j for j in test_ok if meta[j]["task_id"] == meta[i]["task_id"] and meta[j]["trajectory_id"] != rid]
                same = t_rng.sample(own, min(N_SAME, len(own)))
                cands = [i] + same + t_rng.sample([j for j in pool if j != i], K - 1 - len(same))
            rows.append({"split": "test", "state": i, "cands": cands})
    for n, r in enumerate(rows):
        order = list(range(len(r["cands"]))); random.Random(n).shuffle(order); r["order"] = order
    return {"refs": refs, "rows": rows}


def together(argv=None) -> None:
    """Read-together CoderForge rows for the option head: one Qwen3-8B pass over a step's state and 8 candidate actions.
    
    Row = state tail + " rubric:" + each candidate action + " ::" + " answer:", as heads.build_row lays out
    MCQ rows; the head reads the answer token and each separator. Training rows: 10k passing steps of the train pool,
    candidates = own action + 3 same-task + 4 random. Test rows: the last 12 steps of every held-out run, candidates =
    own action + 7 fixed references. The same row spec (spec.json) also indexes the read-apart embeddings.
    """
    ap = argparse.ArgumentParser(prog="coderforge.py together", description=together.__doc__)
    ap.add_argument("--emb", default="data/clm_emb/coderforge-swerebench-8k")
    ap.add_argument("--data", default="data/coderforge")
    ap.add_argument("--out", default="data/clm_emb/coderforge-together")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N train and N test rows only")
    ap.add_argument("--state-cap", type=int, default=0, help="keep the last N state tokens, candidates added on top (0: fit all in 8192)")
    ap.add_argument("--test-like-train", action="store_true", help="test candidates drawn like training ones")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.out, "shards"), exist_ok=True)
    meta = json.load(open(os.path.join(a.emb, "metadata.json")))["samples"]
    spec_path = os.path.join(a.out, "spec.json")
    if not os.path.exists(spec_path):
        json.dump(make_spec(meta, json.load(open("results/coderforge/tasks.json")), a.test_like_train), open(spec_path, "w"))
    rows = json.load(open(spec_path))["rows"]
    if a.limit:
        rows = [r for r in rows if r["split"] == "train"][:a.limit] + [r for r in rows if r["split"] == "test"][:a.limit]
    SHARD = a.limit or 500
    need = {meta[j]["trajectory_id"] for r in rows for j in [r["state"], *r["cands"]]}
    msgs = {}
    for f in ("val.jsonl", "train.jsonl"):
        for line in open(os.path.join(a.data, f)):
            r = json.loads(line)
            if r["trajectory_id"] in need:
                msgs[r["trajectory_id"]] = r["messages"] if isinstance(r["messages"], list) else json.loads(r["messages"])
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision_for(MODEL))
    ids = lambda x: x["input_ids"] if hasattr(x, "keys") else x
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    turn = lambda j: [i for i, m in enumerate(msgs[meta[j]["trajectory_id"]]) if m["role"] == "assistant"][meta[j]["step_idx"]]
    act = {}
    def action_ids(j):
        if j not in act:
            m = msgs[meta[j]["trajectory_id"]][turn(j)]
            act[j] = enc(" " + action_text(m))[:A_CAP]
        return act[j]
    sep, rub, ans = enc(SEP), enc(" rubric:"), enc(ANSWER)
    model = AutoModel.from_pretrained(MODEL, revision=revision_for(MODEL), dtype=torch.bfloat16).cuda().eval()
    for s0 in range(0, len(rows), SHARD):
        path = os.path.join(a.out, "shards", f"{s0:06d}.pt")
        if os.path.exists(path):
            continue
        hq, hk = [], []
        for r in rows[s0:s0 + SHARD]:
            m = msgs[meta[r["state"]]["trajectory_id"]]
            state = ids(tok.apply_chat_template(m[:turn(r["state"])], tokenize=True, add_generation_prompt=False))
            tail, sep_pos = [], []
            for slot in r["order"]:
                tail += action_ids(r["cands"][slot]) + sep
                sep_pos.append(len(tail) - 1)
            tail += ans
            state = state[-(a.state_cap or CAP - len(rub) - len(tail)):] + rub
            with torch.no_grad():
                h = model(input_ids=torch.tensor([state + tail], device="cuda")).last_hidden_state[0]
            hq.append(h[-1].to(torch.bfloat16).cpu())
            hk.append(h[torch.tensor([len(state) + p for p in sep_pos], device="cuda")].to(torch.bfloat16).cpu())
        torch.save({"hq": torch.stack(hq), "hk": hk}, path)
        print(f"[together] {min(s0 + SHARD, len(rows))}/{len(rows)} rows", flush=True)
    parts = [torch.load(os.path.join(a.out, "shards", f), weights_only=False) for f in sorted(os.listdir(os.path.join(a.out, "shards")))]
    torch.save({"hq": torch.cat([p["hq"] for p in parts]), "hk": [x for p in parts for x in p["hk"]],
                "rows": rows, "model": MODEL}, os.path.join(a.out, "features.pt"))
    print("[together] TOGETHER_DONE", flush=True)


# ---------------------------------------------------------------- joint
HERE = os.path.dirname(os.path.abspath(__file__))

SETS = os.environ.get("JOINT_SETS", "mmlu_pro mmlu openbookqa commonsenseqa hellaswag piqa winogrande boolq").split()
EMB = "data/clm_emb/coderforge-swerebench-8k"
TOG = os.environ.get("JOINT_TOG", "data/clm_emb/coderforge-together/features.pt")
WINDOW, N_TASK, N_RAND = 12, 15, 16
LR = {"linear": 1e-3, "comparator": 5e-4, "clm_mlp": 5e-4}


def mcq_batch(f: dict, idx: list[int], device: str, scale: float):
    K = max(len(f["order"][i]) for i in idx)
    hk = torch.zeros(len(idx), K, f["hq"].shape[1])
    mask = torch.zeros(len(idx), K, dtype=torch.bool)
    for b, i in enumerate(idx):
        hk[b, :len(f["order"][i])] = f["hk"][i].float(); mask[b, :len(f["order"][i])] = True
    slot = torch.tensor([f["order"][i].index(f["gold"][i]) for i in idx])
    return f["hq"][idx].float().to(device) * scale, hk.to(device) * scale, mask.to(device), slot.to(device)


class AgentApartSteps:
    """Mode A: steps of the merged embedding dir; candidates drawn per batch (own + same-task + random)."""

    def __init__(self, split: dict, device: str):
        self.meta = json.load(open(os.path.join(EMB, "metadata.json")))["samples"]
        H = 4096
        self.st = torch.load(os.path.join(EMB, "state_embeddings.pt"), mmap=True).to(device).float() * H ** 0.5
        self.ac = torch.load(os.path.join(EMB, "action_embeddings.pt"), mmap=True).to(device).float() * H ** 0.5
        ok = [float(m["reward"] or 0) > 0.5 for m in self.meta]
        tr, va = set(split["train_tasks"]), set(split["val_tasks"])
        self.train = [i for i, m in enumerate(self.meta) if m["task_id"] in tr and ok[i]]
        self.val = [i for i, m in enumerate(self.meta) if m["task_id"] in va and ok[i]]
        self.by_task = collections.defaultdict(list)
        for i in self.train + self.val:
            self.by_task[self.meta[i]["task_id"]].append(i)
        self.val_cand = [self.cands(i, self.val, random.Random(1234)) for i in self.val]
        self.refs = random.Random(4321).sample(self.train, N_TASK + N_RAND)
        self.test_runs = collections.defaultdict(list)
        for i, m in sorted(enumerate(self.meta), key=lambda x: (x[1]["trajectory_id"], x[1]["step_idx"])):
            if m["task_id"] in set(split["holdout"]):
                self.test_runs[m["trajectory_id"]].append(i)

    def cands(self, i, pool, rng):
        same = [j for j in rng.sample(self.by_task[self.meta[i]["task_id"]], min(N_TASK + 1, len(self.by_task[self.meta[i]["task_id"]]))) if j != i][:N_TASK]
        return [i] + same + [j for j in rng.sample(pool, N_RAND + 1) if j != i][:N_RAND]

    def batch(self, idx, cand):
        W = max(map(len, cand))
        c = torch.tensor([r + [r[0]] * (W - len(r)) for r in cand], device=self.st.device)
        mask = torch.tensor([[k < len(r) for k in range(W)] for r in cand], device=self.st.device)
        return self.st[idx], self.ac[c], mask, torch.zeros(len(idx), dtype=torch.long, device=self.st.device)

    def train_batches(self, rng, bs):
        order = self.train[:]; rng.shuffle(order)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            yield self.batch(idx, [self.cands(i, self.train, rng) for i in idx])

    def val_batches(self, bs=512):
        for s in range(0, len(self.val), bs):
            yield self.batch(self.val[s:s + bs], self.val_cand[s:s + bs])

    def run_scores(self, head):
        out = {}
        for rid, steps in self.test_runs.items():
            last = steps[-WINDOW:]
            hq, hk, mask, _ = self.batch(last, [[i] + self.refs for i in last])
            out[rid] = torch.log_softmax(head.score(hq, hk, mask), -1)[:, 0].mean().item()
        return out


class AgentRows:
    """Mode B: the fixed row spec; read together (our heads) or apart (CLM's head) over the same rows."""

    def __init__(self, split: dict, device: str, together: bool):
        f = torch.load(TOG, weights_only=False, mmap=True)
        rows, meta = f["rows"], json.load(open(os.path.join(EMB, "metadata.json")))["samples"]
        self.meta, self.device = meta, device
        if together:
            self.hq, self.hk = f["hq"], f["hk"]
            self.slot_of = [r["order"] for r in rows]
        else:
            st = torch.load(os.path.join(EMB, "state_embeddings.pt"), mmap=True)
            ac = torch.load(os.path.join(EMB, "action_embeddings.pt"), mmap=True)
            self.hq = st[torch.tensor([r["state"] for r in rows])].float() * 64.0
            self.hk = [ac[torch.tensor([r["cands"][s] for s in r["order"]])].float() * 64.0 for r in rows]
            self.slot_of = [r["order"] for r in rows]
        task = lambda r: meta[r["state"]]["task_id"]
        tr, va = set(split["train_tasks"]), set(split["val_tasks"])
        self.train = [n for n, r in enumerate(rows) if r["split"] == "train" and task(r) in tr]
        self.val = [n for n, r in enumerate(rows) if r["split"] == "train" and task(r) in va]
        self.test_runs = collections.defaultdict(list)
        for n, r in enumerate(rows):
            if r["split"] == "test":
                self.test_runs[meta[r["state"]]["trajectory_id"]].append(n)

    def batch(self, idx):
        hq = self.hq[idx].float().to(self.device)
        hk = torch.stack([self.hk[n].float() for n in idx]).to(self.device)
        mask = torch.ones(hk.shape[:2], dtype=torch.bool, device=self.device)
        slot = torch.tensor([self.slot_of[n].index(0) for n in idx], device=self.device)
        return hq, hk, mask, slot

    def train_batches(self, rng, bs):
        order = self.train[:]; rng.shuffle(order)
        for s in range(0, len(order), bs):
            yield self.batch(order[s:s + bs])

    def val_batches(self, bs=512):
        for s in range(0, len(self.val), bs):
            yield self.batch(self.val[s:s + bs])

    def run_scores(self, head):
        out = {}
        for rid, ns in self.test_runs.items():
            hq, hk, mask, slot = self.batch(ns)
            lp = torch.log_softmax(head.score(hq, hk, mask), -1)
            out[rid] = lp[torch.arange(len(ns)), slot].mean().item()
        return out


def loss_of(head, hq, hk, mask, slot):
    tgt = torch.zeros(mask.shape, device=mask.device)
    tgt[torch.arange(len(slot)), slot] = 1.0
    return soft_ce(head.score(hq, hk, mask), tgt, mask)


def joint(argv=None) -> None:
    """One head trained from scratch on MCQ rows and CoderForge agent steps together; scored on 8 MCQ sets and on
    best-of-8 CoderForge picks.
    
    Mode A: every head reads apart (CLM-style last-token embeddings of the context and of each option).
    Mode B: each head reads its own way: our heads read together (one pass over context and options), CLM's head
    reads apart; the agent rows are the same 10k training steps and 8-candidate test rows for both.
    Loss: softmax over a row's candidates, the true one as target. The epoch with the lowest mean of the MCQ and
    agent validation losses is kept.
    """
    ap = argparse.ArgumentParser(prog="coderforge.py joint", description=joint.__doc__)
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--arch", choices=list(LR), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--no-mcq", action="store_true", help="agent data only")
    ap.add_argument("--no-agent", action="store_true", help="MCQ data only")
    ap.add_argument("--out", default="results/joint")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    apart = a.mode == "A" or a.arch == "clm_mlp"
    suffix = "_apart" if apart else ""
    scale = 64.0 if apart else 1.0   # apart vectors are unit length; together states are raw hidden states
    split = json.load(open(f"results/coderforge/split_s{a.seed}.json"))
    tr = torch.load(f"feats/train{suffix}.pt", weights_only=False, mmap=True)
    mval, mtrain = question_split(tr, "feats/train_groups.json", a.seed)
    agent = AgentApartSteps(split, a.device) if a.mode == "A" else AgentRows(split, a.device, together=not apart)
    use_mcq, use_agent = not a.no_mcq, not a.no_agent
    H = tr["hq"].shape[1]
    head = (HEADS[a.arch](H) if a.arch == "clm_mlp" else HEADS[a.arch](H, 128)).to(a.device)
    n_batches = use_mcq * math.ceil(len(mtrain) / a.batch) + use_agent * math.ceil(len(agent.train) / a.batch)
    opt = torch.optim.AdamW(head.parameters(), lr=LR[a.arch])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR[a.arch], total_steps=a.epochs * n_batches, pct_start=0.05)
    best, best_state, history = float("inf"), None, []
    for ep in range(1, a.epochs + 1):
        m_order = mtrain[:]; rng.shuffle(m_order)
        jobs = ([("m", m_order[s:s + a.batch]) for s in range(0, len(m_order), a.batch)] if use_mcq else []) + \
               ([("a", b) for b in agent.train_batches(rng, a.batch)] if use_agent else [])
        rng.shuffle(jobs)
        for kind, b in jobs:
            loss = loss_of(head, *(mcq_batch(tr, b, a.device, scale) if kind == "m" else b))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        head.eval()
        with torch.no_grad():
            mv = sum(loss_of(head, *mcq_batch(tr, mval[s:s + 512], a.device, scale)).item() * len(mval[s:s + 512])
                     for s in range(0, len(mval), 512)) / len(mval) if use_mcq else 0.0   # the paper's runs logged 0 here
            av = [(loss_of(head, *b).item(), len(b[0])) for b in agent.val_batches()]
            av = sum(l * n for l, n in av) / sum(n for _, n in av)
        head.train()
        crit = (mv if use_mcq else 0) + (av if use_agent else 0)
        history.append({"epoch": ep, "mcq_val": mv, "agent_val": av})
        print(f"[joint] {a.mode} {a.arch} s{a.seed} ep {ep}: mcq val {mv:.4f} agent val {av:.4f}", flush=True)
        if crit < best:
            best, best_state = crit, {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval()

    res = {"mode": a.mode, "arch": a.arch, "seed": a.seed, "reads": "apart" if apart else "together",
           "data": "mcq+agent" if use_mcq and use_agent else ("mcq" if use_mcq else "agent"), "mcq": {}, "history": history}
    with torch.no_grad():
        for s in SETS:
            te = torch.load(f"feats/{s}{suffix}.pt", weights_only=False, mmap=True)
            if scale != 1.0:
                te = {**te, "hq": te["hq"].float() * scale, "hk": [x.float() * scale for x in te["hk"]]}
            chunks = all_logits(head, te, list(range(len(te["gold"]))), a.device)
            res["mcq"][s] = sum((lg.argmax(-1) == sl).sum().item() for lg, sl in chunks) / len(te["gold"])
        scores = agent.run_scores(head)
    run_meta = {m["trajectory_id"]: m for m in agent.meta}
    passed = {rid: float(run_meta[rid]["reward"] or 0) > 0.5 for rid in scores}
    by_task = collections.defaultdict(list)
    for rid in scores:
        by_task[run_meta[rid]["task_id"]].append(rid)
    hits, aucs = 0.0, []
    for rids in by_task.values():
        top = max(scores[r] for r in rids); tied = [r for r in rids if scores[r] == top]
        hits += sum(passed[r] for r in tied) / len(tied)
        pos = [scores[r] for r in rids if passed[r]]; neg = [scores[r] for r in rids if not passed[r]]
        if pos and neg:
            aucs.append(sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)))
    res["agent"] = {"picks": hits, "tasks": len(by_task), "auc": sum(aucs) / len(aucs)}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"{a.mode}_{res['data']}_{a.arch}_s{a.seed}.json"), "w"), indent=1)
    print(f"[joint] {a.mode} {a.arch} s{a.seed}: MCQ " + " ".join(f"{k} {v:.3f}" for k, v in res["mcq"].items())
          + f" | agent {hits:g}/{len(by_task)} auc {res['agent']['auc']:.3f}", flush=True)


# ---------------------------------------------------------------- released_clm
def picks_and_auc(scores: dict, meta: list[dict]) -> dict:
    """Tasks solved by each task's top-scored run (ties share credit) and the mean within-task AUC, as in joint."""
    run_meta = {m["trajectory_id"]: m for m in meta}
    passed = {rid: float(run_meta[rid]["reward"] or 0) > 0.5 for rid in scores}
    by_task = collections.defaultdict(list)
    for rid in scores:
        by_task[run_meta[rid]["task_id"]].append(rid)
    hits, aucs = 0.0, []
    for rids in by_task.values():
        top = max(scores[r] for r in rids); tied = [r for r in rids if scores[r] == top]
        hits += sum(passed[r] for r in tied) / len(tied)
        pos = [scores[r] for r in rids if passed[r]]; neg = [scores[r] for r in rids if not passed[r]]
        if pos and neg:
            aucs.append(sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)))
    return {"picks": hits, "tasks": len(by_task), "auc": sum(aucs) / len(aucs)}


@torch.no_grad()
def released_clm(argv=None) -> None:
    """Released CLM-v0.1-8B (or a fine-tuned CLM head, --ckpt) on the joint mode B test rows: same 100 tasks, 8
    candidates and run score as our heads (mean log-softmax of the own action over a run's last 12 steps), plus CLM's
    own rule (mean state-action cosine over the same steps).
    """
    ap = argparse.ArgumentParser(prog="coderforge.py released_clm", description=released_clm.__doc__)
    ap.add_argument("--ckpt", default=None, help="CLM head checkpoint (default: released CLM-v0.1-8B)")
    ap.add_argument("--out", default="results/released_clm_agent/coderforge.json")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "third_party", "CLM", "evaluation"))
    from bon_eval import load_heads
    repo = "Contrastive-LM/CLM-v0.1-8B"
    ckpt = a.ckpt or hf_hub_download(repo, "CLM_v0.1-8B.pt", revision=revision_for(repo))
    scale = min(float(torch.load(ckpt, map_location="cpu", weights_only=False)["logit_scale"].exp()), 100.0)
    sh, ah, _ = load_heads(ckpt, "cuda")
    rows = AgentRows(json.load(open("results/coderforge/split_s0.json")), "cuda", together=False)   # same test rows for every seed
    ours, own = {}, {}
    for rid, ns in rows.test_runs.items():
        hq, hk, _, slot = rows.batch(ns)
        zs = F.normalize(sh(hq / 64.0), dim=-1)   # AgentRows scales apart vectors by sqrt(4096); CLM reads unit vectors
        za = F.normalize(ah((hk / 64.0).flatten(0, 1)), dim=-1).view(*hk.shape[:2], -1)
        cos = torch.einsum("bd,bkd->bk", zs, za)
        i = torch.arange(len(ns))
        ours[rid] = torch.log_softmax(scale * cos, -1)[i, slot].mean().item()
        own[rid] = cos[i, slot].mean().item()
    res = {"checkpoint": ckpt, "scale": scale, "ours": picks_and_auc(ours, rows.meta),
           "clm_own_cosine": picks_and_auc(own, rows.meta)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


# ---------------------------------------------------------------- adp_overlap
ADP = "neulab/agent-data-collection"
ADP_SUBSETS = ["coderforge_preview", "nebius_SWE-agent-trajectories", "swe-gym_openhands_sampled_trajectories", "swe-smith",
               "swe-play-trajectories", "mini-coder"]
SWE_TASK = re.compile(r"([A-Za-z0-9_.\-]+?__[A-Za-z0-9_.\-]+?-\d+)(?=[_.]|$)")   # owner__repo-N


def adp_ids(subset: str, cache: str) -> list[str]:
    """Row ids of one ADP subset (full_std.jsonl at the pinned revision), streamed once and cached as text."""
    path = os.path.join(cache, f"{subset}.txt")
    if not os.path.exists(path):
        url = f"https://huggingface.co/datasets/{ADP}/resolve/{revision_for(ADP)}/{subset}/full_std.jsonl"
        ids = []
        with urllib.request.urlopen(url) as r:
            for line in r:
                m = re.match(rb'\{"id": ?"([^"]*)"', line)
                if m:
                    ids.append(m.group(1).decode())
        os.makedirs(cache, exist_ok=True)
        open(path + ".tmp", "w").write("".join(x + "\n" for x in ids))
        os.replace(path + ".tmp", path)
    return open(path).read().split()


def adp_overlap(argv=None) -> None:
    """Our CoderForge tasks and runs found in ADP, which CLM post-trains on (it does not say which subsets).
    ADP row ids are <run id>_<n>; runs are matched exactly, tasks by their SWE-rebench id. Streams about 35 GB once.
    """
    ap = argparse.ArgumentParser(prog="coderforge.py adp_overlap", description=adp_overlap.__doc__)
    ap.add_argument("--cache", default="data/adp_ids")
    ap.add_argument("--out", default="results/released_clm_agent/adp_overlap.json")
    a = ap.parse_args(argv)
    tasks = json.load(open(TASKS))
    meta = json.load(open(os.path.join(EMB, "metadata.json")))["samples"]
    run = {m["trajectory_id"]: (m["task_id"], float(m["reward"] or 0) > 0.5) for m in meta}
    out = {}
    for sub in ADP_SUBSETS:
        ids = adp_ids(sub, a.cache)
        runs_in = {re.sub(r"_\d+$", "", x) for x in ids}
        tasks_in = {m.group(1) for m in map(SWE_TASK.match, ids) if m}
        out[sub] = {"rows": len(ids)}
        for split, key in (("test", "holdout"), ("train", "train_pool")):
            ts = set(tasks[key])
            mine = [(rid, ok) for rid, (t, ok) in run.items() if t in ts]
            out[sub][split] = {"tasks": len(ts), "tasks_in_adp": len(ts & tasks_in),
                               "passing_runs": sum(ok for _, ok in mine),
                               "passing_in_adp": sum(ok and rid in runs_in for rid, ok in mine),
                               "failing_runs": sum(not ok for _, ok in mine),
                               "failing_in_adp": sum(not ok and rid in runs_in for rid, ok in mine)}
        print(f"[adp] {sub}: {json.dumps(out[sub])}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"dataset": ADP, "revision": revision_for(ADP), "subsets": out}, open(a.out, "w"), indent=1)


COMMANDS = {"fetch": fetch, "judge": judge, "cf_extract": cf_extract, "evaluate": evaluate, "summary": summary, "embed": embed,
            "together": together, "joint": joint, "released_clm": released_clm, "adp_overlap": adp_overlap}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit("usage: python scripts/coderforge.py {fetch|judge|cf_extract|evaluate|summary|embed|together|joint|"
                 "released_clm|adp_overlap} [args]")
    COMMANDS[sys.argv[1]](sys.argv[2:])
