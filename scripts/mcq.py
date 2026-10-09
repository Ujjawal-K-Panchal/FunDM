"""Multiple-choice steps: extract states, train KD and CE heads, the question split, decontamination.

python scripts/mcq.py <command> [args]; commands: extract, train, predict (FunDM); kd_extract, kd_train (legacy reproduction); train_head, extract_features, pass_cost, embed_apart, decontam,
split_overlap.
"""

import argparse
import collections
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

import data
from data import revision_for
from heads import HEADS, soft_ce, build_row, collate

# ---------------------------------------------------------------- kd_extract
HEAD = "The following is a multiple choice question. Answer with the letter of the correct option.\n\nQuestion: "
SUFFIX = " :: answer in option character only (without brackets):"
LETTERS = {"paper": "ABCDEFGHIJ", "one-line": "abcdefghij"}


def build_paper(tok, question: str, options: list[str], rng: random.Random):
    """-> ids of the whole-text prompt, final position, option-end positions, order (slot -> original option index)."""
    order = list(range(len(options)))
    rng.shuffle(order)
    t = f"{HEAD}{question.strip()}\nOptions:"
    ends = []
    for slot, i in enumerate(order):
        t += f"\n{LETTERS['paper'][slot]}. {options[i]}"
        ends.append(len(t) - 1)  # char index of the option's last character
    enc = tok(t + "\nAnswer:", add_special_tokens=False, return_offsets_mapping=True)
    spans, pos, j = enc["offset_mapping"], [], 0
    for e in ends:
        while spans[j][1] <= e:  # first token whose span contains char e
            j += 1
        pos.append(j)
    return enc["input_ids"], len(enc["input_ids"]) - 1, pos, order


def build_one_line(tok, question: str, options: list[str], rng: random.Random):
    """-> ids, final position, option-end positions, order (slot -> original option index)."""
    order = list(range(len(options)))
    rng.shuffle(order)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    ids = enc(question.rstrip() + " answer options:")
    ends = []
    for slot, i in enumerate(order):
        ids += enc(f" ({LETTERS['one-line'][slot]})") + enc(" " + options[i].strip())
        ends.append(len(ids) - 1)
    ids += enc(SUFFIX) + enc(" (")   # the model wants a bracket first; the letter is the next token
    return ids, len(ids) - 1, ends, order


def kd_extract(argv=None, *, primary=False) -> None:
    """One frozen Qwen3-8B pass per row: the KD student's inputs and the teacher's letter log-probabilities.
    
    --prompt paper (default): "The following is a multiple choice question. Answer with the letter of the correct
    option.
    
    Question: <q>
    Options:
    A. ...
    B. ...
    Answer:"; the teacher reads " A", " B", ... after the ":".
    --prompt one-line (compared on validation rows by teacher_prompt_val.py): "<q> answer options: (a) ... (b) ... ::
    answer in option character only (without brackets): ("; the teacher reads "a", "b", ... after the bracket.
    Saves, per layer L, the state at the end of each option and at the final position: <out>_L<L>.pt
    """
    ap = argparse.ArgumentParser(prog="mcq.py extract" if primary else "mcq.py kd_extract", description=kd_extract.__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--set", choices=["train", "extra", "mmlu_pro", *data.EVAL_SETS])
    source.add_argument("--input", help="JSONL rows with question and options; gold is optional")
    ap.add_argument("--layers", default=str(data.EXIT_LAYER) if primary else "16,20,24,28,32")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--tokens-per-batch", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", choices=["paper", "one-line"], default="paper")
    ap.add_argument("--out", required=True, help="prefix; writes <out>_L<layer>.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    layers = [int(x) for x in a.layers.split(",")]

    if a.input:
        items = read_items(a.input)
    elif a.set == "train":
        items = data.arc() + data.sciq() + data.medmcqa(35000, a.seed)
        data.pad_with_borrowed(items, 0.5, 10, a.seed)
    elif a.set == "extra":
        items = data.extra(a.seed)
    elif a.set == "mmlu_pro":
        items = data.mmlu_pro()
    else:
        items = data.EVAL_SETS[a.set]()
    if a.limit:
        items = items[:a.limit]
    if not items:
        raise ValueError("No input rows to extract")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, revision=revision_for(a.model))
    space = " " if a.prompt == "paper" else ""
    letter_ids = [tok.encode(space + c, add_special_tokens=False) for c in LETTERS[a.prompt]]
    assert all(len(x) == 1 for x in letter_ids), letter_ids
    letter_ids = torch.tensor([x[0] for x in letter_ids])

    rng = random.Random(a.seed)
    build = build_paper if a.prompt == "paper" else build_one_line
    rows = [build(tok, it["question"], it["options"], rng) for it in items]
    keep = [i for i, r in enumerate(rows) if len(r[0]) <= a.max_len]
    print(f"[fundm] {a.set or a.input}: {len(items)} items, {len(items) - len(keep)} over {a.max_len} tokens dropped", flush=True)
    if not keep:
        raise ValueError("All rows exceed --max-len")

    model = AutoModelForCausalLM.from_pretrained(a.model, revision=revision_for(a.model), dtype=torch.bfloat16, device_map=a.device).eval()
    order_by_len = sorted(keep, key=lambda i: len(rows[i][0]))
    states, tlog = {}, {}
    mass, t0, done, tokens, i = [], time.time(), 0, 0, 0
    with torch.no_grad():
        while i < len(order_by_len):
            batch = [order_by_len[i]]
            while i + len(batch) < len(order_by_len) and \
                    len(rows[order_by_len[i + len(batch)]][0]) * (len(batch) + 1) <= a.tokens_per_batch:
                batch.append(order_by_len[i + len(batch)])
            i += len(batch)
            T = max(len(rows[j][0]) for j in batch)
            K = max(len(rows[j][2]) for j in batch)
            ids = torch.full((len(batch), T), tok.pad_token_id)
            attn = torch.zeros((len(batch), T), dtype=torch.long)
            pos = torch.zeros((len(batch), K + 1), dtype=torch.long)
            for b, j in enumerate(batch):
                r = rows[j]
                ids[b, :len(r[0])] = torch.tensor(r[0]); attn[b, :len(r[0])] = 1
                pos[b, :len(r[2])] = torch.tensor(r[2]); pos[b, K] = r[1]
            out = model.model(input_ids=ids.to(a.device), attention_mask=attn.to(a.device), output_hidden_states=True)
            pos = pos.to(a.device)
            bi = torch.arange(len(batch), device=a.device)[:, None]
            h = torch.stack([out.hidden_states[L][bi, pos] for L in layers]).to(torch.bfloat16).cpu()  # (nL, B, K+1, H)
            ans = out.last_hidden_state[torch.arange(len(batch), device=a.device), pos[:, K]]
            lg = model.lm_head(ans).float()
            lp = torch.log_softmax(lg, -1)[:, letter_ids.to(a.device)].cpu()
            for b, j in enumerate(batch):
                n = len(rows[j][2])
                states[j] = torch.cat([h[:, b, :n], h[:, b, K:K + 1]], 1)
                tlog[j] = lp[b, :n]
                mass.append(float(lp[b, :n].exp().sum()))
            done += len(batch); tokens += int(attn.sum())
            if done // 2000 != (done - len(batch)) // 2000 or i >= len(order_by_len):
                el = time.time() - t0
                print(f"[fundm] {done}/{len(order_by_len)} rows, {tokens / el:.0f} tok/s, "
                      f"ETA {el / done * (len(order_by_len) - done) / 60:.1f} min", flush=True)

    meta = {"order": [rows[j][3] for j in keep],
            "category": [items[j]["category"] for j in keep], "id": [items[j]["id"] for j in keep],
            "set": a.set or "custom", "model": a.model, "layers": layers, "letter_mass_mean": sum(mass) / len(mass),
            "prompt": a.prompt, "tl": torch.cat([tlog[j] for j in keep])}
    if any(it.get("gold") is not None for it in items):
        meta["gold"] = [items[j].get("gold") for j in keep]
    meta["groups"] = {str(it["id"]): norm(it["question"]) for it in items}
    off = torch.tensor([0] + [len(rows[j][2]) for j in keep]).cumsum(0)
    for n, L in enumerate(layers):
        hq = torch.stack([states[j][n, -1] for j in keep])
        hk = torch.cat([states[j][n, :-1] for j in keep])
        torch.save({**meta, "hq": hq, "hk": hk, "off": off, "layer": L}, f"{a.out}_L{L}.pt")
    print(f"[fundm] saved {a.out}_L*.pt in {(time.time() - t0) / 60:.1f} min; mean teacher mass on valid letters "
          f"{meta['letter_mass_mean']:.3f}", flush=True)


# ---------------------------------------------------------------- kd_train
def rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class MidHead(torch.nn.Module):
    """Parameter-free RMS norm (intermediate states are un-normalised), then the option head."""

    def __init__(self, arch: str, hidden: int):
        super().__init__()
        self.head = HEADS[arch](hidden, 128)

    def score(self, hq, hk, mask):
        return self.head.score(rms(hq.float()), rms(hk.float()), mask)


def padded(d: dict, device: str, kmax: int = 10, *, include_gold=True, require_gold=False):
    """Pad cached states; return hq, hk, mask, optional teacher logits and optional gold slots."""
    N = len(d["off"]) - 1
    off = d["off"]
    n = off[1:] - off[:-1]
    if N == 0 or (n <= 0).any():
        raise ValueError("Every feature row needs at least one option")
    K = int(n.max())
    row = torch.repeat_interleave(torch.arange(N), n)
    col = torch.arange(len(row)) - off[:-1][row]
    hk = torch.zeros(N, K, d["hk"].shape[1], dtype=torch.bfloat16)
    hk[row, col] = d["hk"]
    tl = None
    if "tl" in d:
        tl = torch.full((N, K), float("-inf"))
        tl[row, col] = d["tl"]
        tl = tl.to(device)
    mask = torch.arange(K)[None] < n[:, None]
    slot = None
    gold = d.get("gold") if include_gold or require_gold else None
    if gold is not None and len(gold) == N and all(g is not None for g in gold):
        slot = torch.tensor([o.index(g) for o, g in zip(d["order"], gold, strict=True)], device=device)
    if require_gold and slot is None:
        raise ValueError("--target gold requires a correct-option label for every training row")
    return d["hq"].to(device), hk.to(device), mask.to(device), tl, slot


def merge(parts: list[dict]) -> dict:
    """Concatenate training caches; optional gold metadata never determines the row count."""
    if len(parts) == 1:
        return parts[0]
    out = {k: sum((p[k] for p in parts), []) for k in ("order", "category", "id")}
    if any("gold" in p for p in parts):
        out["gold"] = sum((p.get("gold", [None] * len(p["hq"])) for p in parts), [])
    for k in ("hq", "hk", "tl"):
        out[k] = torch.cat([p[k] for p in parts])
    n = torch.cat([p["off"][1:] - p["off"][:-1] for p in parts])
    out["off"] = torch.cat([torch.zeros(1, dtype=n.dtype), n.cumsum(0)])
    out["letter_mass_mean"] = sum(p["letter_mass_mean"] * len(p["hq"]) for p in parts) / len(out["hq"])
    out["groups"] = {k: v for p in parts for k, v in p.get("groups", {}).items()}
    for k in ("model", "prompt"):
        if k in parts[0]:
            if any(p.get(k) != parts[0][k] for p in parts):
                raise ValueError(f"Training caches disagree on {k}")
            out[k] = parts[0][k]
    return out


def kl(teacher_logits, student_logits):
    """Mean KL(teacher || student) over rows; both are -inf on padding."""
    pt = torch.softmax(teacher_logits, -1)
    lt = torch.log_softmax(teacher_logits, -1)
    ls = torch.log_softmax(student_logits, -1)
    return (pt * (lt.clamp_min(-1e4) - ls.clamp_min(-1e4))).masked_fill(pt == 0, 0).sum(-1).mean()


@torch.no_grad()
def student_logits(head, X, idx=None, bs=1024):
    hq, hk, mask, _, _ = X
    idx = torch.arange(len(hq), device=hq.device) if idx is None else idx
    head.eval()
    out = torch.cat([head.score(hq[idx[s:s + bs]], hk[idx[s:s + bs]], mask[idx[s:s + bs]]).cpu()
                     for s in range(0, len(idx), bs)])
    head.train()
    return out


def kd_train(argv=None, *, primary=False) -> None:
    """Teacher-student KD: a head on a middle layer learns the frozen model's own letter distribution.
    
    Teacher: the full network's next-token distribution over the option letters (softmax over the row's K letters).
    Student: FunDM (or +1T) on the layer-L states at the option ends and the final position.
    Loss: KL(teacher || student). Gold labels are not used for training or checkpoint choice (validation KL).
    --target ablation: the same loss and checkpoint choice with one-hot targets, the teacher's top letter or the gold option.
    """
    ap = argparse.ArgumentParser(prog="mcq.py train" if primary else "mcq.py kd_train", description=kd_train.__doc__)
    ap.add_argument("--feats", default="feats_kd", help="dir with <set>_L<layer>.pt")
    ap.add_argument("--layers", default=str(data.EXIT_LAYER) if primary else "16,20,24,28,32")
    ap.add_argument("--arch", choices=["linear", "comparator"], default="linear")
    ap.add_argument("--tests", default="" if primary else "mmlu_pro,mmlu,openbookqa,commonsenseqa,piqa,hellaswag,winogrande,boolq",
                    help="optional comma-separated evaluation caches; fitting needs none")
    ap.add_argument("--train-sets", default="train", help="comma list of <set>_L<layer>.pt pools to train on")
    ap.add_argument("--groups", default="" if primary else "feats/train_groups.json",
                    help="optional row-id to question/task map; custom caches carry their own groups")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/fundm" if primary else "results/kd")
    ap.add_argument("--save", help="head checkpoint path; requires a single --layers value")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dump", action="store_true", help="also save per-row logits for each test set")
    ap.add_argument("--target", choices=["teacher", "teacher_hard", "gold"], default="teacher",
                    help="teacher: the full model's letter probabilities; teacher_hard: its top letter; gold: the correct option")
    a = ap.parse_args(argv)
    lr = a.lr or (1e-3 if a.arch == "linear" else 5e-4)
    dev = a.device
    if a.save and len(a.layers.split(",")) != 1:
        ap.error("--save requires a single layer")
    os.makedirs(a.out, exist_ok=True)

    for L in [int(x) for x in a.layers.split(",")]:
        torch.manual_seed(a.seed)
        tr = merge([torch.load(f"{a.feats}/{n}_L{L}.pt", weights_only=False) for n in a.train_sets.split(",")])
        val, train = question_split(tr, a.groups, a.seed)
        print(f"[fundm] L{L}: {len(train)} train rows, {len(val)} validation rows", flush=True)
        model_meta = {k: tr[k] for k in ("model", "prompt") if k in tr}
        X = padded(tr, dev, include_gold=a.target == "gold", require_gold=a.target == "gold")
        if X[3] is None:
            raise ValueError("Teacher probabilities are required for FunDM fitting")
        del tr
        if a.target != "teacher":   # one-hot target logits: 0 on the chosen slot, -inf elsewhere
            pick = X[3].argmax(-1) if a.target == "teacher_hard" else X[4]
            onehot = torch.full_like(X[3], float("-inf"))
            onehot[torch.arange(len(pick), device=dev), pick] = 0.0
            X = (*X[:3], onehot, X[4])
        head = MidHead(a.arch, X[0].shape[1]).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=lr)
        steps = a.epochs * math.ceil(len(train) / a.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
        train_t = torch.tensor(train, device=dev)
        val_t = torch.tensor(val, device=dev)
        best, best_state, hist = float("inf"), None, []
        for ep in range(1, a.epochs + 1):
            perm = train_t[torch.randperm(len(train_t), device=dev)]
            tot = 0.0
            for s in range(0, len(perm), a.batch):
                b = perm[s:s + a.batch]
                loss = kl(X[3][b], head.score(X[0][b], X[1][b], X[2][b]))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                tot += loss.item() * len(b)
            vk = float(kl(X[3][val_t].cpu(), student_logits(head, X, val_t)))
            hist.append({"epoch": ep, "train_kl": tot / len(train), "val_kl": vk})
            print(f"[fundm] {a.arch} L{L} ep {ep} train KL {tot / len(train):.4f} val KL {vk:.4f}", flush=True)
            if vk < best:
                best, best_state = vk, {k: v.detach().clone() for k, v in head.state_dict().items()}
        head.load_state_dict(best_state)
        params = sum(p.numel() for p in head.parameters())
        checkpoint = {"state_dict": {k: v.cpu() for k, v in best_state.items()},
                      "arch": a.arch, "hidden": X[0].shape[1], "layer": L, "seed": a.seed,
                      "target": a.target, "best_val_kl": best, "history": hist,
                      "model_name": {"teacher": "FunDM", "teacher_hard": "FunDM hard-target ablation",
                                     "gold": "FunDM-CE (matched layer-24 ablation)"}[a.target]
                                    + ("+1T" if a.arch == "comparator" else ""),
                      "loss": "L_KL" if a.target == "teacher" else "categorical cross-entropy",
                      **model_meta}
        save_path = a.save or f"{a.out}/{a.arch}_L{L}_s{a.seed}_head.pt"
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"[fundm] saved head: {save_path}", flush=True)
        del X
        for name in filter(None, a.tests.split(",")):
            te = torch.load(f"{a.feats}/{name}_L{L}.pt", weights_only=False)
            Y = padded(te, dev)
            sl = student_logits(head, Y)
            if Y[3] is None:
                raise ValueError("--tests needs teacher logits; use predict for state-only inference")
            tl = Y[3].cpu()
            slot = Y[4].cpu() if Y[4] is not None else None
            if a.dump:
                torch.save({"id": te["id"], "student": sl, "teacher": tl},
                           f"{a.out}/{a.arch}_L{L}_s{a.seed}_{name}_preds.pt")
            res = {"set": name, "layer": L, "arch": a.arch, "params": params, "n": len(sl), "seed": a.seed, "target": a.target,
                   "best_val_kl": best, "student": metrics([(sl, slot)], 1.0) if slot is not None else None,
                   "teacher": metrics([(tl, slot)], 1.0) if slot is not None else None,
                   "agree_with_teacher": float((sl.argmax(-1) == tl.argmax(-1)).float().mean()),
                   "teacher_letter_mass": te["letter_mass_mean"], "history": hist}
            with open(f"{a.out}/{a.arch}_L{L}_s{a.seed}_{name}.json", "w") as stream:
                json.dump(res, stream, indent=1)
            accuracy = (f"student {res['student']['accuracy']:.4f} teacher {res['teacher']['accuracy']:.4f}"
                        if slot is not None else "accuracy unavailable (no gold labels)")
            print(f"[fundm] {a.arch} L{L} {name}: {accuracy} agree {res['agree_with_teacher']:.3f}", flush=True)
            del te, Y


# ---------------------------------------------------------------- train_head
ECE_BINS = 15


def batches(f: dict, idx: list[int], device: str):
    """-> hq (B, H), hk (B, K, H), key mask, one-hot targets in row order, gold slot."""
    K = max(len(f["order"][i]) for i in idx)
    H = f["hq"].shape[1]
    hk = torch.zeros(len(idx), K, H, dtype=torch.bfloat16)
    mask = torch.zeros(len(idx), K, dtype=torch.bool)
    slot = torch.tensor([f["order"][i].index(f["gold"][i]) for i in idx])
    for b, i in enumerate(idx):
        n = len(f["order"][i])
        hk[b, :n] = f["hk"][i]
        mask[b, :n] = True
    tgt = torch.zeros(len(idx), K)
    tgt[torch.arange(len(idx)), slot] = 1.0
    return f["hq"][idx].to(device), hk.to(device), mask.to(device), tgt.to(device), slot.to(device)


@torch.no_grad()
def all_logits(head, f, idx, device, bs=512):
    was = head.training
    head.eval()
    out = []
    for s in range(0, len(idx), bs):
        hq, hk, mask, _, slot = batches(f, idx[s:s + bs], device)
        out.append((head.score(hq, hk, mask).cpu(), slot.cpu()))
    head.train(was)
    return out


def fit_temperature(chunks) -> float:
    """One scalar T minimising held-out NLL (Guo et al. 2017)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad()
        # padding slots are -inf, whose gradient w.r.t. T is NaN; a large finite value gives the same softmax
        loss = sum(torch.nn.functional.cross_entropy(lg.clamp_min(-1e4) / log_t.exp(), sl, reduction="sum")
                   for lg, sl in chunks)
        loss.backward()
        return loss
    opt.step(closure)
    return log_t.exp().item()


def question_split(f: dict, groups_path: str, seed: int) -> tuple[list[int], list[int]]:
    """-> validation rows (about 5%), training rows; a repeated question never sits on both sides."""
    group = json.load(open(groups_path)) if groups_path else f.get("groups", {})
    rows = collections.defaultdict(list)
    for i, rid in enumerate(f["id"]):
        rows[group.get(str(rid), str(rid))].append(i)
    qs = sorted(rows)
    random.Random(seed).shuffle(qs)
    val = []
    for q in qs:
        if len(val) >= max(1, len(f["id"]) // 20):
            break
        val += rows[q]
    held = set(val)
    train = [i for i in range(len(f["id"])) if i not in held]
    if not train or not val:
        raise ValueError("Fitting needs at least two question/task groups")
    return val, train


def metrics(chunks, T: float, cats=None) -> dict:
    conf, correct, nll = [], [], []
    for lg, sl in chunks:
        p = torch.softmax(lg / T, -1)
        conf += p.max(-1).values.tolist()
        correct += (p.argmax(-1) == sl).float().tolist()
        nll += (-torch.log(p[torch.arange(len(sl)), sl].clamp_min(1e-12))).tolist()
    conf, correct = np.array(conf), np.array(correct)
    edges = np.linspace(0, 1, ECE_BINS + 1)
    ece = sum(((conf > lo) & (conf <= hi)).mean() * abs(conf[(conf > lo) & (conf <= hi)].mean() - correct[(conf > lo) & (conf <= hi)].mean())
              for lo, hi in zip(edges[:-1], edges[1:]) if ((conf > lo) & (conf <= hi)).any())
    out = {"accuracy": float(correct.mean()), "nll": float(np.mean(nll)), "ece": float(ece),
           "mean_confidence": float(conf.mean())}
    if cats is not None:
        per = collections.defaultdict(list)
        for c, ok in zip(cats, correct):
            per[c].append(ok)
        out["per_category"] = {c: {"accuracy": float(np.mean(v)), "n": len(v)} for c, v in sorted(per.items())}
    return out


def train_head(argv=None) -> None:
    """Train the option head on stored Qwen3-8B states; evaluate once on MMLU-Pro.
    
    Model selection uses about 5% of the training questions, held out by question; MMLU-Pro test is scored only at the end,
    with the same metrics as scripts/baselines.py eval_mmlu_pro, before and after temperature scaling.
    """
    ap = argparse.ArgumentParser(prog="mcq.py train_head", description=train_head.__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", nargs="+", required=True, help="one or more feature files, each scored once")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--arch", choices=list(HEADS), default="linear")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", default="feats/train_groups.json", help="row id -> question, from mcq.py split_overlap")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="results", help="writes head_<set>.json and _preds.jsonl per test set")
    ap.add_argument("--save", default="feats/head_linear.pt", help="trained head weights and temperature")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)

    tr = torch.load(a.train, weights_only=False)
    val, train = question_split(tr, a.groups, a.seed)
    H = tr["hq"].shape[1]
    head = (HEADS[a.arch](H) if a.arch == "clm_mlp" else HEADS[a.arch](H, a.dim)).to(a.device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr)
    steps = a.epochs * math.ceil(len(train) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    best, best_state, history = float("inf"), None, []
    for ep in range(1, a.epochs + 1):
        random.Random(a.seed + ep).shuffle(train)
        tot = 0.0
        for s in range(0, len(train), a.batch):
            hq, hk, mask, tgt, _ = batches(tr, train[s:s + a.batch], a.device)
            loss = soft_ce(head.score(hq, hk, mask), tgt, mask)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(hq)
        vm = metrics(all_logits(head, tr, val, a.device), 1.0)
        history.append({"epoch": ep, "train_loss": tot / len(train), "val": vm})
        print(f"[head] epoch {ep} loss {tot / len(train):.4f} val acc {vm['accuracy']:.4f} nll {vm['nll']:.4f}", flush=True)
        if vm["nll"] < best:
            best, best_state = vm["nll"], {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    T = fit_temperature(all_logits(head, tr, val, a.device))
    os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
    torch.save({"state": head.state_dict(), "temperature": T, "args": vars(a)}, a.save)
    for path in a.test:
        evaluate(head, torch.load(path, weights_only=False), T, len(train), len(val), history, a)


def evaluate(head, te: dict, T: float, n_train: int, n_val: int, history: list, a) -> None:
    """Score one test set once, raw and temperature-scaled; write results and per-question probabilities."""
    test_idx = list(range(len(te["gold"])))
    chunks = all_logits(head, te, test_idx, a.device)
    res = {"dataset": te["set"], "n_questions": len(test_idx), "train_rows": n_train, "val_rows": n_val,
           "head_params": sum(p.numel() for p in head.parameters()), "temperature": T,
           "head": metrics(chunks, 1.0, te["category"]), "head_ts": metrics(chunks, T, te["category"]),
           "history": history, "args": vars(a)}
    out = os.path.join(a.out_dir, f"head_{te['set']}.json")
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    with open(out.replace(".json", "_preds.jsonl"), "w") as fh:
        i = 0
        for lg, sl in chunks:
            for row, s_ in zip(torch.softmax(lg, -1), sl):
                order = te["order"][test_idx[i]]
                p = [0.0] * len(order)
                for slot, orig in enumerate(order):
                    p[orig] = round(float(row[slot]), 6)
                fh.write(json.dumps({"question_id": te["id"][test_idx[i]], "gold": te["gold"][test_idx[i]], "p_head": p}) + "\n")
                i += 1
    print(f"[head] {te['set']} acc {res['head']['accuracy']:.4f} ece {res['head']['ece']:.4f} | "
          f"after T={T:.2f}: ece {res['head_ts']['ece']:.4f}", flush=True)


# ---------------------------------------------------------------- extract_features
def extract_features(argv=None) -> None:
    """Run frozen Qwen3-8B once per row; keep the hidden states the option head reads.
    
    Saves, per row: the answer-token state and one state per option separator (bf16), the
    option order, the gold index and metadata. The head then trains on these in minutes.
    """
    ap = argparse.ArgumentParser(prog="mcq.py extract_features", description=extract_features.__doc__)
    ap.add_argument("--set", choices=["train", "mmlu_pro", *data.EVAL_SETS], required=True)
    ap.add_argument("--medmcqa", type=int, default=35000)
    ap.add_argument("--pad-share", type=float, default=0.5)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--tokens-per-batch", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["joint", "question_option"], default="joint",
                    help="joint: one row holds every option; question_option: each option gets its own row after the "
                         "question, and the query comes from a question-only row")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.set == "train":
        items = data.arc() + data.sciq() + data.medmcqa(a.medmcqa, a.seed)
        data.pad_with_borrowed(items, a.pad_share, 10, a.seed)
    elif a.set == "mmlu_pro":
        items = data.mmlu_pro()
    else:
        items = data.EVAL_SETS[a.set]()
    if a.limit:
        items = items[:a.limit]
    tok = AutoTokenizer.from_pretrained(a.model, revision=revision_for(a.model))
    rng = random.Random(a.seed)
    rows = [build_row(tok, it["question"], it["options"], rng) for it in items]
    keep = [i for i, r in enumerate(rows) if len(r.ids) <= a.max_len]
    print(f"[features] {a.set}: {len(items)} items, {len(items) - len(keep)} over {a.max_len} tokens dropped", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model, revision=revision_for(a.model), dtype=torch.bfloat16, device_map="cuda").eval()
    backbone = model.model   # Qwen3Model: last_hidden_state is after the final norm
    if a.mode != "joint":
        per_option_features(a, tok, backbone, items, rows, keep)
        return
    order = sorted(keep, key=lambda i: len(rows[i].ids))
    hq, hk, t0, done, tokens = {}, {}, time.time(), 0, 0
    i = 0
    with torch.no_grad():
        while i < len(order):
            batch = [order[i]]
            while i + len(batch) < len(order) and len(rows[order[i + len(batch)]].ids) * (len(batch) + 1) <= a.tokens_per_batch:
                batch.append(order[i + len(batch)])
            i += len(batch)
            ids, attn, ans, sep, mask = collate([rows[j] for j in batch], tok.pad_token_id, "cuda")
            h = backbone(input_ids=ids, attention_mask=attn).last_hidden_state
            b = torch.arange(len(batch), device="cuda")
            q_states = h[b, ans].to(torch.bfloat16).cpu()
            k_states = h[b[:, None], sep].to(torch.bfloat16).cpu()
            for n, j in enumerate(batch):
                hq[j] = q_states[n]
                hk[j] = k_states[n, :len(rows[j].sep_pos)]
            done += len(batch); tokens += int(attn.sum())
            if done // 2000 != (done - len(batch)) // 2000 or i >= len(order):
                el = time.time() - t0
                print(f"[features] {done}/{len(order)} rows, {tokens / el:.0f} tokens/s, "
                      f"ETA {el / done * (len(order) - done) / 60:.1f} min", flush=True)
    torch.save({"hq": torch.stack([hq[j] for j in keep]), "hk": [hk[j] for j in keep],
                "order": [rows[j].order for j in keep], "gold": [items[j]["gold"] for j in keep],
                "category": [items[j]["category"] for j in keep], "id": [items[j]["id"] for j in keep],
                "model": a.model, "set": a.set}, a.out)
    print(f"[features] saved {a.out} in {(time.time() - t0) / 60:.1f} min", flush=True)


def per_option_features(a, tok, backbone, items: list[dict], rows: list, keep: list[int]) -> None:
    """Joint-format features read one option per row: same template, final-layer states and readout positions.

    Each option's state is its closing separator in a row holding the question and only that option. The query is
    the answer state of a question-only row. Options keep the joint rows' order, so slots and the validation split
    match the joint features.
    """
    units, slots = {}, []   # (ids, position) -> unit index; per kept item: (query unit, option units in row order)
    def unit(ids: list[int], pos: int) -> int:
        return units.setdefault((tuple(ids), pos), len(units))
    for j in keep:
        q = build_row(tok, items[j]["question"], [])
        keys = []
        for o in rows[j].order:
            r = build_row(tok, items[j]["question"], [items[j]["options"][o]])
            keys.append(unit(r.ids, r.sep_pos[0]))
        slots.append((unit(q.ids, q.answer_pos), keys))
    todo = sorted(units, key=lambda u: len(u[0]))
    print(f"[features] {a.set} {a.mode}: {len(keep)} items, {len(todo)} distinct rows, "
          f"{sum(len(u[0]) for u in todo)} tokens", flush=True)
    state, t0, done, tokens, i = {}, time.time(), 0, 0, 0
    with torch.no_grad():
        while i < len(todo):
            batch = [todo[i]]
            while i + len(batch) < len(todo) and len(todo[i + len(batch)][0]) * (len(batch) + 1) <= a.tokens_per_batch:
                batch.append(todo[i + len(batch)])
            i += len(batch)
            T = max(len(u[0]) for u in batch)
            ids = torch.full((len(batch), T), tok.pad_token_id)
            attn = torch.zeros((len(batch), T), dtype=torch.long)
            for b, u in enumerate(batch):
                ids[b, :len(u[0])] = torch.tensor(u[0]); attn[b, :len(u[0])] = 1
            h = backbone(input_ids=ids.cuda(), attention_mask=attn.cuda()).last_hidden_state
            pos = torch.tensor([u[1] for u in batch], device="cuda")
            st = h[torch.arange(len(batch), device="cuda"), pos].to(torch.bfloat16).cpu()
            for b, u in enumerate(batch):
                state[units[u]] = st[b]
            done += len(batch); tokens += int(attn.sum())
            if done // 20000 != (done - len(batch)) // 20000 or i >= len(todo):
                el = time.time() - t0
                print(f"[features] {done}/{len(todo)} rows, {tokens / el:.0f} tokens/s, "
                      f"ETA {el / done * (len(todo) - done) / 60:.1f} min", flush=True)
    torch.save({"hq": torch.stack([state[q] for q, _ in slots]), "hk": [torch.stack([state[k] for k in ks]) for _, ks in slots],
                "order": [rows[j].order for j in keep], "gold": [items[j]["gold"] for j in keep],
                "category": [items[j]["category"] for j in keep], "id": [items[j]["id"] for j in keep],
                "model": a.model, "set": f"{a.set}_{a.mode}", "mode": a.mode}, a.out)
    print(f"[features] saved {a.out} in {(time.time() - t0) / 60:.1f} min", flush=True)


def pass_cost(argv=None) -> None:
    """Forward passes and prompt tokens per question: one full-list row versus question-plus-option rows (CPU only)."""
    ap = argparse.ArgumentParser(prog="mcq.py pass_cost", description=pass_cost.__doc__)
    ap.add_argument("--set", choices=["mmlu_pro", *data.EVAL_SETS], default="mmlu_pro")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    items = data.mmlu_pro() if a.set == "mmlu_pro" else data.EVAL_SETS[a.set]()
    tok = AutoTokenizer.from_pretrained(a.model, revision=revision_for(a.model))
    n = len(items)
    joint = sum(len(build_row(tok, it["question"], it["options"]).ids) for it in items) / n
    per = sum(len(build_row(tok, it["question"], []).ids) + sum(len(build_row(tok, it["question"], [o]).ids) for o in it["options"])
              for it in items) / n
    res = {"set": a.set, "questions": n, "options": sum(len(it["options"]) for it in items) / n,
           "full_list": {"passes": 1, "tokens": joint}, "question_option": {"passes": 1 + sum(len(it["options"]) for it in items) / n,
                                                                            "tokens": per}}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[cost] {a.set}: full list 1 pass, {joint:.0f} tokens; question + option {res['question_option']['passes']:.2f} "
          f"passes, {per:.0f} tokens per question", flush=True)


# ---------------------------------------------------------------- embed_apart
def embed_apart(argv=None) -> None:
    """CLM-style features: the question and each option embedded in separate passes (vLLM pooling, last token).
    
    Same rows and row order as mcq.py extract_features, so mcq.py train_head gets the same validation split. The question
    goes through CLM's state_text and options are verbatim, as CLM's Engine sends them. Output format matches
    mcq.py extract_features; "order" is the identity.
    """
    from clm.embedder import Embedder
    from clm.schema import state_text
    ap = argparse.ArgumentParser(prog="mcq.py embed_apart", description=embed_apart.__doc__)
    ap.add_argument("--set", choices=["train", "mmlu_pro", *data.EVAL_SETS], required=True)
    ap.add_argument("--medmcqa", type=int, default=35000)
    ap.add_argument("--pad-share", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-url", default="http://127.0.0.1:8091/v1/embeddings")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.set == "train":
        items = data.arc() + data.sciq() + data.medmcqa(a.medmcqa, a.seed)
        data.pad_with_borrowed(items, a.pad_share, 10, a.seed)
    elif a.set == "mmlu_pro":
        items = data.mmlu_pro()
    else:
        items = data.EVAL_SETS[a.set]()
    states = [state_text(it["question"], None) for it in items]
    texts = list(dict.fromkeys(states + [o for it in items for o in it["options"]]))
    print(f"[apart] {a.set}: {len(items)} rows, {len(texts)} distinct texts", flush=True)
    emb = Embedder(url=a.emb_url, model="qwen3-8b", batch=64, cache_size=len(texts) + 1)
    vec, t0 = {}, time.time()
    for i in range(0, len(texts), 8192):
        part = texts[i:i + 8192]
        v, _ = emb.embed(part)
        vec.update((t, torch.from_numpy(x).to(torch.bfloat16)) for t, x in zip(part, v))
        print(f"[apart] {min(i + 8192, len(texts))}/{len(texts)} texts, {(time.time() - t0) / 60:.1f} min", flush=True)
    torch.save({"hq": torch.stack([vec[s] for s in states]),
                "hk": [torch.stack([vec[o] for o in it["options"]]) for it in items],
                "order": [list(range(len(it["options"]))) for it in items], "gold": [it["gold"] for it in items],
                "category": [it["category"] for it in items], "id": [it["id"] for it in items],
                "model": "Qwen/Qwen3-8B (vLLM pooling, last token, L2-normalized)", "set": f"{a.set}_apart"}, a.out)
    print(f"[apart] saved {a.out}", flush=True)


# ---------------------------------------------------------------- decontam
N = 13


def words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).split()


def shingles(s: str) -> set[str]:
    w = words(s)
    return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)} if len(w) >= N else {" ".join(w)}


def decontam(argv=None) -> None:
    """Drop extra-pool rows whose question shares a 13-word span with any eval-set question (MMLU-Pro, MMLU, the 7 sets).
    """
    test = {"mmlu_pro": data.mmlu_pro(), **{k: f() for k, f in data.EVAL_SETS.items()}}
    index: dict[str, str] = {}
    for name, items in test.items():
        for it in items:
            for sh in shingles(it["question"]):
                index.setdefault(sh, name)
    pool = data.sources(10 ** 9, 0)
    bad, by_pair = [], Counter()
    for it in pool:
        hit = next((index[sh] for sh in shingles(it["question"]) if sh in index), None)
        if hit:
            bad.append(it["id"]); by_pair[(it["category"], hit)] += 1
    os.makedirs("feats", exist_ok=True)
    json.dump(bad, open(data.EXCLUDED, "w"))
    print(f"[decontam] pool {len(pool)} rows, flagged {len(bad)} ({len(bad) / len(pool):.4f})", flush=True)
    for (src, tgt), n in by_pair.most_common():
        print(f"[decontam]   {src} -> {tgt}: {n}", flush=True)
    print("[decontam] per source:", dict(Counter(it["category"] for it in pool)), flush=True)


# ---------------------------------------------------------------- split_overlap
GROUPS = "feats/train_groups.json"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def crossing(key: list[str], val: set[int]) -> dict:
    groups = defaultdict(list)
    for i, k in enumerate(key):
        groups[k].append(i)
    dup = [g for g in groups.values() if len(g) > 1]
    cross = [g for g in dup if any(i in val for i in g) and any(i not in val for i in g)]
    return {"unique": len(groups), "duplicate_groups": len(dup), "groups_crossing_split": len(cross),
            "val_rows": len(val), "val_rows_with_twin_in_train": sum(i in val for g in cross for i in g)}


def split_overlap(argv=None) -> None:
    """Writes row id -> question for the question-level split, and counts duplicates crossing the old and new splits.
    """
    ap = argparse.ArgumentParser(prog="mcq.py split_overlap", description=split_overlap.__doc__)
    ap.add_argument("--groups-only", action="store_true", help="write only the groups file (no feats/train.pt needed)")
    a = ap.parse_args(argv)
    items = data.arc() + data.sciq() + data.medmcqa(35000, 0)   # same order as mcq.py extract_features
    if a.groups_only:
        os.makedirs(os.path.dirname(GROUPS), exist_ok=True)
        json.dump({str(it["id"]): norm(it["question"]) for it in items}, open(GROUPS, "w"))
        return
    feats = torch.load("feats/train.pt", weights_only=False, mmap=True)
    assert [it["id"] for it in items] == list(feats["id"]), "feature rows do not line up with the rebuilt items"
    assert len(set(feats["id"])) == len(items), "row ids are not unique"
    json.dump({str(it["id"]): norm(it["question"]) for it in items}, open(GROUPS, "w"))
    keys = {"question": [norm(it["question"]) for it in items],
            "question+answer": [norm(it["question"]) + " || " + norm(it["options"][it["gold"]]) for it in items]}
    out = {"rows": len(items)}
    for seed in (0, 1, 2):
        order = list(range(len(items)))
        random.Random(seed).shuffle(order)             # the old row split
        row_val = set(order[:max(1, len(order) // 20)])
        q_val = set(question_split(feats, GROUPS, seed)[0])
        for name, key in keys.items():
            out[f"seed{seed}/row_split/{name}"] = crossing(key, row_val)
            out[f"seed{seed}/question_split/{name}"] = crossing(key, q_val)
    print(json.dumps(out, indent=1))
    json.dump(out, open("results/split_overlap.json", "w"), indent=1)


def read_items(path):
    """Read candidate rows without requiring correct-answer labels."""
    items = []
    with open(path) as stream:
        for i, line in enumerate(stream):
            if not line.strip():
                continue
            it = json.loads(line)
            if not isinstance(it.get("question"), str) or not isinstance(it.get("options"), list):
                raise ValueError(f"Row {i + 1} needs question (text) and options (list)")
            if not 2 <= len(it["options"]) <= 10 or not all(isinstance(x, str) and x.strip() for x in it["options"]):
                raise ValueError(f"Row {i + 1} needs 2--10 nonempty option strings")
            if it.get("gold") is not None and (type(it["gold"]) is not int or not 0 <= it["gold"] < len(it["options"])):
                raise ValueError(f"Row {i + 1}: gold must be an option index")
            items.append({**it, "id": str(it.get("id", i)), "category": it.get("category", "custom")})
    if len({it["id"] for it in items}) != len(items):
        raise ValueError("Input row ids must be unique")
    return items


@torch.no_grad()
def predict(argv=None):
    """FunDM inference from raw questions/options or cached states. No teacher probabilities or gold labels needed."""
    ap = argparse.ArgumentParser(prog="mcq.py predict", description=predict.__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="JSONL questions and options; uses only layers 1 through the checkpoint depth")
    source.add_argument("--feats", help="cached layer states; teacher probabilities and gold are optional")
    ap.add_argument("--head", required=True, help="checkpoint saved by train")
    ap.add_argument("--out", required=True, help="output JSONL with probabilities and selected option index")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tokens-per-batch", type=int, default=16384)
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args(argv)
    ck = torch.load(a.head, weights_only=False, map_location="cpu")
    head = MidHead(ck["arch"], ck["hidden"]).to(a.device).eval()
    head.load_state_dict(ck["state_dict"])
    if a.feats:
        features = torch.load(a.feats, weights_only=False, map_location="cpu")
        if features.get("layer") != ck["layer"]:
            raise ValueError("Checkpoint and feature layers differ")
        for k in ("model", "prompt"):
            if k in ck and k in features and ck[k] != features[k]:
                raise ValueError(f"Checkpoint and features disagree on {k}")
        X = padded(features, a.device, include_gold=False)
        logits = student_logits(head, X)
        ids, orders = features["id"], features["order"]
        probabilities = [p[:len(o)].tolist() for p, o in zip(torch.softmax(logits, -1), orders, strict=True)]
    else:
        if ck.get("prompt") != "paper":
            raise ValueError("Raw inference needs main-paper prompt metadata; use --feats for agent or other prompts")
        items = read_items(a.input)
        if not items:
            raise ValueError("No input rows")
        model_name = ck.get("model", "Qwen/Qwen3-8B")
        tok = AutoTokenizer.from_pretrained(model_name, revision=revision_for(model_name))
        rows = [build_paper(tok, it["question"], it["options"], random.Random(i)) for i, it in enumerate(items)]
        if any(len(row[0]) > a.max_len for row in rows):
            raise ValueError("An input exceeds --max-len; shorten it before inference")
        model = AutoModel.from_pretrained(model_name, revision=revision_for(model_name),
                                         dtype=torch.bfloat16, device_map=a.device).eval()
        if ck["layer"] >= len(model.layers):
            raise ValueError("Raw inference requires an intermediate-layer checkpoint")
        model.layers = torch.nn.ModuleList(list(model.layers[:ck["layer"]]))
        model.norm = torch.nn.Identity()  # training reads the unnormalized intermediate layer output
        probabilities = []
        i = 0
        while i < len(rows):
            batch = [rows[i]]
            while i + len(batch) < len(rows) and max(len(r[0]) for r in [*batch, rows[i + len(batch)]]) * (len(batch) + 1) <= a.tokens_per_batch:
                batch.append(rows[i + len(batch)])
            i += len(batch)
            T, K = max(len(r[0]) for r in batch), max(len(r[2]) for r in batch)
            tokens = torch.full((len(batch), T), tok.pad_token_id, device=a.device)
            attention = torch.zeros_like(tokens)
            positions = torch.zeros((len(batch), K), dtype=torch.long, device=a.device)
            mask = torch.zeros_like(positions, dtype=torch.bool)
            answer = torch.tensor([r[1] for r in batch], device=a.device)
            for j, row in enumerate(batch):
                tokens[j, :len(row[0])] = torch.tensor(row[0], device=a.device)
                attention[j, :len(row[0])] = 1
                positions[j, :len(row[2])] = torch.tensor(row[2], device=a.device)
                mask[j, :len(row[2])] = True
            h = model(input_ids=tokens, attention_mask=attention, use_cache=False).last_hidden_state
            bi = torch.arange(len(batch), device=a.device)
            logits = head.score(h[bi, answer], h[bi[:, None], positions], mask)
            for row, p in zip(batch, torch.softmax(logits, -1).cpu(), strict=True):
                probabilities.append(p[:len(row[2])].tolist())
        ids, orders = [it["id"] for it in items], [r[3] for r in rows]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as stream:
        for rid, order, p in zip(ids, orders, probabilities, strict=True):
            original = [0.0] * len(order)
            for slot, option in enumerate(order):
                original[option] = p[slot]
            stream.write(json.dumps({"id": rid, "probabilities": original,
                                     "choice": max(range(len(original)), key=original.__getitem__)}) + "\n")
    print(f"[fundm] wrote {len(ids)} predictions to {a.out}")


COMMANDS = {"extract": lambda argv: kd_extract(argv, primary=True), "train": lambda argv: kd_train(argv, primary=True), "predict": predict, "kd_extract": kd_extract, "kd_train": kd_train, "train_head": train_head, "extract_features": extract_features, "pass_cost": pass_cost, "embed_apart": embed_apart, "decontam": decontam, "split_overlap": split_overlap}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit("usage: python scripts/mcq.py {extract|train|predict|kd_extract|kd_train|train_head|extract_features|pass_cost|embed_apart|decontam|split_overlap} [args]")
    COMMANDS[sys.argv[1]](sys.argv[2:])
