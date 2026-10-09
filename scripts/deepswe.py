"""Best-of-4 on CLM's released DeepSWE embeddings, trained on steps as CLM's head is.

Training: each step of a passing Opus 5 run on CLM's 59 training tasks is a state and its true action; the head
picks the true action among 32 candidates (15 from the same task, 16 at random). The 7 validation tasks select
the epoch. This is exactly CLM's 22,576-step fine-tuning set. Test: a run's score is the mean, over its final 12
steps, of the log-probability of its own action against 31 fixed reference actions; the best of 4 runs is picked.
"""
import argparse
import collections
import json
import math
import os
import random

import torch

from heads import HEADS, soft_ce  # noqa: E402


EMB_DIR = "data/clm_emb/tarsur385_deepswe-prm-embeddings-8k"
POLICY = "mini_swe_agent_claude_opus_5_max"
WINDOW, N_TASK, N_RAND = 12, 15, 16   # CLM's final-step window; same-task and random negatives per state


def load(device: str):
    """-> metadata rows, state and action embeddings (N, H) on device, scaled to about unit size per coordinate."""
    meta = json.load(open(os.path.join(EMB_DIR, "metadata.json")))["samples"]
    st = torch.load(os.path.join(EMB_DIR, "state_embeddings.pt"), mmap=True, weights_only=True)
    ac = torch.load(os.path.join(EMB_DIR, "action_embeddings.pt"), mmap=True, weights_only=True)
    scale = st.shape[1] ** 0.5   # the released vectors are unit length
    return meta, st.to(device).float() * scale, ac.to(device).float() * scale


def candidates(idx: list[int], task_of: list[str], pool: list[int], by_task: dict, rng: random.Random) -> torch.Tensor:
    """-> (B, 1 + N_TASK + N_RAND) action rows: the true action first, then same-task and random negatives."""
    rows = []
    for i in idx:
        same = [j for j in rng.sample(by_task[task_of[i]], min(N_TASK + 1, len(by_task[task_of[i]]))) if j != i][:N_TASK]
        rand = [j for j in rng.sample(pool, N_RAND + 1) if j != i][:N_RAND]
        rows.append([i] + same + rand)
    width = max(map(len, rows))
    return torch.tensor([r + [-1] * (width - len(r)) for r in rows])


def step_loss(head, st, ac, cand: torch.Tensor, idx: list[int]) -> torch.Tensor:
    mask = (cand >= 0).to(st.device)
    hk = ac[cand.clamp_min(0).to(st.device)]
    tgt = torch.zeros(mask.shape, device=st.device)
    tgt[:, 0] = 1.0
    return soft_ce(head.score(st[idx], hk, mask), tgt, mask)


@torch.no_grad()
def run_scores(head, st, ac, runs: dict[str, list[int]], refs: list[int]) -> dict[str, float]:
    """Mean over each run's final WINDOW steps of log p(own action | state, own action + fixed references)."""
    head.eval()
    out = {}
    for rid, steps in runs.items():
        last = steps[-WINDOW:]
        cand = torch.tensor([[i] + refs for i in last], device=st.device)
        mask = torch.ones(cand.shape, dtype=torch.bool, device=st.device)
        lp = torch.log_softmax(head.score(st[last], ac[cand], mask), -1)[:, 0]
        out[rid] = lp.mean().item()
    head.train()
    return out


def best_of(scores: dict[str, float], runs_of_task: dict[str, list[str]], passed: dict[str, bool]) -> tuple[float, dict]:
    """Expected success of picking the top run per task; exact ties count as the mean outcome of the tied runs."""
    hits, picks = 0.0, {}
    for t, rids in sorted(runs_of_task.items()):
        top = max(scores[r] for r in rids)
        tied = [r for r in rids if scores[r] == top]
        hits += sum(passed[r] for r in tied) / len(tied)
        picks[t] = tied
    return hits, picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(HEADS), default="comparator")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="heads/deepswe/task_split.json")
    ap.add_argument("--out", default="results/deepswe_steps")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default=None, help="where to torch.save the selected head module")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    split = json.load(open(a.split))
    train_t, val_t, held_t = set(split["train_tasks"]), set(split["val_tasks"]), set(split.get("holdout", []))
    meta, st, ac = load(a.device)
    task_of = [m["task_id"] for m in meta]
    ok = [float(m["reward"] or 0) > 0.5 for m in meta]
    train = [i for i, m in enumerate(meta) if m["task_id"] in train_t and ok[i]]
    val = [i for i, m in enumerate(meta) if m["task_id"] in val_t and ok[i]]
    by_task = collections.defaultdict(list)
    for i in train + val:
        by_task[task_of[i]].append(i)
    val_cand = candidates(val, task_of, val, by_task, random.Random(1234))
    refs = random.Random(4321).sample(train, 1 + N_TASK + N_RAND - 1)   # fixed reference actions for scoring

    runs, passed = collections.defaultdict(list), {}
    for i, m in sorted(enumerate(meta), key=lambda x: (x[1]["trajectory_id"], x[1]["step_idx"])):
        if m["task_id"] in held_t and m.get("config") == POLICY:
            runs[m["trajectory_id"]].append(i)
            passed[m["trajectory_id"]] = ok[i]
    runs_of_task = collections.defaultdict(list)
    for rid, steps in runs.items():
        runs_of_task[task_of[steps[0]]].append(rid)

    H = st.shape[1]
    head = (HEADS[a.arch](H) if a.arch == "clm_mlp" else HEADS[a.arch](H, 128)).to(a.device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr)
    steps = a.epochs * math.ceil(len(train) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    best, best_state, history = float("inf"), None, []
    for ep in range(1, a.epochs + 1):
        order = train[:]
        rng.shuffle(order)
        tot = 0.0
        for s in range(0, len(order), a.batch):
            idx = order[s:s + a.batch]
            loss = step_loss(head, st, ac, candidates(idx, task_of, train, by_task, rng), idx)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        head.eval()
        with torch.no_grad():
            vl = sum(step_loss(head, st, ac, val_cand[s:s + 512], val[s:s + 512]).item() * len(val[s:s + 512])
                     for s in range(0, len(val), 512)) / len(val)
        head.train()
        history.append({"epoch": ep, "train_loss": tot / len(train), "val_loss": vl})
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    if a.save:
        torch.save(head, a.save)
    if not held_t:   # validation-only split: no held-out scoring
        os.makedirs(a.out, exist_ok=True)
        json.dump({"best_val_loss": best, "history": history, "args": vars(a)},
                  open(os.path.join(a.out, f"{a.arch}_s{a.seed}.json"), "w"), indent=1)
        print(f"[dsteps] {a.arch} s{a.seed}: best val loss {best:.4f}", flush=True)
        return
    hits, picks = best_of(run_scores(head, st, ac, runs, refs), runs_of_task, passed)
    n = len(runs_of_task)
    res = {"successes": hits, "tasks": n, "accuracy": hits / n,
           "random": sum(sum(passed[r] for r in rs) / len(rs) for rs in runs_of_task.values()) / n,
           "oracle": sum(any(passed[r] for r in rs) for rs in runs_of_task.values()) / n,
           "train_steps": len(train), "val_steps": len(val), "heldout_runs": len(runs),
           "head_params": sum(p.numel() for p in head.parameters()), "best_val_loss": best,
           "history": history, "picks": picks, "args": vars(a)}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"{a.arch}_s{a.seed}.json"), "w"), indent=1)
    print(f"[dsteps] {a.arch} s{a.seed}: {hits:g}/{n} = {hits / n:.3f} (random {res['random']:.3f}, oracle "
          f"{res['oracle']:.3f}; train steps {len(train)}, val steps {len(val)}, held-out runs {len(runs)})", flush=True)


if __name__ == "__main__":
    main()
