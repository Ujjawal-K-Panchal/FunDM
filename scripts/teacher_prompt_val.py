"""Teacher accuracy on the held-out validation split of the 38k training pool, for both prompts.

python scripts/teacher_prompt_val.py [--seed 0] [--out results/teacher_prompt_val.json]

Reads the cached teacher letter log-probabilities (feats_kd = multi-line, feats_kd_oneline = single-line); no GPU.
The validation rows are the ones kd_train holds out (question split of the multi-line features, the paper's prompt).
Gold labels are used here only to score the frozen teacher.
"""
import argparse
import json
import random

import torch

from mcq import question_split


def teacher_correct(f: dict, rows: list[int]) -> list[float]:
    """1.0 where the teacher's top letter is the gold option, per row."""
    off, tl = f["off"], f["tl"]
    return [float(int(tl[off[i]:off[i + 1]].argmax()) == f["order"][i].index(f["gold"][i])) for i in rows]


def letter_mass(f: dict, rows: list[int]) -> float:
    """Mean probability the teacher puts on the valid option letters, before renormalizing."""
    off, tl = f["off"], f["tl"]
    return sum(float(tl[off[i]:off[i + 1]].exp().sum()) for i in rows) / len(rows)


def chance(f: dict, rows: list[int]) -> float:
    return sum(1 / len(f["order"][i]) for i in rows) / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", default="feats/train_groups.json")
    ap.add_argument("--multi", default="feats_kd/train_L32.pt")
    ap.add_argument("--single", default="feats_kd_oneline/train_L32.pt")
    ap.add_argument("--out", default="results/teacher_prompt_val.json")
    a = ap.parse_args()

    load = lambda p: torch.load(p, weights_only=False, mmap=True)  # noqa: E731
    multi, single = load(a.multi), load(a.single)
    val, train = question_split(multi, a.groups, a.seed)
    print(f"train pool: {len(multi['id'])} rows multi-line, {len(single['id'])} single-line; "
          f"validation split {len(val)} rows, training split {len(train)} rows")

    at = {rid: i for i, rid in enumerate(single["id"])}
    shared = [i for i in val if multi["id"][i] in at]
    j = [at[multi["id"][i]] for i in shared]
    if len(shared) != len(val):
        print(f"{len(val) - len(shared)} validation rows are missing from the single-line features (over the length cap)")

    ok_m, ok_s = teacher_correct(multi, shared), teacher_correct(single, j)
    n = len(shared)
    acc_m, acc_s = sum(ok_m) / n, sum(ok_s) / n
    only_m = sum(m > s for m, s in zip(ok_m, ok_s))
    only_s = sum(s > m for m, s in zip(ok_m, ok_s))
    d = [m - s for m, s in zip(ok_m, ok_s)]
    rng = random.Random(0)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    res = {"n_validation_rows": n, "seed": a.seed,
           "multi_line": {"accuracy": acc_m, "letter_mass": letter_mass(multi, shared)},
           "single_line": {"accuracy": acc_s, "letter_mass": letter_mass(single, j)},
           "chance": chance(multi, shared), "difference": acc_m - acc_s, "difference_ci95": [boots[50], boots[1949]],
           "only_multi_right": only_m, "only_single_right": only_s}
    print(f"\nteacher accuracy on {n} validation rows (chance {res['chance']:.3f}):")
    print(f"  multi-line  {acc_m:.4f}   letter mass {res['multi_line']['letter_mass']:.3f}")
    print(f"  single-line {acc_s:.4f}   letter mass {res['single_line']['letter_mass']:.3f}")
    print(f"  difference  {acc_m - acc_s:+.4f}  95% bootstrap CI [{boots[50]:+.4f}, {boots[1949]:+.4f}]  "
          f"only multi right {only_m}, only single right {only_s}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
