"""Numbers reported only in the text: the full model's readout and the throughput timing.

python scripts/extras.py <command> [args]; commands: readouts, timing.
"""

import argparse
import contextlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from clm.embedder import Embedder, l2
from clm.engine import DEFAULT_MODEL, Engine
from clm.heads import HeadPair
from transformers import AutoModelForCausalLM, AutoTokenizer

import data
from data import revision_for
from heads import ComparatorHead, OptionHead, build_row, collate
from baselines import pairs_for
from mcq import LETTERS, build_paper, MidHead

# ---------------------------------------------------------------- readouts
READOUT_LETTERS = "ABCDEFGHIJ"


def letter_prompt(q: str, opts: list[str]) -> str:
    lines = "\n".join(f"{READOUT_LETTERS[i]}. {o}" for i, o in enumerate(opts))
    return f"The following is a multiple choice question. Answer with the letter of the correct option.\n\n" \
           f"Question: {q.strip()}\nOptions:\n{lines}\nAnswer:"


def cont_logprobs(llm, tok, pairs: list[tuple[str, str]]) -> list[float]:
    """Sum of log p(continuation tokens | context) for each (context, continuation)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    prompts, spans = [], []
    for ctx, cont in pairs:
        c, k = tok.encode(ctx, add_special_tokens=False), tok.encode(cont, add_special_tokens=False)
        prompts.append(TokensPrompt(prompt_token_ids=c + k))
        spans.append((len(c), c + k))
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)
    res = []
    for o, (start, ids) in zip(outs, spans):
        res.append(sum(o.prompt_logprobs[t][ids[t]].logprob for t in range(start, len(ids))))
    return res


def summarize(scores: list[np.ndarray], gold: list[int]) -> dict:
    conf, correct, nll = [], [], []
    for s, g in zip(scores, gold):
        p = np.exp(s - s.max()); p /= p.sum()
        conf.append(p.max()); correct.append(float(p.argmax() == g)); nll.append(-np.log(max(p[g], 1e-12)))
    conf, correct = np.array(conf), np.array(correct)
    edges = np.linspace(0, 1, 16)
    ece = sum(((conf > lo) & (conf <= hi)).mean() * abs(conf[(conf > lo) & (conf <= hi)].mean() - correct[(conf > lo) & (conf <= hi)].mean())
              for lo, hi in zip(edges[:-1], edges[1:]) if ((conf > lo) & (conf <= hi)).any())
    return {"accuracy": float(correct.mean()), "nll": float(np.mean(nll)), "ece": float(ece)}


def readouts(argv=None) -> None:
    """Qwen3-8B's own pick on MMLU-Pro, 0-shot, no reasoning: answer-letter and answer log-probability readouts.
    
    letter: options listed as A., B., ...; score = log p(" X" | prompt) for each letter.
    logprob: score = sum log p(option | "Question: ... Answer:"); logprob_norm divides by the option's characters.
    Each readout's softmax over the options gives NLL and ECE; accuracy is its argmax.
    """
    from vllm import LLM
    ap = argparse.ArgumentParser(prog="extras.py readouts", description=readouts.__doc__)
    ap.add_argument("--util", type=float, default=0.8)
    ap.add_argument("--out", default="results/qwen_mmlu_pro.json")
    a = ap.parse_args(argv)
    items = data.mmlu_pro()
    llm = LLM("Qwen/Qwen3-8B", revision=revision_for("Qwen/Qwen3-8B"), tokenizer_revision=revision_for("Qwen/Qwen3-8B"),
              dtype="bfloat16", gpu_memory_utilization=a.util, max_model_len=4096,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    gold = [it["gold"] for it in items]
    out = {"n_questions": len(items), "prompting": "0-shot, no chat template, no reasoning"}
    pairs = [(letter_prompt(it["question"], it["options"]), " " + READOUT_LETTERS[i]) for it in items
             for i in range(len(it["options"]))]
    flat = cont_logprobs(llm, tok, pairs)
    letter, k = [], 0
    for it in items:
        n = len(it["options"]); letter.append(np.array(flat[k:k + n])); k += n
    out["letter"] = summarize(letter, gold)
    print(f"[qwen] letter acc {out['letter']['accuracy']:.4f}", flush=True)
    pairs = [(f"Question: {it['question'].strip()}\nAnswer:", " " + o.strip()) for it in items for o in it["options"]]
    flat = cont_logprobs(llm, tok, pairs)
    lp, lpn, k = [], [], 0
    for it in items:
        n = len(it["options"]); s = np.array(flat[k:k + n]); k += n
        lp.append(s); lpn.append(s / np.array([max(1, len(o.strip())) for o in it["options"]]))
    out["logprob"], out["logprob_norm"] = summarize(lp, gold), summarize(lpn, gold)
    print(f"[qwen] logprob acc {out['logprob']['accuracy']:.4f} | per-char acc {out['logprob_norm']['accuracy']:.4f}",
          flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    with open(a.out.replace(".json", "_preds.jsonl"), "w") as f:
        for it, a1, a2 in zip(items, letter, lp):
            f.write(json.dumps({"question_id": it["id"], "gold": it["gold"], "letter": a1.round(4).tolist(),
                                "logprob": a2.round(4).tolist()}) + "\n")


# ---------------------------------------------------------------- timing
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KD_LAYERS = (16, 20, 24, 28, 32)


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def pack(lengths: list[int], tokens_per_batch: int) -> list[list[int]]:
    """Indices sorted by length, grouped so each padded group fits the token budget."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur = [], []
    for i in order:
        if cur and lengths[i] * (len(cur) + 1) > tokens_per_batch:
            out.append(cur); cur = []
        cur.append(i)
    return out + [cur]


@contextlib.contextmanager
def truncated(bb, L: int):
    """The backbone's first L layers; the output is the raw layer-L state the KD heads read."""
    layers, norm = bb.layers, bb.norm
    bb.layers, bb.norm = torch.nn.ModuleList(layers[:L]), torch.nn.Identity()
    try:
        yield
    finally:
        bb.layers, bb.norm = layers, norm


class Timer:
    def __init__(self, tok, backbone, clm: HeadPair, linear: OptionHead, comp: ComparatorHead, lm_head=None, kd=None):
        self.tok, self.bb, self.clm, self.linear, self.comp = tok, backbone, clm, linear, comp
        self.lm_head, self.kd = lm_head, kd or {}
        ids = [tok.encode(" " + c, add_special_tokens=False)[0] for c in LETTERS["paper"]]
        self.letter_ids = torch.tensor(ids, device="cuda")

    def kd_encode(self, rows):
        """KD rows (ids, final position, option-end positions, order) -> final state, option-end states, mask."""
        T, K = max(len(r[0]) for r in rows), max(len(r[2]) for r in rows)
        ids = torch.full((len(rows), T), self.tok.pad_token_id)
        attn = torch.zeros((len(rows), T), dtype=torch.long)
        ends = torch.zeros((len(rows), K), dtype=torch.long)
        mask = torch.zeros((len(rows), K), dtype=torch.bool)
        final = torch.tensor([r[1] for r in rows])
        for b, r in enumerate(rows):
            ids[b, :len(r[0])] = torch.tensor(r[0]); attn[b, :len(r[0])] = 1
            ends[b, :len(r[2])] = torch.tensor(r[2]); mask[b, :len(r[2])] = True
        h = self.bb(input_ids=ids.cuda(), attention_mask=attn.cuda()).last_hidden_state
        b = torch.arange(len(rows), device="cuda")
        return h[b, final.cuda()], h[b[:, None], ends.cuda()], mask.cuda()

    def teacher_pick(self, rows):
        """The full model's letter pick: all 36 layers, the LM head at the final position, softmax over the letters."""
        hq, _, mask = self.kd_encode(rows)
        lp = torch.log_softmax(self.lm_head(hq).float(), -1)[:, self.letter_ids[:mask.shape[1]]]
        return torch.softmax(lp.masked_fill(~mask, float("-inf")), -1)

    def last_states(self, texts: list[str]) -> torch.Tensor:
        enc = self.tok(texts, return_tensors="pt", padding=True).to("cuda")
        h = self.bb(**enc).last_hidden_state
        last = enc["attention_mask"].sum(1) - 1   # right padding
        return h[torch.arange(len(texts), device="cuda"), last]

    def clm_encode(self, built: list[tuple[str, list[str]]]):
        texts = [t for s, cs in built for t in [s, *cs]]
        return self.last_states(texts)

    def clm_head(self, built, states: torch.Tensor) -> list[torch.Tensor]:
        s_idx, a_idx, n, k = [], [], [], 0
        for _, cs in built:
            s_idx.append(k); a_idx.extend(range(k + 1, k + 1 + len(cs))); n.append(len(cs)); k += 1 + len(cs)
        x = F.normalize(states.float(), dim=-1)   # CLM's embedder L2-normalises encoder states
        zs = F.normalize(self.clm.state_head(x[s_idx]), dim=-1)
        za = F.normalize(self.clm.action_head(x[a_idx]), dim=-1)
        out, k = [], 0
        for i, m in enumerate(n):
            out.append(torch.softmax(self.clm.scale * (za[k:k + m] @ zs[i]), -1)); k += m
        return out

    def ours_encode(self, rows):
        ids, attn, ans, sep, mask = collate(rows, self.tok.pad_token_id, "cuda")
        h = self.bb(input_ids=ids, attention_mask=attn).last_hidden_state
        b = torch.arange(len(rows), device="cuda")
        return h[b, ans], h[b[:, None], sep], mask

    def ours_head(self, head, hq, hk, mask):
        return torch.softmax(head.score(hq, hk, mask), -1)


class LocalEmbedder(Embedder):
    """CLM's Embedder (dedup + text cache) with its vLLM fetch replaced by our in-process encoder."""

    def __init__(self, timer: Timer, lengths: dict[str, int], tokens_per_batch: int):
        super().__init__(max_tokens=None, batch=1 << 30)   # one fetch per call, packed here
        self.timer, self.lengths, self.tpb = timer, lengths, tokens_per_batch

    def _fetch(self, texts: list[str]) -> tuple[list[np.ndarray], int]:
        lens = [self.lengths[x] for x in texts]
        h = torch.empty(len(texts), self.timer.bb.config.hidden_size, device="cuda", dtype=torch.float32)
        for g in pack(lens, self.tpb):
            h[g] = self.timer.last_states([texts[i] for i in g]).float()
        return [l2(v) for v in h.cpu().numpy()], sum(lens)


def clm_cached(t: Timer, built, lengths: dict[str, int], tokens_per_batch: int, ckpt: str,
               warm_options: bool) -> tuple[dict, list[torch.Tensor]]:
    """CLM's serving path (Engine._cached: text cache + device arena), cold or with options pre-cached."""
    eng = Engine(embedder=LocalEmbedder(t, lengths, tokens_per_batch), checkpoint=ckpt, device="cuda")
    head = eng.heads[DEFAULT_MODEL]
    ns, dim = head.namespace, head.proj_dim
    states, cands = [s for s, _ in built], [c for _, cs in built for c in cs]
    if warm_options:
        eng._cached(f"{ns}/action", dim, cands, [], head.project_actions)
    tokens: list[int] = []
    t0 = sync_time()
    zq = eng._cached(f"{ns}/state", dim, states, tokens, head.project_states)
    za = eng._cached(f"{ns}/action", dim, cands, tokens, head.project_actions)
    probs, k = [], 0
    for i, (_, cs) in enumerate(built):
        probs.append(torch.softmax(head.scale * (za[k:k + len(cs)] @ zq[i]), -1)); k += len(cs)
    res = {"questions_per_s": len(built) / (sync_time() - t0), "encoder_tokens": sum(tokens),
           "arena": eng.arena.stats() if eng.arena else None}
    del eng
    return res, probs


def batch1(t: Timer, items, built, rows) -> dict:
    res = {m: {"total": [], "head": []} for m in ("clm", "linear", "comparator")}
    for it, b, r in zip(items, built, rows):
        t0 = sync_time(); st = t.clm_encode([b]); t1 = sync_time(); t.clm_head([b], st); t2 = sync_time()
        res["clm"]["total"].append(t2 - t0); res["clm"]["head"].append(t2 - t1)
        t0 = sync_time(); hq, hk, m = t.ours_encode([r]); t1 = sync_time()
        t.ours_head(t.linear, hq, hk, m); t2 = sync_time()
        res["linear"]["total"].append(t2 - t0); res["linear"]["head"].append(t2 - t1)
        t3 = sync_time(); t.ours_head(t.comp, hq, hk, m); t4 = sync_time()
        res["comparator"]["total"].append((t1 - t0) + (t4 - t3)); res["comparator"]["head"].append(t4 - t3)
    ms = lambda xs, q: float(np.percentile(xs, q) * 1000)
    return {m: {"total_ms_median": ms(v["total"], 50), "total_ms_p90": ms(v["total"], 90),
                "head_ms_median": ms(v["head"], 50), "head_ms_p90": ms(v["head"], 90)} for m, v in res.items()}


def throughput(t: Timer, built, rows, tokens_per_batch: int, ckpt: str, reps: int, kd_rows=None) -> dict:
    """One untimed pass per method, then `reps` timed passes interleaved across methods; median kept."""
    texts = [x for s, cs in built for x in [s, *cs]]
    lengths = {x: len(t.tok.encode(x)) for x in texts}
    ours_len = [len(r.ids) for r in rows]

    def clm():
        # CLM: pack every text of every question by length (its best case), then run the heads once.
        t0 = sync_time()
        states = torch.empty(len(texts), t.bb.config.hidden_size, device="cuda", dtype=torch.bfloat16)
        for g in pack([lengths[x] for x in texts], tokens_per_batch):
            states[g] = t.last_states([texts[i] for i in g])
        p = t.clm_head(built, states)
        return {"questions_per_s": len(built) / (sync_time() - t0), "encoder_tokens": sum(lengths[x] for x in texts)}, p

    def ours(head):
        t0 = sync_time()
        for g in pack(ours_len, tokens_per_batch):
            hq, hk, m = t.ours_encode([rows[i] for i in g]); t.ours_head(head, hq, hk, m)
        return {"questions_per_s": len(rows) / (sync_time() - t0)}, None

    def kd(arch, L):
        t0 = sync_time()
        with truncated(t.bb, L):
            for g in pack(kd_len, tokens_per_batch):
                hq, hk, m = t.kd_encode([kd_rows[i] for i in g]); t.ours_head(t.kd[arch], hq, hk, m)
        return {"questions_per_s": len(kd_rows) / (sync_time() - t0)}, None

    def teacher():
        t0 = sync_time()
        for g in pack(kd_len, tokens_per_batch):
            t.teacher_pick([kd_rows[i] for i in g])
        return {"questions_per_s": len(kd_rows) / (sync_time() - t0)}, None

    methods = {"clm": clm,
               "clm_cached": lambda: clm_cached(t, built, lengths, tokens_per_batch, ckpt, False),
               "clm_cached_warm_options": lambda: clm_cached(t, built, lengths, tokens_per_batch, ckpt, True),
               "linear": lambda: ours(t.linear), "comparator": lambda: ours(t.comp)}
    if kd_rows:
        kd_len = [len(r[0]) for r in kd_rows]
        methods["kd_teacher"] = teacher
        for L in KD_LAYERS:
            for arch in ("linear", "comparator"):
                methods[f"kd_{arch}_L{L}"] = lambda arch=arch, L=L: kd(arch, L)
    for f in methods.values():
        f()   # untimed: allocator growth and first-shape kernels
    runs = {m: [] for m in methods}
    for _ in range(reps):
        for m, f in methods.items():
            runs[m].append(f())
    out, probs = {}, {m: rs[-1][1] for m, rs in runs.items()}
    for m, rs in runs.items():
        qps = [r[0]["questions_per_s"] for r in rs]
        out[m] = {**rs[-1][0], "questions_per_s": float(np.median(qps)), "questions_per_s_reps": qps}
    for name in ("clm_cached", "clm_cached_warm_options"):
        out[name]["argmax_agrees_with_clm"] = float(np.mean(
            [a.argmax().item() == b.argmax().item() for a, b in zip(probs["clm"], probs[name])]))
    out["unique_text_fraction"] = len(lengths) / len(texts)
    return out


def timing(argv=None) -> None:
    """Per-prediction time of CLM vs our heads, on the same GPU, backbone (Qwen3-8B, bf16) and questions.
    
    CLM: one forward per text (question + each option, last-token state), then its state/action heads.
    CLM cached: CLM's own Engine, Embedder and VectorArena, with only the vLLM fetch swapped for the same encoder.
    Ours (CE): one forward over the joined row (question + all options), then the linear or comparator head.
    KD: the KD prompt through the first L layers only (no final norm, no LM head), then the KD head; the teacher is the
    same prompt through all 36 layers and the LM head (the full model's letter pick).
    Head weights don't affect time, so our heads are fresh modules of the trained shape; CLM's is the released checkpoint.
    Reports batch-1 median/p90 (encode + head, and head only) and batched throughput. Writes results/timing.json.
    """
    ap = argparse.ArgumentParser(prog="extras.py timing", description=timing.__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--tokens-per-batch", type=int, default=16384)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--no-batch1", action="store_true", help="skip batch-1 latency; throughput only")
    ap.add_argument("--reps", type=int, default=2, help="timed throughput passes per method, after one untimed")
    ap.add_argument("--no-kd", action="store_true", help="skip the KD early-exit heads and their teacher")
    ap.add_argument("--out", default="results/timing.json")
    a = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(a.model, revision=revision_for(a.model), padding_side="right")
    lm = AutoModelForCausalLM.from_pretrained(a.model, revision=revision_for(a.model), dtype=torch.bfloat16, device_map="cuda").eval()
    bb = lm.model
    ckpt = os.environ["CLM_CKPT"]
    clm = HeadPair("clm-latest", ckpt, device="cuda").ensure()
    hidden = bb.config.hidden_size
    kd = {arch: MidHead(arch, hidden).cuda().eval() for arch in ("linear", "comparator")}
    t = Timer(tok, bb, clm, OptionHead(hidden).cuda().eval(), ComparatorHead(hidden).cuda().eval(), lm.lm_head, kd)
    sets = {"mmlu_pro": data.mmlu_pro, **data.EVAL_SETS}
    report = {"gpu": torch.cuda.get_device_name(), "model": a.model, "dtype": "bf16", "n_per_set": a.n, "sets": {}}
    with torch.inference_mode():
        for s, load in sets.items():
            items = load()
            items = random.Random(0).sample(items, min(a.n, len(items)))
            # BoolQ uses CLM's yes/no form; ours reads the passage + question with options no/yes.
            built = [pairs_for(s, it) for it in items]
            rows = [build_row(tok, it["question"], it["options"]) for it in items]
            rng = random.Random(0)
            kd_rows = None if a.no_kd else [build_paper(tok, it["question"], it["options"], rng) for it in items]
            batch1(t, items[:a.warmup], built[:a.warmup], rows[:a.warmup])
            r = {"batched": throughput(t, built, rows, a.tokens_per_batch, ckpt, a.reps, kd_rows),
                 "mean_tokens": {"clm": float(np.mean([sum(len(tok.encode(x)) for x in [st, *cs]) for st, cs in built])),
                                 "ours": float(np.mean([len(r.ids) for r in rows]))}}
            if kd_rows:
                r["mean_tokens"]["kd"] = float(np.mean([len(x[0]) for x in kd_rows]))
            if not a.no_batch1:
                r["batch1"] = batch1(t, items, built, rows)
            report["sets"][s] = r
            print(s, json.dumps(r), flush=True)
    json.dump(report, open(os.path.join(ROOT, a.out), "w"), indent=1)


COMMANDS = {"readouts": readouts, "timing": timing}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit("usage: python scripts/extras.py {readouts|timing} [args]")
    COMMANDS[sys.argv[1]](sys.argv[2:])
