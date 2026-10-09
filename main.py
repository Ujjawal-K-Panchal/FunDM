"""Reproduce the paper's tables, plotted data and experiment diagnostics: .venv/bin/python main.py --table 1  (see --help).

Each table or quantitative figure runs missing steps and checks every displayed value against the paper (exit 1 on a difference).
A step runs only when its outputs are missing, so tables share caches (feats*/, data/). Step logs go to logs/<step>.log.
"""
import argparse
import contextlib
import functools
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from data import EXIT_LAYER, revision_for  # noqa: E402

PY, PY_CLM = ".venv/bin/python", ".venv-clm/bin/python"
QWEN, CLM_REPO = "Qwen/Qwen3-8B", "Contrastive-LM/CLM-v0.1-8B"
SETS = ["mmlu_pro", "mmlu", "openbookqa", "commonsenseqa", "piqa", "hellaswag", "winogrande", "boolq"]
NAMES = ["MMLU-Pro", "MMLU", "OpenBookQA", "CommonsenseQA", "PIQA", "HellaSwag", "WinoGrande", "BoolQ"]
EVAL7 = ["mmlu", "openbookqa", "commonsenseqa", "hellaswag", "piqa", "winogrande", "boolq"]   # order of the paper's runs
ARCHS, LAYERS, MAIN, SEEDS = ("linear", "comparator"), [16, 20, 24, 28, 32], EXIT_LAYER, range(5)


# ---------------------------------------------------------------- steps
class Step:
    def __init__(self, out, cmds, deps=(), env=None, server=None):
        self.out, self.cmds, self.deps, self.env, self.server = list(out), cmds, list(deps), env or {}, server


STEPS: dict[str, Step] = {}


def step(name, out, *cmds, deps=(), env=None, server=None):
    STEPS[name] = Step(out, cmds, deps, env, server)


def clm_ckpt() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(CLM_REPO, "CLM_v0.1-8B.pt", revision=revision_for(CLM_REPO))


def py(code: str) -> list:
    return [PY, "-c", code]


def _up(port: int) -> bool:
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).status == 200
    except OSError:
        return False


@contextlib.contextmanager
def vllm(port: int, util: float):
    """The pinned Qwen3-8B embedding server CLM reads from (the flags of the paper's runs)."""
    rev = revision_for(QWEN)
    log = open(f"logs/vllm_{port}.log", "w")
    p = subprocess.Popen([".venv-clm/bin/vllm", "serve", QWEN, "--revision", rev, "--tokenizer-revision", rev,
                          "--served-model-name", "qwen3-8b", "--runner", "pooling", "--enforce-eager", "--enable-prefix-caching",
                          "--max-model-len", "2048", "--gpu-memory-utilization", str(util), "--max-num-seqs", "64",
                          "--port", str(port)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(180):
            if _up(port):
                break
            if p.poll() is not None:
                sys.exit(f"[step] vLLM exited; see logs/vllm_{port}.log")
            time.sleep(5)
        else:
            sys.exit(f"[step] vLLM did not come up; see logs/vllm_{port}.log")
        yield
    finally:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait()


def ensure(*names: str) -> None:
    """Run each step (and, first, its missing dependencies) unless all its outputs exist."""
    for name in names:
        s = STEPS[name]
        if s.out and all(os.path.exists(p) for p in s.out):
            continue
        ensure(*s.deps)
        for p in s.out:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        env = {**os.environ, **{k: v() if callable(v) else v for k, v in s.env.items()}}
        t0 = time.time()
        print(f"[step] {name}: running (logs/{name}.log)", flush=True)
        with open(f"logs/{name}.log", "a") as log, (vllm(*s.server) if s.server else contextlib.nullcontext()):
            for cmd in s.cmds:
                cmd = [x() if callable(x) else x for x in cmd]
                log.write(f"$ {' '.join(cmd)}\n")
                log.flush()
                if subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode:
                    sys.exit(f"[step] {name} failed; see logs/{name}.log")
        missing = [p for p in s.out if not os.path.exists(p)]
        if missing:
            sys.exit(f"[step] {name} finished without writing {missing}")
        print(f"[step] {name}: done in {(time.time() - t0) / 60:.1f} min", flush=True)


# Multiple-choice FunDM (Tables 1, 4, 8-10; Figure 3).
step("groups", ["feats/train_groups.json"], [PY, "scripts/mcq.py", "split_overlap", "--groups-only"])
step("decontam", ["feats/extra_excluded.json"], [PY, "scripts/mcq.py", "decontam"])
for s in ["train", "extra", *SETS]:
    step(f"kd_feats_{s}", [f"feats_kd/{s}_L32.pt"], [PY, "scripts/mcq.py", "kd_extract", "--set", s, "--out", f"feats_kd/{s}"],
         deps=["decontam"] if s == "extra" else [])
step("kd_oneline_feats_train", ["feats_kd_oneline/train_L32.pt"],   # single-line prompt, for the validation comparison only
     [PY, "scripts/mcq.py", "kd_extract", "--set", "train", "--prompt", "one-line", "--out", "feats_kd_oneline/train"])
step("teacher_prompt_val", ["results/teacher_prompt_val.json"], [PY, "scripts/teacher_prompt_val.py"],
     deps=["groups", "kd_feats_train", "kd_oneline_feats_train"])
for a in ARCHS:
    base = ["groups", "kd_feats_train", *(f"kd_feats_{s}" for s in SETS)]
    step(f"kd_{a}", [f"results/kd/{a}_L{L}_s0_{s}.json" for L in LAYERS for s in SETS],
         [PY, "scripts/mcq.py", "kd_train", "--feats", "feats_kd", "--arch", a, "--out", "results/kd"], deps=base)
    step(f"kd_big_{a}", [f"results/kd_big/{a}_L{L}_s0_{s}.json" for L in LAYERS for s in SETS],
         [PY, "scripts/mcq.py", "kd_train", "--feats", "feats_kd", "--arch", a, "--train-sets", "train,extra", "--out", "results/kd_big"],
         deps=base + ["kd_feats_extra"])
    for t in ("gold", "teacher_hard", "teacher"):   # Table 4: only the target changes; seeds 0-2 at layer 24
        for s in (0, 1, 2):
            if t == "teacher" and s == 0:
                continue   # the main KD run, results/kd
            step(f"kd_target_{t}_{a}_s{s}", [f"results/kd_target/{t}/{a}_L{MAIN}_s{s}_{x}.json" for x in SETS],
                 [PY, "scripts/mcq.py", "kd_train", "--feats", "feats_kd", "--arch", a, "--layers", str(MAIN), "--seed", str(s),
                  "--target", t, "--out", f"results/kd_target/{t}"], deps=base)
KD_TARGET = [f"kd_target_{t}_{a}_s{s}" for t in ("gold", "teacher_hard", "teacher") for a in ARCHS for s in (0, 1, 2)
             if (t, s) != ("teacher", 0)]

# Original final-layer CE heads (Tables 5-7).
for s in ["train", *SETS]:
    step(f"ce_feats_{s}", [f"feats/{s}.pt"], [PY, "scripts/mcq.py", "extract_features", "--set", s, "--out", f"feats/{s}.pt"])
CE_FEATS = ["groups", *(f"ce_feats_{s}" for s in ["train", *SETS])]
TESTS = [f"feats/{s}.pt" for s in ["mmlu_pro", *EVAL7]]
step("ce_linear", [f"results/head_{s}{x}" for s in SETS for x in (".json", "_preds.jsonl")],
     [PY, "scripts/mcq.py", "train_head", "--train", "feats/train.pt", "--test", *TESTS], deps=CE_FEATS)
step("ce_comparator", [f"results/comparator_sets/head_{s}.json" for s in SETS],
     [PY, "scripts/mcq.py", "train_head", "--train", "feats/train.pt", "--test", *TESTS, "--arch", "comparator", "--lr", "5e-4",
      "--out-dir", "results/comparator_sets", "--save", "feats/head_comparator_sets.pt"], deps=CE_FEATS)
for name, extra in [("together_linear_s1", ["--seed", "1"]), ("together_linear_s2", ["--seed", "2"]),
                    ("together_clm_mlp", ["--arch", "clm_mlp", "--lr", "5e-4"])]:
    step(f"grid_{name}", [f"results/grid/{name}/head_mmlu_pro.json"],
         [PY, "scripts/mcq.py", "train_head", "--out-dir", f"results/grid/{name}", "--save", f"feats/grid_{name}.pt",
          "--train", "feats/train.pt", "--test", "feats/mmlu_pro.pt", *extra], deps=["groups", "ce_feats_train", "ce_feats_mmlu_pro"])
for s in ("mmlu_pro", "train", *EVAL7):   # CLM-style separate embeddings from the vLLM server
    step(f"apart_feats_{s}", [f"feats/{s}_apart.pt"], [PY_CLM, "scripts/mcq.py", "embed_apart", "--set", s, "--out", f"feats/{s}_apart.pt"],
         server=(8091, 0.2))
for a, extra in [("linear", []), ("comparator", ["--arch", "comparator", "--lr", "5e-4"]), ("clm_mlp", ["--arch", "clm_mlp", "--lr", "5e-4"])]:
    step(f"grid_apart_{a}", [f"results/grid/apart_{a}/head_mmlu_pro_apart.json"],
         [PY, "scripts/mcq.py", "train_head", "--out-dir", f"results/grid/apart_{a}", "--save", f"feats/grid_apart_{a}.pt",
          "--train", "feats/train_apart.pt", "--test", "feats/mmlu_pro_apart.pt", *extra],
         deps=["groups", "apart_feats_train", "apart_feats_mmlu_pro"])
step("clm_shaped_sets", [f"results/clm_shaped_sets/head_{s}_apart.json" for s in SETS],   # CLM architecture from scratch, every set
     [PY, "scripts/mcq.py", "train_head", "--out-dir", "results/clm_shaped_sets", "--train", "feats/train_apart.pt",
      "--test", *(f"feats/{s}_apart.pt" for s in SETS), "--arch", "clm_mlp", "--lr", "5e-4",
      "--save", "feats/head_clm_shaped_sets.pt"],
     deps=["groups", "apart_feats_train", *(f"apart_feats_{s}" for s in SETS)])

# Table 6: option only (CLM format, above), question + option, full list; three seeds each.
# Question + option reads each option in its own row after the question, like the joint row (template, final-layer states,
# readout positions); its query comes from a question-only row.
MCQ_SEEDS = (0, 1, 2)
HEAD_ARGS = {"linear": [], "comparator": ["--arch", "comparator", "--lr", "5e-4"], "clm_mlp": ["--arch", "clm_mlp", "--lr", "5e-4"]}
for s in ("train", "mmlu_pro"):
    step(f"ce_feats_question_option_{s}", [f"feats/{s}_question_option.pt"],
         [PY, "scripts/mcq.py", "extract_features", "--set", s, "--mode", "question_option", "--out", f"feats/{s}_question_option.pt"])
step("pass_cost", ["results/pass_cost.json"], [PY, "scripts/mcq.py", "pass_cost", "--out", "results/pass_cost.json"])
for a, extra in HEAD_ARGS.items():
    for s in MCQ_SEEDS:
        step(f"grid_question_option_{a}_s{s}", [f"results/grid/question_option_{a}_s{s}/head_mmlu_pro_question_option.json"],
             [PY, "scripts/mcq.py", "train_head", "--out-dir", f"results/grid/question_option_{a}_s{s}", "--save",
              f"feats/grid_question_option_{a}_s{s}.pt", "--train", "feats/train_question_option.pt",
              "--test", "feats/mmlu_pro_question_option.pt", "--seed", str(s), *extra],
             deps=["groups", "ce_feats_question_option_train", "ce_feats_question_option_mmlu_pro"])
        if s:   # seed 0 of the CLM-format and joint arms is above
            step(f"grid_apart_{a}_s{s}", [f"results/grid/apart_{a}_s{s}/head_mmlu_pro_apart.json"],
                 [PY, "scripts/mcq.py", "train_head", "--out-dir", f"results/grid/apart_{a}_s{s}", "--save", f"feats/grid_apart_{a}_s{s}.pt",
                  "--train", "feats/train_apart.pt", "--test", "feats/mmlu_pro_apart.pt", "--seed", str(s), *extra],
                 deps=["groups", "apart_feats_train", "apart_feats_mmlu_pro"])
            if a != "linear":
                step(f"grid_together_{a}_s{s}", [f"results/grid/together_{a}_s{s}/head_mmlu_pro.json"],
                     [PY, "scripts/mcq.py", "train_head", "--out-dir", f"results/grid/together_{a}_s{s}", "--save",
                      f"feats/grid_together_{a}_s{s}.pt", "--train", "feats/train.pt", "--test", "feats/mmlu_pro.pt", "--seed", str(s), *extra],
                     deps=["groups", "ce_feats_train", "ce_feats_mmlu_pro"])
GRID = [f"grid_question_option_{a}_s{s}" for a in HEAD_ARGS for s in MCQ_SEEDS] + [
    f"grid_apart_{a}{f'_s{s}' if s else ''}" for a in HEAD_ARGS for s in MCQ_SEEDS] + [
    "ce_linear", "ce_comparator", "grid_together_linear_s1", "grid_together_linear_s2", "grid_together_clm_mlp"] + [
    f"grid_together_{a}_s{s}" for a in ("comparator", "clm_mlp") for s in (1, 2)]

# Released and fine-tuned CLM (Tables 1, 5).
step("clm_mmlu_pro", ["results/clm_mmlu_pro.json", "results/clm_mmlu_pro_preds.jsonl"], [PY_CLM, "scripts/baselines.py", "eval_mmlu_pro"],
     env={"CLM_CKPT": clm_ckpt}, server=(8090, 0.2))
step("clm_sets", [f"results/clm_{s}.json" for s in EVAL7], [PY_CLM, "scripts/baselines.py", "eval_sets", "--sets", *EVAL7],
     env={"CLM_CKPT": clm_ckpt}, server=(8090, 0.35))
step("clm_choice_data", ["data/clm_choice/mcq/train.parquet", "data/clm_choice/mcq/test.parquet"], [PY, "scripts/baselines.py", "choice_data"])
step("qwen_main_ref", [], py(  # CLM's finetune.py loads Qwen/Qwen3-8B by name, so the cache's main must be the pinned snapshot
    "import os; from huggingface_hub import snapshot_download; from huggingface_hub.constants import HF_HUB_CACHE\n"
    f"pin = '{revision_for(QWEN)}'; snapshot_download('{QWEN}', revision=pin)\n"
    "ref = os.path.join(HF_HUB_CACHE, 'models--Qwen--Qwen3-8B', 'refs', 'main')\n"
    "os.path.exists(ref) or open(ref, 'w').write(pin)\n"
    "assert open(ref).read().strip() == pin, ref + ' is not the pinned Qwen3-8B revision'"))
for i in range(3):
    step(f"clm_ft_s{i}", [f"results/clm_choice/s{i}/finetune_summary.json", f"results/clm_choice/s{i}/best_head.pt"],
         [PY_CLM, "third_party/CLM/train/finetune.py", "--task", "choice", "--data", "data/clm_choice", "--workflow", "mcq",
          "--init-ckpt", clm_ckpt, "--out-dir", f"results/clm_choice/s{i}", "--embed-cache", "data/clm_choice/embeddings",
          "--gpu-mem", "0.25", "--seed", str(i)],
         deps=["clm_choice_data", "qwen_main_ref"] + ([f"clm_ft_s{i - 1}"] if i else []), env={"HF_HUB_OFFLINE": "1"})

# CoderForge (Table 3; Figure 4).
step("cf_data", ["data/coderforge/train.jsonl", "data/coderforge/val.jsonl"], [PY, "scripts/coderforge.py", "fetch"])
step("cf_judge", ["results/coderforge/judge.jsonl"], [PY, "scripts/coderforge.py", "judge"], deps=["cf_data"])
step("cf_feats_test", ["feats_cf/cf_test_L32.pt"], [PY, "scripts/coderforge.py", "cf_extract", "--set", "test", "--out", "feats_cf/cf_test"],
     deps=["cf_data"])
step("cf_feats_train", ["feats_cf/cf_train_L32.pt", "feats_cf/groups.json"],
     [PY, "scripts/coderforge.py", "cf_extract", "--set", "train", "--out", "feats_cf/cf_train", "--groups", "feats_cf/groups.json"], deps=["cf_data"])
for a in ARCHS:
    for s in SEEDS:
        step(f"kd_cf_{a}_s{s}", [f"results/kd_cf/{a}_L{L}_s{s}_cf_test{x}" for L in LAYERS for x in (".json", "_preds.pt")],
             [PY, "scripts/mcq.py", "kd_train", "--feats", "feats_cf", "--arch", a, "--train-sets", "cf_train", "--tests", "cf_test",
              "--groups", "feats_cf/groups.json", "--epochs", "30", "--batch", "32", "--seed", str(s), "--out", "results/kd_cf", "--dump"],
             deps=["cf_feats_test", "cf_feats_train"])
step("cf_summary", ["results/kd_cf/summary.json"], [PY, "scripts/coderforge.py", "summary"],
     deps=["cf_judge", "cf_feats_test", "cf_feats_train", *(f"kd_cf_{a}_s{s}" for a in ARCHS for s in SEEDS)])
EMB = "data/clm_emb/coderforge-swerebench-8k"
step("cf_holdout", [f"{EMB}/holdout.json"], py(
    "import json; from pathlib import Path\n"
    "tasks = json.loads(Path('results/coderforge/tasks.json').read_text())\n"
    f"Path('{EMB}/holdout.json').write_text(json.dumps(tasks['holdout'], indent=1) + '\\n')"))
step("cf_embed", [f"{EMB}/{f}" for f in ("metadata.json", "state_embeddings.pt", "action_embeddings.pt")],
     [PY, "scripts/coderforge.py", "embed"], deps=["cf_data"])
step("cf_together", ["data/clm_emb/coderforge-together/features.pt"], [PY, "scripts/coderforge.py", "together"], deps=["cf_embed"])
for a in ("linear", "comparator", "clm_mlp"):
    out = "results/joint_agentonly" if a == "clm_mlp" else "results/joint_agentonly_a"
    mcq = ["apart_feats_train"] if a == "clm_mlp" else ["ce_feats_train"]   # read only for the paper's random stream
    for s in SEEDS:
        step(f"joint_{a}_s{s}", [f"{out}/B_agent_{a}_s{s}.json"],
             [PY, "scripts/coderforge.py", "joint", "--mode", "B", "--no-mcq", "--arch", a, "--seed", str(s), "--out", out],
             deps=["groups", "cf_embed", "cf_together", *mcq], env={"JOINT_SETS": ""})

# DeepSWE supervised separate-state evaluation (Table 11).
DS_EMB = "data/clm_emb/tarsur385_deepswe-prm-embeddings-8k"
step("ds_head", ["heads/deepswe/best_head.pt", "heads/deepswe/task_split.json", "heads/deepswe/heldout_tasks.json"], py(
    "from huggingface_hub import snapshot_download\n"
    f"snapshot_download('Contrastive-LM/deepswe-clm-heads-8k', revision='{revision_for('Contrastive-LM/deepswe-clm-heads-8k')}', "
    "local_dir='heads/deepswe')"))
step("ds_emb", [f"{DS_EMB}/{f}" for f in ("metadata.json", "state_embeddings.pt", "action_embeddings.pt")], py(
    "import sys; sys.path.insert(0, 'third_party/CLM/preprocessing'); import hf_embeddings\n"
    "from huggingface_hub import snapshot_download\n"
    f"src = snapshot_download('tarsur385/deepswe-prm-embeddings-8k', repo_type='dataset', "
    f"revision='{revision_for('tarsur385/deepswe-prm-embeddings-8k')}', local_dir='data/hf/tarsur385_deepswe-prm-embeddings-8k')\n"
    f"hf_embeddings.download(src, '{DS_EMB}')"))
for a in ("linear", "comparator", "clm_mlp"):
    for s in SEEDS:
        step(f"ds_steps_{a}_s{s}", [f"results/deepswe_steps/{a}_s{s}.json"],
             [PY, "scripts/deepswe.py", "--arch", a, "--lr", "1e-3" if a == "linear" else "5e-4", "--seed", str(s), "--epochs", "30"],
             deps=["ds_head", "ds_emb"])
step("ds_clm", ["results/deepswe_clm_repro.json"],
     [PY_CLM, "third_party/CLM/evaluation/bon_eval.py", "--embeddings-dir", DS_EMB, "--checkpoint", "heads/deepswe/best_head.pt",
      "--tasks-file", "heads/deepswe/heldout_tasks.json", "--n", "4", "--window", "12", "--output", "results/deepswe_clm_repro.json"],
     deps=["ds_head", "ds_emb"])

# Experiments reported only in the text.
step("qwen_readouts", ["results/qwen_mmlu_pro.json"], [PY_CLM, "scripts/extras.py", "readouts"])
step("timing", ["results/timing_cached.json"], [PY, "scripts/extras.py", "timing", "--no-batch1", "--out", "results/timing_cached.json"],
     env={"CLM_CKPT": clm_ckpt, "PYTHONPATH": "third_party/CLM/src"})   # the main venv, with CLM's code on the path
step("overlap", ["results/split_overlap.json"], [PY, "scripts/mcq.py", "split_overlap"], deps=["groups", "ce_feats_train"])
step("cf_released_clm", ["results/released_clm_agent/coderforge.json"], [PY, "scripts/coderforge.py", "released_clm"],
     deps=["cf_together"])
step("cf_adp_overlap", ["results/released_clm_agent/adp_overlap.json"], [PY, "scripts/coderforge.py", "adp_overlap"],
     deps=["cf_embed"])
for s in SEEDS:   # released CLM fine-tuned on CoderForge with its authors' script (their DeepSWE settings)
    step(f"cf_clm_ft_s{s}", [f"results/clm_ft_agent/s{s}/summary.json", f"results/clm_ft_agent/s{s}/best_head.pt"],
         [PY_CLM, "third_party/CLM/train/finetune.py", "--task", "clm", "--emb-dir", EMB, "--holdout-tasks", f"{EMB}/holdout.json",
          "--init-ckpt", clm_ckpt, "--batch", "512", "--seed", str(s), "--out-dir", f"results/clm_ft_agent/s{s}"], deps=["cf_holdout", "cf_embed"])
    step(f"cf_clm_ft_eval_s{s}", [f"results/released_clm_agent/clm_ft_s{s}.json"],
         [PY, "scripts/coderforge.py", "released_clm", "--ckpt", f"results/clm_ft_agent/s{s}/best_head.pt",
          "--out", f"results/released_clm_agent/clm_ft_s{s}.json"], deps=[f"cf_clm_ft_s{s}", "cf_together"])


# ---------------------------------------------------------------- results
def j(path: str):
    return json.load(open(path))


def f(x, d=3) -> str:
    return f"{x:.{d}f}"


def pm(xs, d) -> str:
    return f"{statistics.mean(xs):.{d}f}±{statistics.stdev(xs):.{d}f}"


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}%"


# Qwen3-8B: 36 layers, hidden 4096, MLP 12288, 32 query / 8 KV heads of 128, untied 151,936-token embeddings.
LAYER_PARAMS = 2 * 4096 * 32 * 128 + 2 * 4096 * 8 * 128 + 3 * 4096 * 12288 + 2 * 4096 + 2 * 128
EMBED = 151936 * 4096
TOTAL = 2 * EMBED + 36 * LAYER_PARAMS + 4096  # final RMS norm
CLM_HEAD = 2 * (4096 * 1536 + 1536 + 1536 * 1536 + 1536 + 1536 * 512 + 512 + 2 * 1536) + 1   # two MLPs and a cosine scale
FOH_HEAD = 2 * 4096 * 128                                                         # W_q, W_k


def params(layers: int) -> int:
    return EMBED + layers * LAYER_PARAMS + (4096 if layers == 36 else 0)


def pcell(n) -> str:
    return f"{n / 1e9:.2f}B ({100 * n / TOTAL:.0f}%)"


def bcount(n) -> str:
    return f"{n / 1e9:.2f}B"


def mcount(n) -> str:
    return f"{n / 1e6:.3g}M" if n else "0"


@functools.cache
def kd_runs(res: str) -> dict:
    return {(a, L, s): j(f"results/{res}/{a}_L{L}_s0_{s}.json") for a in ARCHS for L in LAYERS for s in SETS}


@functools.cache
def kd_means(res: str) -> dict:
    r = kd_runs(res)
    return {(a, L): statistics.mean(r[a, L, s]["student"]["accuracy"] for s in SETS) for a in ARCHS for L in LAYERS}


def teacher() -> dict:
    return {s: kd_runs("kd")["linear", MAIN, s]["teacher"]["accuracy"] for s in SETS}


def head_params(res: str, test: str) -> tuple[int, int]:
    return tuple(j(f"results/{res}/{a}_L{MAIN}_s0_{test}.json")["params"] for a in ARCHS)


@functools.cache
def clm() -> dict:
    return {s: j(f"results/clm_{s}.json")["clm"]["accuracy"] for s in SETS}


@functools.cache
def ft_seeds() -> list:
    return [j(f"results/clm_choice/s{i}/finetune_summary.json") for i in range(3)]


def clmft() -> dict:
    return {s: statistics.mean(d["test"]["per_question"][s] for d in ft_seeds()) for s in SETS}


@functools.cache
def ce() -> dict:
    return {s: j(f"results/head_{s}.json") for s in SETS}


@functools.cache
def cmp_ce() -> dict:
    return {s: j(f"results/comparator_sets/head_{s}.json") for s in SETS}


def grid() -> dict:   # not cached: tables call it before and after their own grid steps run
    return {k: j(p)["head"] for k, p in {
        "sep_linear": "results/grid/apart_linear/head_mmlu_pro_apart.json",
        "sep_comparator": "results/grid/apart_comparator/head_mmlu_pro_apart.json",
        "sep_clm_mlp": "results/grid/apart_clm_mlp/head_mmlu_pro_apart.json",
        "joint_clm_mlp": "results/grid/together_clm_mlp/head_mmlu_pro.json",
        "linear_s1": "results/grid/together_linear_s1/head_mmlu_pro.json",
        "linear_s2": "results/grid/together_linear_s2/head_mmlu_pro.json"}.items() if os.path.exists(p)}


def arm_acc(arm: str, a: str, s: int) -> float:
    """Table 5: MMLU-Pro accuracy of head a (seed s): option only (CLM format), question + option, or the full list."""
    if arm == "clm":
        return j(f"results/grid/apart_{a}{f'_s{s}' if s else ''}/head_mmlu_pro_apart.json")["head"]["accuracy"]
    if arm == "joint":
        seed0 = {"linear": "results/head_mmlu_pro.json", "comparator": "results/comparator_sets/head_mmlu_pro.json",
                 "clm_mlp": "results/grid/together_clm_mlp/head_mmlu_pro.json"}
        return j(seed0[a] if s == 0 else f"results/grid/together_{a}_s{s}/head_mmlu_pro.json")["head"]["accuracy"]
    return j(f"results/grid/{arm}_{a}_s{s}/head_mmlu_pro_{arm}.json")["head"]["accuracy"]


def arm_mean(arm: str, a: str) -> float:
    return statistics.mean(arm_acc(arm, a, s) for s in MCQ_SEEDS)


def target_mean(t: str, a: str, s: int) -> float:
    """Table 3: mean accuracy over the eight benchmarks at the exit layer when head a learns target t (seed s)."""
    d = "results/kd" if (t, s) == ("teacher", 0) else f"results/kd_target/{t}"
    return statistics.mean(j(f"{d}/{a}_L{MAIN}_s{s}_{x}.json")["student"]["accuracy"] for x in SETS)


def ds_steps() -> dict:
    return {a: [j(f"results/deepswe_steps/{a}_s{s}.json")["successes"] for s in SEEDS] for a in ("clm_mlp", "linear", "comparator")}


def ce_row(name, d, params):
    return [name, f(d["accuracy"]), f(d["nll"], 2), f(d["ece"]), params]


def table_5():
    return [[name] + [pm([target_mean(t, a, s) for s in MCQ_SEEDS], 3) for a in ARCHS]
            for name, t in (("Correct-option labels", "gold"), ("Teacher's top choice", "teacher_hard"),
                            ("Teacher probabilities (FunDM)", "teacher"))] + [
        ["Full model (teacher), 36 layers", f(statistics.mean(teacher().values()))]]


def table_6():
    km = kd_means("kd")
    return [[f"{L}" + (" (two-thirds)" if L == MAIN else ""), pcell(params(L)), f"{L / 36:.0%}",
             f(km["linear", L]), f(km["comparator", L])] for L in LAYERS] + [
        ["36 + LM head (teacher)", pcell(TOTAL), "100%", f(statistics.mean(teacher().values()))]]

def table_1():
    kd, (pl, pc), p36 = kd_runs("kd"), head_params("kd", "mmlu_pro"), params(36)
    shaped = {s: j(f"results/clm_shaped_sets/head_{s}_apart.json")["head"]["accuracy"] for s in SETS}
    cols = [clm(), clmft(), shaped,
            {s: kd["linear", MAIN, s]["student"]["accuracy"] for s in SETS},
            {s: kd["comparator", MAIN, s]["student"]["accuracy"] for s in SETS}, teacher()]
    return ([[n] + [f(c[s]) for c in cols] for s, n in zip(SETS, NAMES)]
            + [["Mean"] + [f(statistics.mean(c.values())) for c in cols],
               ["Total parameters"] + [bcount(x) for x in (p36 + CLM_HEAD, p36 + CLM_HEAD, p36 + CLM_HEAD,
                                                           params(MAIN) + pl, params(MAIN) + pc, TOTAL)],
               ["Trainable parameters"] + [mcount(x) for x in (CLM_HEAD, CLM_HEAD, CLM_HEAD, pl, pc, 0)],
               ["Layers run (of 36)", "36", "36", "36", str(MAIN), str(MAIN), "36 + LM head"]])


def table_ce_sets():
    C, T = {x: ce()[x]["head"]["accuracy"] for x in SETS}, teacher()
    return [[n, f(C[x]), f(T[x])] for x, n in zip(SETS, NAMES)] + [
        ["Mean", f(statistics.mean(C.values())), f(statistics.mean(T.values()))]]


def qps(s: str, m: str) -> str:
    """Batched MMLU-Pro-style throughput from the timing run, one decimal."""
    return f(j(TIMING_REPORT)["sets"][s]["batched"][m]["questions_per_s"], 1)


def table_3():
    return [[name, p] + [pm([arm_acc(c, a, s) for s in MCQ_SEEDS], 3) for c in ("clm", "question_option", "joint")]
            for name, a, p in (("fohce", "linear", "1.05M"), ("fohcce", "comparator", "5.38M"), ("CLM-shaped MLPs", "clm_mlp", "18.9M"))] + [
        ["Released CLM (pretrained)", "18.9M", f(j("results/clm_mmlu_pro.json")["clm"]["accuracy"]), "--", "--"]]


def table_2():
    cf, (pl, pc), p36 = j("results/kd_cf/summary.json"), head_params("kd_cf", "cf_test"), params(36)
    agent = {k: [j(f"results/{d}/B_agent_{a}_s{s}.json")["agent"] for s in SEEDS]
             for k, d, a in (("clm_shaped", "joint_agentonly", "clm_mlp"),)}
    hm = {a: cf["heads"][a][str(MAIN)] for a in ARCHS}
    # Released CLM follows its authors' mean-cosine rule, as in the paper.
    rc = j("results/released_clm_agent/coderforge.json")["clm_own_cosine"]
    ft = [j(f"results/released_clm_agent/clm_ft_s{s}.json")["clm_own_cosine"] for s in SEEDS]
    return [["Released CLM, no training†", bcount(p36 + CLM_HEAD), mcount(CLM_HEAD), f(rc["picks"], 1), f(rc["auc"])],
            ["Released CLM, fine-tuned†", bcount(p36 + CLM_HEAD), mcount(CLM_HEAD), pm([r["picks"] for r in ft], 1),
             pm([r["auc"] for r in ft], 3)]] + [
        [name, bcount(p36 + hp), mcount(hp), pm([r["picks"] for r in agent[k]], 1), pm([r["auc"] for r in agent[k]], 3)]
        for name, k, hp in (("CLM-shaped, action-trained", "clm_shaped", CLM_HEAD),)] + [
        [name, bcount(params(MAIN) + hp), mcount(hp), pm(hm[a]["picks"], 1), pm(hm[a]["auc"], 3)]
        for name, a, hp in ((f"fohkd, layer {MAIN}", "linear", pl), (f"fohckd, layer {MAIN}", "comparator", pc))] + [
        ["Full-model judge (teacher)", bcount(TOTAL), "0", str(cf["teacher"]["picks"]), f(cf["teacher"]["auc"])]]


def figure_4():
    """Means and sample standard deviations for the two panels of Figure 4."""
    cf = j("results/kd_cf/summary.json")
    return [[str(L)] + [pm(cf["heads"][a][str(L)][k], d)
            for k, d in (("picks", 1), ("auc", 3)) for a in ARCHS] for L in LAYERS] + [
        ["Full-model judge (36 layers + lm_head)", str(cf["teacher"]["picks"]), f(cf["teacher"]["auc"])]]

def table_4():
    g, lin = grid(), ce()["mmlu_pro"]["head"]
    return [["fohce (seed 0)", "1.05M"] + ce_row("", lin, "")[1:4],
            ["fohce (seeds 0, 1, 2)", "1.05M", pm([lin["accuracy"], g["linear_s1"]["accuracy"], g["linear_s2"]["accuracy"]], 3), "--", "--"],
            ["fohcce", "5.38M"] + ce_row("", cmp_ce()["mmlu_pro"]["head"], "")[1:4],
            ["CLM-shaped MLPs", "18.9M"] + ce_row("", g["joint_clm_mlp"], "")[1:4]]


def table_7():
    kd = kd_runs("kd")
    return [[name] + [f(kd[a, L, "mmlu_pro"]["best_val_kl"]) for L in LAYERS] for name, a in (("fohkd", "linear"), ("fohckd", "comparator"))]


def table_8():
    km, bm = kd_means("kd"), kd_means("kd_big")
    return [[str(L), f(km["linear", L]), f(bm["linear", L]), f(km["comparator", L]), f(bm["comparator", L])] for L in LAYERS] + [
        ["36 + LM head (teacher)", f(statistics.mean(teacher().values()))]]


def figure_3():
    """Exact accuracy and displayed retention: Figure 3 normalizes reported three-decimal scores."""
    kd, T = kd_runs("kd"), teacher()
    return [[f"{name}: {n}"] + [
        f"{f(kd[a, L, s]['student']['accuracy'])} ({100 * float(f(kd[a, L, s]['student']['accuracy'])) / float(f(T[s])):.1f}%)"
        for L in LAYERS] + [f(T[s])]
        for name, a in (("fohkd", "linear"), ("fohckd", "comparator")) for s, n in zip(SETS, NAMES)]

def table_10():
    rows = [("CLM, all texts batched", "36", "clm"), ("CLM", "36", "clm_cached"), ("CLM, options prepared", "36", "clm_cached_warm_options"), ("fohce", "36", "linear"),
            ("Full model (teacher)", "36 + LM head", "kd_teacher")] + [
        ("fohkd", str(L), f"kd_linear_L{L}") for L in reversed(LAYERS)]
    return [[name, layers] + [qps(s, m) for s in SETS] for name, layers, m in rows]


def table_9():
    ds, rep = ds_steps(), j("results/deepswe_clm_repro.json")
    return [["Random", f(rep["random_pick"] * 38, 1), pct(rep["random_pick"])]] + [
        [name, pm(ds[a], 1), pct(statistics.mean(ds[a]) / 38)]
        for name, a in (("CLM-shaped, from scratch", "clm_mlp"), ("fohce (read separately)", "linear"),
                        ("fohcce (read separately)", "comparator"))] + [
        ["CLM, released DeepSWE head", str(rep["selectors"]["clm:last_12_mean"]["resolved"]), pct(rep["selectors"]["clm:last_12_mean"]["rate"])],
        ["Oracle", str(round(rep["oracle_any"] * 38)), pct(rep["oracle_any"])]]


ALL_KD = [f"kd_{a}" for a in ARCHS]
CLM_FT = [f"clm_ft_s{i}" for i in range(3)]
TABLES_OLD = {  # original paper number -> (steps, row builder); historical result paths are retained
    1: (ALL_KD + ["clm_mmlu_pro", "clm_sets", "clm_shaped_sets"] + CLM_FT, table_1),
    2: (["cf_summary", "cf_released_clm"] + [f"cf_clm_ft_eval_s{s}" for s in SEEDS]
        + [f"joint_clm_mlp_s{s}" for s in SEEDS], table_2),
    3: (ALL_KD + KD_TARGET, table_5),
    4: (ALL_KD + ["ce_linear"], table_ce_sets),
    5: (GRID + ["clm_mmlu_pro"], table_3),
    6: (["ce_linear", "ce_comparator", "grid_together_linear_s1", "grid_together_linear_s2", "grid_together_clm_mlp"], table_4),
    7: (ALL_KD, table_6),
    8: (ALL_KD, table_7),
    9: (ALL_KD + [f"kd_big_{a}" for a in ARCHS], table_8),
    10: ([f"ds_steps_{a}_s{s}" for a in ("linear", "comparator", "clm_mlp") for s in SEEDS] + ["ds_clm"], table_9),
    11: ([], table_10),
}


def serving_models():
    if os.path.isdir(SERVING_REPORT):
        return {name:j(os.path.join(SERVING_REPORT,name+".json")) for name in ("clm","teacher","fundm","fundm_1t","ce","ce_1t")}
    return j(SERVING_REPORT)["models"]


def table_efficiency():
    data = serving_models()
    rows = [("CLM (released)","clm",params(36)+CLM_HEAD),
            ("Full model + LM head","teacher",TOTAL),
            ("foh","fundm",params(MAIN)+1_048_576),
            ("fohc","fundm_1t",params(MAIN)+5_383_168),
            ("fohce (ablation)","ce",params(36)+1_048_576),
            ("fohcce (ablation)","ce_1t",params(36)+5_383_168)]
    return [[name,str(data[m]["layers"]),bcount(p),
             f(data[m]["memory_throughput"]["cuda_peak_reserved_bytes"]/2**30,1),
             f(data[m]["throughput_median"],1),f(data[m]["first_output_median_ms"],1),
             f"{data[m]['throughput_median']/data['teacher']['throughput_median']:.2f}x"] for name,m,p in rows]


TABLES = {n if n == 1 else n+1: v for n,v in TABLES_OLD.items()}
TABLES[2] = ([],table_efficiency)


FIGURES = {3: (ALL_KD, figure_3), 4: (["cf_summary"], figure_4)}


# ---------------------------------------------------------------- numbers in the text
def text_claims() -> dict:
    """Claim -> (steps, value as printed). Values are computed lazily, so a script evaluates only the claims it needs."""
    km, bm = (lambda: kd_means("kd")), (lambda: kd_means("kd_big"))
    kd, T = (lambda: kd_runs("kd")), teacher
    TM = lambda: statistics.mean(T().values())  # noqa: E731
    rng = lambda a, b: f"{round(100 * min(a, b))}--{round(100 * max(a, b))}%"  # noqa: E731
    ret = lambda a, L: km()[a, L] / TM()  # noqa: E731
    cf = lambda: j("results/kd_cf/summary.json")  # noqa: E731
    vt = lambda: cf()["vs_teacher"]  # noqa: E731
    span = lambda key: f"{min(min(v[key]) for v in vt().values())}--{max(max(v[key]) for v in vt().values())}"  # noqa: E731
    nll = lambda: {s: (kd()["comparator", MAIN, s]["student"]["nll"], kd()["comparator", MAIN, s]["teacher"]["nll"]) for s in SETS}  # noqa: E731
    margins = lambda: [kd()[a, MAIN, s]["student"]["accuracy"] - clmft()[s] for a in ARCHS for s in SETS]  # noqa: E731
    mass = lambda: {s: kd()["linear", MAIN, s]["teacher_letter_mass"] for s in SETS}  # noqa: E731
    pv = lambda: j("results/teacher_prompt_val.json")  # noqa: E731
    hist = lambda d: [e["val"]["nll"] for e in d["history"]]  # noqa: E731
    tc = lambda: j(TIMING_REPORT)["sets"]  # noqa: E731
    qps = lambda s, m: tc()[s]["batched"][m]["questions_per_s"]  # noqa: E731
    span2 = lambda xs: (lambda lo, hi: lo if lo == hi else f"{lo}--{hi}")(f"{min(xs):.2f}", f"{max(xs):.2f}")  # noqa: E731
    ratio = lambda num, den: span2([qps(s, num) / qps(s, den) for s in tc()])  # noqa: E731
    grid_d = lambda: [arm_mean("joint", a) - arm_mean("clm", a) for a in HEAD_ARGS]  # noqa: E731
    tgt = lambda t, a: statistics.mean(target_mean(t, a, s) for s in MCQ_SEEDS)  # noqa: E731
    span1 = lambda xs: f"{100 * min(xs):.1f}--{100 * max(xs):.1f}"  # noqa: E731
    span0 = lambda xs: f"{round(100 * min(xs))}--{round(100 * max(xs))}"  # noqa: E731
    ds_means = lambda: [statistics.mean(v) for v in ds_steps().values()]  # noqa: E731
    word = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"]
    KD, BIG, CF, CLMS = ALL_KD, [f"kd_big_{a}" for a in ARCHS], ["cf_summary"], ["clm_mmlu_pro", "clm_sets"]
    DS = TABLES[11][0]
    ADP, PV = ["cf_adp_overlap"], ["teacher_prompt_val"]
    adp = lambda: j("results/released_clm_agent/adp_overlap.json")["subsets"]["coderforge_preview"]["test"]  # noqa: E731
    return {
        "retention at 24 layers, Linear": (KD, lambda: pct(ret("linear", MAIN))),
        "retention at 24 layers, +1T": (KD, lambda: pct(ret("comparator", MAIN))),
        "retention at 20 layers": (KD, lambda: rng(ret("linear", 20), ret("comparator", 20))),
        "retention at 16 layers": (KD, lambda: rng(ret("linear", 16), ret("comparator", 16))),
        "28-layer means": (KD, lambda: f"{f(km()['linear', 28])} and {f(km()['comparator', 28])}"),
        "parameter saving at 24 layers": ([], lambda: f"{1 - params(MAIN) / TOTAL:.0%}"),
        "layer saving at 24 layers": ([], lambda: f"{1 - MAIN / 36:.0%}"),
        "margin over CLM ft., smallest": (KD + CLM_FT, lambda: f"{100 * min(margins()):.1f}"),
        "margin over CLM ft., largest": (KD + CLM_FT, lambda: f"{100 * max(margins()):.1f}"),
        "CLM ft. mean": (CLM_FT, lambda: f(statistics.mean(clmft().values()))),
        "CE mean": (["ce_linear"], lambda: f(statistics.mean(c["head"]["accuracy"] for c in ce().values()))),
        "+1T MMLU-Pro NLL vs teacher": (KD, lambda: f"{f(nll()['mmlu_pro'][0], 2)} versus {f(nll()['mmlu_pro'][1], 2)}"),
        "sets where +1T NLL < teacher": (KD, lambda: word[sum(x < y for x, y in nll().values())] + " of eight"),
        "mean agreement": (KD, lambda: f"{100 * statistics.mean(statistics.mean(kd()[a, MAIN, s]['agree_with_teacher'] for s in SETS) for a in ARCHS):.0f}%"),
        "MMLU-Pro agreement": (KD, lambda: rng(*(kd()[a, MAIN, "mmlu_pro"]["agree_with_teacher"] for a in ARCHS))),
        "judge tasks": (CF, lambda: f"solves {cf()['teacher']['picks']} tasks"),
        "random choice": (CF, lambda: f(cf()["random"], 1)),
        "Linear vs +1T AUC at 24 layers": (CF, lambda: f"{f(statistics.mean(cf()['heads']['linear'][str(MAIN)]['auc']))} and "
                                                       f"{f(statistics.mean(cf()['heads']['comparator'][str(MAIN)]['auc']))}"),
        "+1T task interval": (CF, lambda: f"{vt()['comparator']['picks_ci'][0]:+.1f} to {vt()['comparator']['picks_ci'][1]:+.1f}".replace("+-", "-")),
        "+1T AUC interval": (CF, lambda: f"{vt()['comparator']['auc_ci'][0]:+.3f} to {vt()['comparator']['auc_ci'][1]:+.3f}"),
        "Linear task interval": (CF, lambda: f"{vt()['linear']['picks_ci'][0]:+.1f} to {vt()['linear']['picks_ci'][1]:+.1f}"),
        "different picks": (CF, lambda: span("swaps")),
        "head right, judge wrong": (CF, lambda: span("head_right_judge_wrong")),
        "judge right, head wrong": (CF, lambda: span("judge_right_head_wrong")),
        "near-tie tasks": (CF, lambda: f"{vt()['linear']['judge_top2_gap_under_05']} tasks"),
        "judge says yes": (CF, lambda: f"{cf()['test']['judge_says_yes']:.0%}"),
        "options-prompt AUC": (CF, lambda: f(cf()["options_prompt"]["auc"])),
        "letter mass, agents": (CF, lambda: pct(cf()["test"]["letter_mass"])),
        "yes/no mass": (CF, lambda: pct(cf()["judge_yes_no_mass"])),
        "joint-encoding gain": (TABLES[6][0], lambda: f"{round(100 * min(grid_d()))}--{round(100 * max(grid_d()))}"),
        "question gain": (TABLES[6][0], lambda: span0([arm_mean("question_option", a) - arm_mean("clm", a) for a in HEAD_ARGS])
                                                 + " points over option only"),
        "question + option accuracy": (TABLES[6][0], lambda: "--".join(f(x) for x in (lambda v: (min(v), max(v)))(
            [arm_mean("question_option", a) for a in HEAD_ARGS]))),
        "other options cost": (TABLES[6][0], lambda: span1([arm_mean("question_option", a) - arm_mean("joint", a) for a in HEAD_ARGS])
                                                      + " points lower"),
        "pass cost": (["pass_cost"], lambda: (lambda c: f"one pass of {c['full_list']['tokens']:.0f} tokens per question, against "
                                                        f"{c['question_option']['passes']:.1f} passes and {c['question_option']['tokens']:.0f} tokens")(
            j("results/pass_cost.json"))),
        "labels on KD states": (TABLES[4][0], lambda: f"{f(tgt('gold', 'linear'))} for foh and {f(tgt('gold', 'comparator'))} for fohc"),
        "soft over hard targets": (TABLES[4][0], lambda: f"{span1([tgt('teacher', a) - tgt('teacher_hard', a) for a in ARCHS])} points above the"),
        "soft over label targets": (TABLES[4][0], lambda: f"{span1([tgt('teacher', a) - tgt('gold', a) for a in ARCHS])} points above correct labels"),
        "KL at 24 layers": (KD, lambda: f"{f(kd()['comparator', MAIN, 'mmlu_pro']['best_val_kl'])} versus {f(kd()['linear', MAIN, 'mmlu_pro']['best_val_kl'])}"),
        "KL at 32 layers": (KD, lambda: f"{f(kd()['comparator', 32, 'mmlu_pro']['best_val_kl'])} versus {f(kd()['linear', 32, 'mmlu_pro']['best_val_kl'])}"),
        "MMLU-Pro retention at 20 layers": (KD, lambda: rng(*(kd()[a, 20, "mmlu_pro"]["student"]["accuracy"] / T()["mmlu_pro"] for a in ARCHS))),
        "data, 24 layers": (KD + BIG, lambda: f"{100 * (bm()['linear', MAIN] - km()['linear', MAIN]):.1f} points for fohkd and "
                                              f"{100 * (bm()['comparator', MAIN] - km()['comparator', MAIN]):.1f} for fohckd"),
        "data, 20 layers": (KD + BIG, lambda: f"{100 * (bm()['linear', 20] - km()['linear', 20]):.1f} and {100 * (bm()['comparator', 20] - km()['comparator', 20]):.1f}"),
        "data, 16 layers Linear": (KD + BIG, lambda: f"loses about {100 * (km()['linear', 16] - bm()['linear', 16]):.1f}"),
        "data, 16 layers +1T": (KD + BIG, lambda: f"gains about {100 * (bm()['comparator', 16] - km()['comparator', 16]):.1f}"),
        "prompt validation rows": (PV, lambda: f"{pv()['n_validation_rows']} held-out validation rows"),
        "prompt validation chance": (PV, lambda: f"chance {f(pv()['chance'])}"),
        "teacher validation accuracy": (PV, lambda: f"{f(pv()['multi_line']['accuracy'])} with the multi-line prompt and "
                                                    f"{f(pv()['single_line']['accuracy'])} with a single-line prompt"),
        "prompt validation gap": (PV, lambda: f"{100 * pv()['difference']:.1f}-point difference (95% bootstrap interval "
                                              f"{100 * pv()['difference_ci95'][0]:.1f} to {100 * pv()['difference_ci95'][1]:.1f})"),
        "original option order": (["qwen_readouts"], lambda: pct(j("results/qwen_mmlu_pro.json")["letter"]["accuracy"])),
        "shuffled order": (KD, lambda: pct(T()["mmlu_pro"])),
        "letter mass, MMLU-Pro": (KD, lambda: f"{mass()['mmlu_pro']:.0%}"),
        "letter mass, other sets": (KD, lambda: f"{round(100 * min(v for k, v in mass().items() if k != 'mmlu_pro'))}--"
                                                f"{round(100 * max(v for k, v in mass().items() if k != 'mmlu_pro'))}%"),
        "repeated questions": (["overlap"], lambda: f"{j('results/split_overlap.json')['seed0/question_split/question']['duplicate_groups']} repeated"),
        "CLM fitting rows": (CLM_FT, lambda: f"{ft_seeds()[0]['train_questions']} fitting rows"),
        "CE epoch, Linear": (["ce_linear"], lambda: f"epoch {hist(ce()['mmlu_pro']).index(min(hist(ce()['mmlu_pro']))) + 1}"),
        "CE epoch, +1T": (["ce_comparator"], lambda: f"epoch {hist(cmp_ce()['mmlu_pro']).index(min(hist(cmp_ce()['mmlu_pro']))) + 1}"),
        "CE NLL rise, Linear": (["ce_linear"], lambda: f"{f(min(hist(ce()['mmlu_pro'])))} to {f(hist(ce()['mmlu_pro'])[-1], 2)}"),
        "CE NLL rise, +1T": (["ce_comparator"], lambda: f"{f(min(hist(cmp_ce()['mmlu_pro'])))} to {f(hist(cmp_ce()['mmlu_pro'])[-1])}"),
        "BoolQ before CLM fine-tuning": (CLM_FT, lambda: pct(statistics.mean(d["init_test"]["per_question"]["boolq"] for d in ft_seeds()))),
        "BoolQ after CLM fine-tuning": (CLM_FT, lambda: pct(clmft()["boolq"])),
        "BoolQ released Engine": (CLMS, lambda: pct(clm()["boolq"])),
        "BoolQ always yes": (["ce_linear"], lambda: (lambda yes: pct(sum(g == 1 for g in yes) / len(yes)))([r["gold"] for r in map(json.loads, open("results/head_boolq_preds.jsonl"))])),
        "CE over CLM": ([], lambda: ratio("linear", "clm_cached")),
        "KD 24 over teacher": ([], lambda: ratio(f"kd_linear_L{MAIN}", "kd_teacher")),
        "KD 20 over teacher": ([], lambda: ratio("kd_linear_L20", "kd_teacher")),
        "KD 24 over CLM": ([], lambda: ratio(f"kd_linear_L{MAIN}", "clm_cached")),
        "+1T versus linear": ([], lambda: f"within {100 * max(abs(qps(s, f'kd_comparator_L{L}') / qps(s, f'kd_linear_L{L}') - 1) for s in tc() for L in LAYERS):.1f}%"),
        "KD prompt tokens": ([], lambda: f"{tc()['mmlu_pro']['mean_tokens']['kd']:.0f} versus {tc()['mmlu_pro']['mean_tokens']['ours']:.0f} tokens"),
        "DeepSWE fitting steps": (DS, lambda: str(j("results/deepswe_steps/linear_s0.json")["train_steps"])),
        "DeepSWE validation steps": (DS, lambda: str(j("results/deepswe_steps/linear_s0.json")["val_steps"])),
        "DeepSWE mixed tasks": (DS, lambda: f"Only {j('results/deepswe_clm_repro.json')['mixed_tasks']} tasks can"),
        "DeepSWE CLM lead": (DS, lambda: f"{31 - max(ds_means()):.1f}--{31 - min(ds_means()):.1f} tasks"),
        "DeepSWE seed spread": (DS, lambda: f"variation reaches {max(statistics.stdev(v) for v in ds_steps().values()):.1f} tasks"),
        "released CLM on CoderForge": (["cf_released_clm"], lambda: (lambda r: f"solves {f(r['picks'], 1)} tasks with AUC {f(r['auc'])}")(
            j("results/released_clm_agent/coderforge.json")["ours"])),
        "released CLM, own cosine": (["cf_released_clm"], lambda: (lambda r: f"gives {f(r['picks'], 1)} tasks and AUC {f(r['auc'])}")(
            j("results/released_clm_agent/coderforge.json")["clm_own_cosine"])),
        "released CLM fine-tuned, own cosine": ([f"cf_clm_ft_eval_s{s}" for s in SEEDS], lambda: (lambda r: (
            f"gives {pm([x['picks'] for x in r], 1)} tasks and AUC {pm([x['auc'] for x in r], 3)}"))(
            [j(f"results/released_clm_agent/clm_ft_s{s}.json")["clm_own_cosine"] for s in SEEDS])),
        "CE action-trained, base": ([f"joint_linear_s{s}" for s in SEEDS], lambda: (lambda r:
            f"solves {pm([x['picks'] for x in r], 1)} tasks with AUC {pm([x['auc'] for x in r], 3)}")(
                [j(f"results/joint_agentonly_a/B_agent_linear_s{s}.json")["agent"] for s in SEEDS])),
        "CE action-trained, +1T": ([f"joint_comparator_s{s}" for s in SEEDS], lambda: (lambda r:
            f"solves {pm([x['picks'] for x in r], 1)} tasks with AUC {pm([x['auc'] for x in r], 3)}")(
                [j(f"results/joint_agentonly_a/B_agent_comparator_s{s}.json")["agent"] for s in SEEDS])),
        "ADP passing test runs": (ADP, lambda: f"{adp()['passing_in_adp']} of the {adp()['passing_runs']} passing runs"),
        "ADP test tasks": (ADP, lambda: f"covering {adp()['tasks_in_adp']} of the {adp()['tasks']} tasks"),
        "ADP failing test runs": (ADP, lambda: f"{adp()['failing_in_adp'] or 'none'} of the {adp()['failing_runs']} failing runs"),
    }


# ---------------------------------------------------------------- reporting
TIMING_REPORT = os.path.join(ROOT, "assets/timing-data.json")
SERVING_REPORT = os.path.join(ROOT, "assets/serving-data.json")
EXPECTED = j(os.path.join(ROOT, "expected.json"))
SHOW = {"foh": "FunDM", "fohc": "FunDM+1T", "fohce": "FunDM-CE", "fohcce": "FunDM-CE+1T",
        "fohkd": "FunDM", "fohckd": "FunDM+1T"}


def show(cell: str) -> str:
    return re.sub(r"\bfoh\w*\b", lambda m: SHOW.get(m.group(0), m.group(0)), cell)


# Cells from runs that are not bit-deterministic (row label or None = every row, columns): released CLM (vLLM) and
# the DeepSWE comparator (GPU attention). Small rerun differences in these count as a match.
NOISY_CELLS = {1: (None, {1}), 6: ("Released CLM (pretrained)", {2}), 11: ("fohcce (read separately)", {1, 2})}


def check_table(n: int, build: bool = True) -> int:
    """Print Table n and compare every cell with the paper; returns the number of differing rows."""
    steps, rows_of = TABLES[n]
    if build:
        ensure(*steps)
    exp = EXPECTED["tables"][str(n)]
    got = rows_of()
    label, cols = NOISY_CELLS.get(n, ("", set()))
    width = [max(len(show(r[i])) for r in got if i < len(r)) for i in range(max(map(len, got)))]
    print(f"\nTable {n} ({exp['label']})")
    bad = 0
    for g, e in zip(got, exp["rows"]):
        diff = {i for i, (x, y) in enumerate(zip(g, e)) if x != y} | ({-1} if len(g) != len(e) else set())
        clm_ok = diff and (label is None or g[0] == label) and diff <= cols and all(near_text(g[i], e[i]) for i in diff)
        miss = bool(diff) and not clm_ok
        bad += miss
        print(("!! " if miss else "   ") + "  ".join(show(c).ljust(w) for c, w in zip(g, width)))
        if miss:
            print("   paper: " + "  ".join(show(c) for c in e))
    bad += abs(len(got) - len(exp["rows"]))
    print(f"Table {n}: {'matches the paper' if not bad else f'{bad} rows differ from the paper'}")
    return bad


def check_figure(n: int, build: bool = True) -> int:
    """Print and verify plotted values, including exact source accuracies and seed SDs."""
    steps, rows_of = FIGURES[n]
    if build:
        ensure(*steps)
    exp, got = EXPECTED["figures"][str(n)], rows_of()
    width = [max(len(show(r[i])) for r in got if i < len(r)) for i in range(max(map(len, got)))]
    print(f"\nFigure {n} data ({exp['label']})")
    bad = 0
    for g, e in zip(got, exp["rows"]):
        miss = g != e
        bad += miss
        print(("!! " if miss else "   ") + "  ".join(show(c).ljust(w) for c, w in zip(g, width)))
        if miss:
            print("   paper: " + "  ".join(show(c) for c in e))
    bad += abs(len(got) - len(exp["rows"]))
    print(f"Figure {n}: {'matches the paper' if not bad else f'{bad} rows differ from the paper'}")
    return bad


NOISY_TEXT = {"BoolQ released Engine", "DeepSWE CLM lead", "DeepSWE seed spread"}
NUM = r"[-+]?\d+(?:\.\d+)?"


def near_text(a: str, b: str) -> bool:
    """Same words; decimals within 5 in the last printed digit, counts within 1%."""
    xs, ys = re.findall(NUM, a), re.findall(NUM, b)
    if re.sub(NUM, "#", a) != re.sub(NUM, "#", b) or len(xs) != len(ys):
        return False
    tol = lambda y: 5 * 10.0 ** -len(y.split(".")[1]) if "." in y else 0.01 * abs(float(y))  # noqa: E731
    return all(abs(float(x) - float(y)) <= tol(y) + 1e-9 for x, y in zip(xs, ys))


def check_text(names, build: bool = True) -> int:
    """Print named diagnostics and compare them with the saved references; returns the number that differ."""
    claims, paper = text_claims(), dict(EXPECTED["text"])
    if build:
        ensure(*sorted({s for n in names for s in claims[n][0]}))
    bad = 0
    for n in names:
        value = claims[n][1]()
        ok = value == paper[n]
        clm_ok = not ok and n in NOISY_TEXT and near_text(value, paper[n])
        bad += not (ok or clm_ok)
        print(f"{'   ' if ok or clm_ok else '!! '}{n}: {value}" + ("" if ok or clm_ok else f"   (paper: {paper[n]})"))
    return bad


def finish(bad: int) -> None:
    print("all match" if not bad else f"{bad} differences")
    sys.exit(1 if bad else 0)


TITLES_OLD = {
    1: "accuracy on eight benchmarks: FunDM, CLM baselines and the teacher",
    2: "CoderForge selection: FunDM, CLM baselines and the teacher",
    3: "matched layer-24 training targets: labels, hard teacher targets, teacher probabilities",
    4: "original layer-36 supervised CE recipe across eight benchmarks",
    5: "CE encoding ablation: option only, question plus option, and full list",
    6: "CE head architecture on joint final-layer states",
    7: "FunDM accuracy versus exit depth", 8: "FunDM validation KL by depth",
    9: "FunDM with the 38k versus 123k training pool",
    10: "DeepSWE supervised heads on separate released embeddings",
    11: "throughput of CLM, original CE and FunDM heads by depth"}
TITLES = {n if n == 1 else n+1: v for n,v in TITLES_OLD.items()}
TITLES[2] = "MMLU-Pro inference: released CLM, teacher, FunDM, FunDM+1T and both CE ablations"
FIGURE_TITLES = {3: "per-benchmark accuracy and teacher-relative retention", 4: "CoderForge depth sweep: means and seed SDs"}
TEXT = {"prompt": ["prompt validation rows", "prompt validation chance", "teacher validation accuracy", "prompt validation gap"],
        "option-order": ["original option order", "shuffled order"],
        "timing": ["CE over CLM", "KD 24 over teacher", "KD 20 over teacher", "KD 24 over CLM", "+1T versus linear",
                   "KD prompt tokens"],
        "overlap": ["repeated questions"],
        "controls": ["joint-encoding gain", "question gain", "question + option accuracy", "other options cost", "pass cost",
                     "labels on KD states", "soft over hard targets", "soft over label targets", "CE action-trained, base", "CE action-trained, +1T"],
        "clm-coderforge": ["released CLM on CoderForge", "released CLM, own cosine", "released CLM fine-tuned, own cosine",
                           "ADP passing test runs", "ADP test tasks", "ADP failing test runs"]}


def build_caches() -> None:
    S = ["train", *SETS]
    ensure("groups", "decontam", "kd_feats_extra", *(f"kd_feats_{s}" for s in S), "kd_oneline_feats_train", "teacher_prompt_val",
           *(f"ce_feats_{s}" for s in S), "ce_feats_question_option_train", "ce_feats_question_option_mmlu_pro",
           *(f"apart_feats_{s}" for s in S), "clm_choice_data",
           "cf_data", "cf_judge", "cf_feats_test", "cf_feats_train", "cf_holdout", "cf_embed", "cf_together", "ds_head", "ds_emb")
    print("all caches built")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="tables:\n" + "\n".join(f"  {n:>2}  {s}" for n, s in sorted(TITLES.items())) + "\nfigure data:\n" + "\n".join(f"  {n:>2}  {s}" for n, s in FIGURE_TITLES.items())
        + "\n\ntext numbers: " + ", ".join(TEXT))
    ap.add_argument("--serving-report", help="native vLLM report JSON, or directory containing six model JSON reports")
    ap.add_argument("--timing-report", help="check timing cells against a fresh extras.py timing report instead of the archived run")
    ap.add_argument("--table", type=int, nargs="+", choices=sorted(TABLES), metavar="N", help="reproduce Table N")
    ap.add_argument("--figure", type=int, nargs="+", choices=sorted(FIGURES), metavar="N", help="reproduce the plotted data of Figure N")
    ap.add_argument("--text", nargs="+", choices=sorted(TEXT), help="reproduce selected experiment diagnostics")
    ap.add_argument("--all", action="store_true", help="every table, plotted value and reference diagnostic")
    ap.add_argument("--caches", action="store_true", help="build every cache from scratch (about 3 GPU-days)")
    global SERVING_REPORT, TIMING_REPORT
    a = ap.parse_args()
    if a.serving_report:
        SERVING_REPORT = os.path.abspath(a.serving_report)
    if a.timing_report:
        TIMING_REPORT = os.path.abspath(a.timing_report)
    if not (a.table or a.figure or a.text or a.all or a.caches):
        ap.error("give --table, --figure, --text, --all or --caches")
    if a.caches:
        build_caches()
    bad = 0
    for n in sorted(TABLES) if a.all else a.table or []:
        bad += check_table(n)
    for n in sorted(FIGURES) if a.all else a.figure or []:
        bad += check_figure(n)
    names = [c for c, _ in EXPECTED["text"]] if a.all else [c for k in a.text or [] for c in TEXT[k]]
    if names:
        print()
        bad += check_text(names)
    if a.table or a.figure or a.text or a.all:
        finish(bad)


if __name__ == "__main__":
    main()
