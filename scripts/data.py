"""Inputs: pinned Hugging Face revisions, the FunDM exit depth, the multiple-choice sets as (question, options, gold, category,
id) items, and the added training pool (RACE, MathQA, Cosmos QA, QASC, LogiQA; rows flagged by scripts/mcq.py decontam
dropped)."""
import json
import os
import random
import re

from datasets import load_dataset

MODELS = {
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "Contrastive-LM/CLM-v0.1-8B": "e939398d4556fcd9400c76fa8c5a513202f42b0a",
    # Released CLM DeepSWE head (downloaded to heads/deepswe, see README).
    "Contrastive-LM/deepswe-clm-heads-8k": "c60876f3fdf7a75dc58d33b776e469b7e903d0ee",
}

EXIT_LAYER = 24  # FunDM exit depth for every task: 24 of Qwen3-8B's 36 layers

DATASETS = {
    # Training pool.
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "allenai/sciq": "2c94ad3e1aafab77146f384e23536f97a4849815",
    "openlifescienceai/medmcqa": "91c6572c454088bf71b679ad90aa8dffcd0d5868",
    # Evaluation sets.
    "TIGER-Lab/MMLU-Pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "allenai/openbookqa": "388097ea7776314e93a529163e0fea805b8a6454",
    "tau/commonsense_qa": "94630fe30dad47192a8546eb75f094926d47e155",
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "baber/piqa": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
    "allenai/winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
    "google/boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
    # Added training data. math_qa, cosmos_qa and logiqa are script-based on main: these are parquet-branch commits.
    "ehovy/race": "2fec9fd81f1dc971569a9b729c43f2f0e6436637",
    "allenai/math_qa": "fafb9f7ee5b9ec4da9499f9c4177a4c91389f2d6",
    "allenai/cosmos_qa": "ed50fe83db62b355759053b1cd5eef344ee371e0",
    "allenai/qasc": "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070",
    "lucasmccabe/logiqa": "fa9f9918fa81eca088805c1395d7f592f7755ae0",
    # CoderForge runs (config trajectories, split SWE_Rebench), rebuilt by scripts/coderforge.py fetch.
    "togethercomputer/CoderForge-Preview": "060fca96cf723b2ebab3181e9e59fafd273df3cb",
    # CLM's released DeepSWE embeddings (converted into data/clm_emb by main.py).
    "tarsur385/deepswe-prm-embeddings-8k": "e57c9483f9cc09eb2328ff41b3aa9c0452dea513",
    # Agent Data Protocol collection, CLM's post-training source (overlap check, scripts/coderforge.py adp_overlap).
    "neulab/agent-data-collection": "31a76bfb0124d77ae7322eabbb0171bf11ee2c67",
}


def revision_for(repo_id: str) -> str | None:
    """Pinned revision of a model or dataset id; None leaves it unpinned."""
    return MODELS.get(repo_id, DATASETS.get(repo_id))



def arc() -> list[dict]:
    out = []
    for cfg in ("ARC-Easy", "ARC-Challenge"):
        for r in load_dataset("allenai/ai2_arc", cfg, split="train", revision=DATASETS["allenai/ai2_arc"]):
            labels = r["choices"]["label"]
            if r["answerKey"] in labels:
                out.append(dict(question=r["question"], options=list(r["choices"]["text"]),
                                gold=labels.index(r["answerKey"]), category="arc", id=r["id"]))
    return out


def sciq() -> list[dict]:
    out = []
    for i, r in enumerate(load_dataset("allenai/sciq", split="train", revision=DATASETS["allenai/sciq"])):
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        out.append(dict(question=r["question"], options=opts, gold=0, category="sciq", id=f"sciq-{i}"))
    return out


def medmcqa(n: int, seed: int) -> list[dict]:
    rows = load_dataset("openlifescienceai/medmcqa", split="train", revision=DATASETS["openlifescienceai/medmcqa"])
    idx = random.Random(seed).sample(range(len(rows)), n)
    out = []
    for i in idx:
        r = rows[i]
        if r["choice_type"] != "single":
            continue
        out.append(dict(question=r["question"], options=[r["opa"], r["opb"], r["opc"], r["opd"]],
                        gold=int(r["cop"]), category="medmcqa", id=r["id"]))
    return out


def mmlu_pro() -> list[dict]:
    return [dict(question=r["question"], options=list(r["options"]), gold=int(r["answer_index"]),
                 category=r["category"], id=str(r["question_id"]))
            for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test", revision=DATASETS["TIGER-Lab/MMLU-Pro"])]


def mmlu() -> list[dict]:
    return [dict(question=r["question"], options=list(r["choices"]), gold=int(r["answer"]),
                 category=r["subject"], id=f"mmlu-{i}")
            for i, r in enumerate(load_dataset("cais/mmlu", "all", split="test", revision=DATASETS["cais/mmlu"]))]


def _lettered(rows, qkey: str, tag: str) -> list[dict]:
    out = []
    for r in rows:
        labels = r["choices"]["label"]
        out.append(dict(question=r[qkey], options=list(r["choices"]["text"]), gold=labels.index(r["answerKey"]),
                        category=tag, id=str(r["id"])))
    return out


def openbookqa() -> list[dict]:
    return _lettered(load_dataset("allenai/openbookqa", "main", split="test", revision=DATASETS["allenai/openbookqa"]), "question_stem", "openbookqa")


def commonsenseqa() -> list[dict]:
    return _lettered(load_dataset("tau/commonsense_qa", split="validation", revision=DATASETS["tau/commonsense_qa"]), "question", "commonsenseqa")


def _hellaswag_text(text: str) -> str:
    """lm-eval-harness's HellaSwag cleanup."""
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def hellaswag() -> list[dict]:
    out = []
    for i, r in enumerate(load_dataset("Rowan/hellaswag", split="validation", revision=DATASETS["Rowan/hellaswag"])):
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        out.append(dict(question=_hellaswag_text(r["activity_label"] + ": " + ctx),
                        options=[_hellaswag_text(e) for e in r["endings"]], gold=int(r["label"]),
                        category="hellaswag", id=f"hellaswag-{i}"))
    return out


def piqa() -> list[dict]:
    return [dict(question=r["goal"], options=[r["sol1"], r["sol2"]], gold=int(r["label"]),
                 category="piqa", id=f"piqa-{i}")
            for i, r in enumerate(load_dataset("baber/piqa", split="validation", revision=DATASETS["baber/piqa"]))]


def winogrande() -> list[dict]:
    return [dict(question=r["sentence"], options=[r["option1"], r["option2"]], gold=int(r["answer"]) - 1,
                 category="winogrande", id=f"winogrande-{i}")
            for i, r in enumerate(load_dataset("allenai/winogrande", "winogrande_xl", split="validation", revision=DATASETS["allenai/winogrande"]))]


def boolq() -> list[dict]:
    """Options are ["no", "yes"]; passage and query are kept for CLM's yes/no question type."""
    return [dict(question=f"{r['passage']}\nQuestion: {r['question']}?", options=["no", "yes"],
                 gold=int(bool(r["answer"])), category="boolq", id=f"boolq-{i}",
                 passage=r["passage"], query=r["question"] + "?")
            for i, r in enumerate(load_dataset("google/boolq", split="validation", revision=DATASETS["google/boolq"]))]


EVAL_SETS = {"mmlu": mmlu, "openbookqa": openbookqa, "commonsenseqa": commonsenseqa, "hellaswag": hellaswag,
             "piqa": piqa, "winogrande": winogrande, "boolq": boolq}


def pad_with_borrowed(items: list[dict], share: float, k_max: int, seed: int) -> None:
    """Give a share of items extra wrong options borrowed from other items, up to k_max in total."""
    rng = random.Random(seed)
    pool = [o for it in items for j, o in enumerate(it["options"]) if j != it["gold"]]
    for it in items:
        if rng.random() < share and len(it["options"]) < k_max:
            extra = rng.randint(1, k_max - len(it["options"]))
            it["options"] = it["options"] + rng.sample(pool, extra)


# Added training pool (Table 8).
EXCLUDED = "feats/extra_excluded.json"


def race(n: int, seed: int) -> list[dict]:
    rows = load_dataset("ehovy/race", "all", split="train", revision=DATASETS["ehovy/race"])
    return [dict(question=r["article"].strip() + "\n\n" + r["question"].strip(), options=list(r["options"]),
                 gold="ABCD".index(r["answer"]), category="race", id=f"race-{r['example_id']}")
            for r in (rows[i] for i in random.Random(seed).sample(range(len(rows)), min(n, len(rows))))]


def mathqa() -> list[dict]:
    out = []
    for i, r in enumerate(load_dataset("allenai/math_qa", split="train", revision=DATASETS["allenai/math_qa"])):
        parts = re.split(r"(?:^|\s*,\s*)([a-e]) \) ", r["options"])  # ['', 'a', text, 'b', text, ...]
        opts = [t.strip() for t in parts[2::2]]
        if parts[1::2] != list("abcde")[:len(opts)] or len(opts) != 5:
            continue
        out.append(dict(question=r["Problem"].strip(), options=opts, gold="abcde".index(r["correct"]),
                        category="mathqa", id=f"mathqa-{i}"))
    return out


def cosmosqa() -> list[dict]:
    return [dict(question=r["context"].strip() + "\n\n" + r["question"].strip(),
                 options=[r[f"answer{j}"] for j in range(4)], gold=int(r["label"]), category="cosmosqa", id=f"cosmosqa-{i}")
            for i, r in enumerate(load_dataset("allenai/cosmos_qa", split="train", revision=DATASETS["allenai/cosmos_qa"]))]


def qasc() -> list[dict]:
    return [dict(question=r["question"], options=list(r["choices"]["text"]),
                 gold=r["choices"]["label"].index(r["answerKey"]), category="qasc", id=f"qasc-{r['id']}")
            for r in load_dataset("allenai/qasc", split="train", revision=DATASETS["allenai/qasc"])]


def logiqa() -> list[dict]:
    return [dict(question=r["context"].strip() + "\n\n" + r["query"].strip(), options=list(r["options"]),
                 gold=int(r["correct_option"]), category="logiqa", id=f"logiqa-{i}")
            for i, r in enumerate(load_dataset("lucasmccabe/logiqa", split="train", revision=DATASETS["lucasmccabe/logiqa"]))]


def sources(race_n: int, seed: int) -> list[dict]:
    return race(race_n, seed) + mathqa() + cosmosqa() + qasc() + logiqa()


def extra(seed: int, race_n: int = 15000) -> list[dict]:
    """Decontaminated pool, wrong options padded with borrowed ones like the base pool."""
    if not os.path.exists(EXCLUDED):
        raise SystemExit(f"{EXCLUDED} missing: run scripts/mcq.py decontam first")
    bad = set(json.load(open(EXCLUDED)))
    items = [it for it in sources(race_n, seed) if it["id"] not in bad]
    pad_with_borrowed(items, 0.5, 10, seed)
    return items
