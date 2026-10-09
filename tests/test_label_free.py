"""CPU checks for label-free fitting and inference; no downloaded model or datasets."""
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mcq


def features(labels=True):
    rng = torch.Generator().manual_seed(17)
    counts = [2 + i % 3 for i in range(40)]
    order = [list(reversed(range(k))) for k in counts]
    d = {"hq": torch.randn(40, 16, generator=rng).to(torch.bfloat16),
         "hk": torch.randn(sum(counts), 16, generator=rng).to(torch.bfloat16),
         "tl": torch.randn(sum(counts), generator=rng),
         "off": torch.tensor([0] + counts).cumsum(0), "order": order,
         "id": [f"row{i}" for i in range(40)], "category": ["test"] * 40,
         "letter_mass_mean": 0.8, "layer": 24, "model": "Qwen/Qwen3-8B", "prompt": "paper"}
    if labels:
        d["gold"] = [i % k for i, k in enumerate(counts)]
    return d


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, **kwargs):
        if text.startswith(" ") and len(text) == 2:
            return [ord(text[1]) + 1]
        return [ord(c) + 1 for c in text]

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) + 1 for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class LabelFreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fit(self, tmp, name, d, arch="linear", target="teacher", tests=""):
        base = Path(tmp) / name
        cache = base / "features"
        cache.mkdir(parents=True)
        torch.save(d, cache / "train_L24.pt")
        torch.save(features(False), cache / "evaluation_L24.pt")
        out = base / "results"
        args = ["--feats", str(cache), "--out", str(out), "--arch", arch,
                "--device", "cpu", "--seed", "3", "--epochs", "3", "--batch", "8",
                "--target", target, "--tests", tests]
        with contextlib.redirect_stdout(io.StringIO()):
            mcq.kd_train(args, primary=True)
        return torch.load(out / f"{arch}_L24_s3_head.pt", weights_only=False), out

    def test_removing_gold_preserves_updates_and_checkpoint_for_both_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            for arch in ("linear", "comparator"):
                labeled, _ = self.fit(tmp, f"{arch}-labeled", features(), arch)
                unlabeled, _ = self.fit(tmp, f"{arch}-unlabeled", features(False), arch)
                self.assertEqual(labeled["history"], unlabeled["history"])
                self.assertEqual(labeled["best_val_kl"], unlabeled["best_val_kl"])
                for key in labeled["state_dict"]:
                    self.assertTrue(torch.equal(labeled["state_dict"][key], unlabeled["state_dict"][key]), key)

    def test_hard_teacher_target_needs_no_gold_but_supervised_target_does(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fit(tmp, "hard", features(False), target="teacher_hard")
            with self.assertRaisesRegex(ValueError, "requires a correct-option label"):
                self.fit(tmp, "gold", features(False), target="gold")

    def test_accuracy_evaluation_is_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = self.fit(tmp, "optional", features(False), tests="evaluation")
            result = json.loads((out / "linear_L24_s3_evaluation.json").read_text())
            self.assertIsNone(result["student"])
            self.assertIsNone(result["teacher"])
            self.assertGreaterEqual(result["agree_with_teacher"], 0)

    def test_merging_unlabeled_and_labeled_caches_preserves_teacher_inputs(self):
        labeled = mcq.merge([features(), features()])
        mixed = mcq.merge([features(), features(False)])
        a = mcq.padded(labeled, "cpu", include_gold=False)
        b = mcq.padded(mixed, "cpu", include_gold=False)
        for left, right in zip(a[:4], b[:4], strict=True):
            self.assertTrue(torch.equal(left, right))
        self.assertIsNone(a[4])
        self.assertIsNone(b[4])

    def test_raw_prefix_prediction_matches_full_model_cached_intermediate_states(self):
        torch.manual_seed(11)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=256, hidden_size=16,
                                            intermediate_size=32, num_hidden_layers=3,
                                            num_attention_heads=2, num_key_value_heads=2,
                                            head_dim=8, use_cache=False)).to(torch.bfloat16).eval()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "input.jsonl"
            source.write_text(json.dumps({"id": "q", "question": "Choose a number.",
                                          "options": ["one", "two", "three"]}) + "\n")
            prefix = tmp / "cache"
            with patch.object(mcq.AutoTokenizer, "from_pretrained", return_value=Tokenizer()), \
                 patch.object(mcq.AutoModelForCausalLM, "from_pretrained", return_value=copy.deepcopy(model)), \
                 contextlib.redirect_stdout(io.StringIO()):
                mcq.kd_extract(["--input", str(source), "--out", str(prefix), "--layers", "1", "--device", "cpu"], primary=True)
            cache = torch.load(tmp / "cache_L1.pt", weights_only=False)
            self.assertNotIn("gold", cache)
            del cache["tl"]  # inference must not depend on teacher targets
            torch.save(cache, tmp / "cache_L1.pt")
            for arch in ("linear", "comparator"):
                head = mcq.MidHead(arch, 16)
                checkpoint = tmp / "head.pt"
                torch.save({"state_dict": head.state_dict(), "arch": arch, "hidden": 16,
                            "layer": 1, "model": "Qwen/Qwen3-8B", "prompt": "paper"}, checkpoint)
                with contextlib.redirect_stdout(io.StringIO()):
                    mcq.predict(["--head", str(checkpoint), "--feats", str(tmp / "cache_L1.pt"),
                                 "--out", str(tmp / "cached.jsonl"), "--device", "cpu"])
                with patch.object(mcq.AutoTokenizer, "from_pretrained", return_value=Tokenizer()), \
                     patch.object(mcq.AutoModel, "from_pretrained", return_value=copy.deepcopy(model.model)), \
                     contextlib.redirect_stdout(io.StringIO()):
                    mcq.predict(["--head", str(checkpoint), "--input", str(source),
                                 "--out", str(tmp / "raw.jsonl"), "--device", "cpu"])
                cached = json.loads((tmp / "cached.jsonl").read_text())
                raw = json.loads((tmp / "raw.jsonl").read_text())
                self.assertEqual(cached["choice"], raw["choice"])
                torch.testing.assert_close(torch.tensor(cached["probabilities"]),
                                           torch.tensor(raw["probabilities"]), rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
