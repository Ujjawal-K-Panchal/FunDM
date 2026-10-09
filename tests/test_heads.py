"""Head correctness checks on CPU and CUDA when available; no downloads required.

Permutation tests act on extracted states, not reordered backbone prompts.
Every tested row contains at least one real option, as required by scoring.
"""
import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from heads import OptionHead, ComparatorHead, CLMStyleHead


class HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def head(self, name, device):
        torch.manual_seed(271)
        constructors = {
            "linear": lambda: OptionHead(8, dim=4),
            "comparator": lambda: ComparatorHead(8, dim=4, width=16, heads=2, ff=32, dropout=0.1),
            "clm_mlp": lambda: CLMStyleHead(8, width=16, proj=4),
        }
        return constructors[name]().to(device).eval()

    def inputs(self, device, counts=(1, 2, 10, 17)):
        torch.manual_seed(103)
        q = torch.randn(len(counts), 8, device=device)
        k = torch.randn(len(counts), max(counts), 8, device=device)
        mask = torch.arange(max(counts), device=device)[None] < torch.tensor(counts, device=device)[:, None]
        return q, k, mask

    def test_option_scores_match_hand_calculation(self):
        for device in self.devices:
            with self.subTest(device=device):
                head = OptionHead(2, dim=2).to(device)
                with torch.no_grad():
                    head.w_q.weight.copy_(torch.tensor([[1., 2.], [0., -1.]], device=device))
                    head.w_k.weight.copy_(torch.tensor([[2., 0.], [1., 1.]], device=device))
                # q=[5,-2], keys=[6,7],[-2,4]: dot products 16,-18.
                q = torch.tensor([[1., 2.]], device=device)
                k = torch.tensor([[[3., 4.], [-1., 5.], [99., 99.]]], device=device)
                mask = torch.tensor([[True, True, False]], device=device)
                scores = head.score(q, k, mask)
                expected = torch.tensor([[16 / math.sqrt(2), -18 / math.sqrt(2)]], device=device)
                torch.testing.assert_close(scores[:, :2], expected)
                self.assertTrue(torch.isneginf(scores[:, 2]).all())

    def test_forward_gathers_correct_positions_and_ignores_padded_indices(self):
        for device in self.devices:
            with self.subTest(device=device):
                head = OptionHead(2, dim=2).to(device)
                with torch.no_grad():
                    head.w_q.weight.copy_(torch.eye(2, device=device))
                    head.w_k.weight.copy_(torch.eye(2, device=device))
                h = torch.tensor([[[1., 0.], [0., 2.], [1., 1.]],
                                  [[2., 0.], [1., 3.], [0., 1.]]], device=device)
                answer = torch.tensor([2, 1], device=device)
                positions = torch.tensor([[0, 1, 999], [2, -999, 999]], device=device)
                mask = torch.tensor([[True, True, False], [True, False, False]], device=device)
                actual = head(h, answer, positions, mask)
                expected = torch.tensor([[1., 2., -math.inf], [3., -math.inf, -math.inf]], device=device) / math.sqrt(2)
                torch.testing.assert_close(actual, expected)

    def test_variable_option_counts_and_probability_normalization(self):
        for device in self.devices:
            for name in ("linear", "comparator", "clm_mlp"):
                with self.subTest(device=device, head=name):
                    q, k, mask = self.inputs(device)
                    scores = self.head(name, device).score(q, k, mask)
                    self.assertEqual(tuple(scores.shape), (4, 17))
                    self.assertTrue(torch.isfinite(scores[mask]).all())
                    self.assertTrue(torch.isneginf(scores[~mask]).all())
                    p = scores.softmax(-1)
                    torch.testing.assert_close(p.sum(-1), torch.ones(4, device=device))
                    self.assertTrue((p[~mask] == 0).all())
                    self.assertEqual(p[0, 0].item(), 1.0)

    def test_padded_states_cannot_change_valid_scores(self):
        for device in self.devices:
            for name in ("linear", "comparator", "clm_mlp"):
                with self.subTest(device=device, head=name):
                    head = self.head(name, device)
                    q, k, mask = self.inputs(device)
                    changed = k.clone()
                    changed[~mask] = 1000 * torch.randn_like(changed[~mask])
                    a, b = head.score(q, k, mask), head.score(q, changed, mask)
                    torch.testing.assert_close(a[mask], b[mask], rtol=1e-5, atol=1e-5)

    def test_batch_matches_individual_unpadded_rows(self):
        for device in self.devices:
            for name in ("linear", "comparator", "clm_mlp"):
                with self.subTest(device=device, head=name):
                    head = self.head(name, device)
                    q, k, mask = self.inputs(device)
                    batched = head.score(q, k, mask)
                    for row in range(len(q)):
                        n = int(mask[row].sum())
                        individual = head.score(q[row:row+1], k[row:row+1, :n], mask[row:row+1, :n])
                        torch.testing.assert_close(batched[row, :n], individual[0], rtol=1e-4, atol=2e-5)

    def test_option_state_permutation_permutates_scores(self):
        for device in self.devices:
            for name in ("linear", "comparator", "clm_mlp"):
                with self.subTest(device=device, head=name):
                    head = self.head(name, device)
                    q, k, mask = self.inputs(device)
                    permutation = torch.tensor([16, 0, 7, 2, 14, 1, 10, 5, 3, 15, 4, 6, 8, 9, 11, 13, 12], device=device)
                    expected = head.score(q, k, mask)[:, permutation]
                    actual = head.score(q, k[:, permutation], mask[:, permutation])
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)

    def test_backward_is_finite_and_padding_has_zero_gradient(self):
        for device in self.devices:
            for name in ("linear", "comparator", "clm_mlp"):
                with self.subTest(device=device, head=name):
                    head = self.head(name, device).train()
                    q, k, mask = self.inputs(device, counts=(2, 3, 10, 17))
                    q.requires_grad_()
                    k.requires_grad_()
                    loss = -head.score(q, k, mask).log_softmax(-1)[:, 0].mean()
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for label, tensor in [("query", q), ("keys", k), *head.named_parameters()]:
                        self.assertIsNotNone(tensor.grad, label)
                        self.assertTrue(torch.isfinite(tensor.grad).all(), label)
                    self.assertTrue((k.grad[~mask] == 0).all())
                    self.assertGreater(q.grad.abs().sum().item(), 0)
                    self.assertGreater(k.grad[mask].abs().sum().item(), 0)
                    self.assertGreater(head.w_q.weight.grad.abs().sum().item() if name != "clm_mlp" else head.state[0].weight.grad.abs().sum().item(), 0)

    def test_bfloat16_states_agree_with_explicit_float_conversion(self):
        for device in self.devices:
            for name in ("linear", "comparator", "clm_mlp"):
                with self.subTest(device=device, head=name):
                    head = self.head(name, device)
                    q, k, mask = self.inputs(device)
                    q, k = q.bfloat16(), k.bfloat16()
                    torch.testing.assert_close(head.score(q, k, mask), head.score(q.float(), k.float(), mask), rtol=0, atol=0)

    def test_clm_cosine_scale_is_capped_at_100(self):
        for device in self.devices:
            with self.subTest(device=device):
                head = self.head("clm_mlp", device)
                q, k, mask = self.inputs(device)
                with torch.no_grad():
                    head.logit_scale.fill_(math.log(1000))
                    qs, ks = head.state(q), head.action(k)
                    reference = 100 * (qs[:, None] * ks).sum(-1) / (qs.norm(dim=-1)[:, None] * ks.norm(dim=-1))
                    actual = head.score(q, k, mask)
                torch.testing.assert_close(actual[mask], reference[mask], rtol=1e-5, atol=2e-5)
                self.assertTrue((actual[mask].abs() <= 100.0001).all())


if __name__ == "__main__":
    unittest.main()
