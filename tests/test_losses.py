"""Independent loss/gradient checks on CPU and CUDA; no model downloads.

Valid inputs have at least one real option and positive target mass per row.
Padding logits are -inf; soft_ce's mask alone does not mask finite logits.
"""
import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from heads import soft_ce
from mcq import kl


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def tensor(self, values, device, requires_grad=False):
        return torch.tensor(values, device=device, dtype=torch.float64, requires_grad=requires_grad)

    def assert_scalar(self, actual, expected):
        self.assertTrue(torch.isfinite(actual))
        self.assertAlmostEqual(actual.item(), expected, places=9)

    def test_kl_matches_independent_probability_calculation(self):
        teacher = [[0.75, 0.25], [0.2, 0.8]]
        student = [[0.5, 0.5], [0.6, 0.4]]
        expected = sum(sum(p * math.log(p / q) for p, q in zip(t, s)) for t, s in zip(teacher, student)) / 2
        for device in self.devices:
            with self.subTest(device=device):
                t, s = self.tensor(teacher, device).log(), self.tensor(student, device).log()
                self.assert_scalar(kl(t, s), expected)
                self.assertGreater(abs(kl(t, s).item() - kl(s, t).item()), 0.01)

    def test_kl_identical_distributions_are_zero(self):
        for device in self.devices:
            with self.subTest(device=device):
                logits = self.tensor([[1., -2., -math.inf], [0., 3., 4.]], device)
                self.assert_scalar(kl(logits, logits), 0)

    def test_kl_padding_does_not_affect_value_or_gradient(self):
        for device in self.devices:
            with self.subTest(device=device):
                t = self.tensor([[math.log(0.7), math.log(0.3)]], device)
                s = self.tensor([[math.log(0.4), math.log(0.6)]], device, True)
                padded_t = torch.cat([t, self.tensor([[-math.inf, -math.inf]], device)], -1)
                padded_s = self.tensor([[math.log(0.4), math.log(0.6), -math.inf, -math.inf]], device, True)
                plain, padded = kl(t, s), kl(padded_t, padded_s)
                self.assert_scalar(padded, plain.item())
                plain.backward()
                padded.backward()
                torch.testing.assert_close(padded_s.grad[:, :2], s.grad, rtol=0, atol=1e-12)
                self.assertTrue(torch.isfinite(padded_s.grad).all())
                self.assertTrue((padded_s.grad[:, 2:] == 0).all())

    def test_kl_zero_teacher_probability_has_no_nan(self):
        for device in self.devices:
            with self.subTest(device=device):
                t = self.tensor([[0., -math.inf]], device)
                s = self.tensor([[math.log(0.25), math.log(0.75)]], device, True)
                loss = kl(t, s)
                self.assert_scalar(loss, math.log(4))
                loss.backward()
                torch.testing.assert_close(s.grad, self.tensor([[-0.75, 0.75]], device))

    def test_kl_student_gradient_matches_probability_difference(self):
        for device in self.devices:
            with self.subTest(device=device):
                teacher = [[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]]
                student = [[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]
                t = self.tensor(teacher, device).log()
                s = self.tensor(student, device).log().requires_grad_()
                kl(t, s).backward()
                expected = (self.tensor(student, device) - self.tensor(teacher, device)) / 2
                torch.testing.assert_close(s.grad, expected, rtol=0, atol=1e-12)

    def test_soft_ce_matches_independent_calculation_and_masks_target_mass(self):
        for device in self.devices:
            with self.subTest(device=device):
                logits = self.tensor([[math.log(0.25), math.log(0.75), -math.inf]], device)
                targets = self.tensor([[3., 1., 999.]], device)
                mask = torch.tensor([[True, True, False]], device=device)
                self.assert_scalar(soft_ce(logits, targets, mask), -0.75 * math.log(0.25) - 0.25 * math.log(0.75))

    def test_soft_ce_hard_targets_match_negative_log_probability(self):
        for device in self.devices:
            with self.subTest(device=device):
                logits = self.tensor([[math.log(0.25), math.log(0.75)], [math.log(0.8), math.log(0.2)]], device)
                targets = self.tensor([[0., 1.], [1., 0.]], device)
                mask = torch.ones_like(logits, dtype=torch.bool)
                self.assert_scalar(soft_ce(logits, targets, mask), -(math.log(0.75) + math.log(0.8)) / 2)

    def test_soft_ce_target_rescaling_preserves_loss(self):
        for device in self.devices:
            with self.subTest(device=device):
                logits = self.tensor([[2., -1., 3.], [0., -2., 1.]], device)
                targets = self.tensor([[2., 1., 3.], [1., 4., 1.]], device)
                mask = torch.ones_like(logits, dtype=torch.bool)
                scales = self.tensor([[7.], [0.125]], device)
                self.assert_scalar(soft_ce(logits, targets * scales, mask), soft_ce(logits, targets, mask).item())

    def test_soft_ce_gradient_matches_probability_difference(self):
        for device in self.devices:
            with self.subTest(device=device):
                logits = self.tensor([[math.log(0.2), math.log(0.8), -math.inf]], device, True)
                targets = self.tensor([[3., 1., 0.]], device)
                mask = torch.tensor([[True, True, False]], device=device)
                soft_ce(logits, targets, mask).backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                torch.testing.assert_close(logits.grad, self.tensor([[-0.55, 0.55, 0.]], device), rtol=0, atol=1e-12)

    def test_losses_average_rows_not_option_counts(self):
        for device in self.devices:
            with self.subTest(device=device):
                t = self.tensor([[0., -math.inf, -math.inf], [math.log(0.2), math.log(0.3), math.log(0.5)]], device)
                s = self.tensor([[0., -math.inf, -math.inf], [math.log(0.4), math.log(0.4), math.log(0.2)]], device)
                mask = torch.isfinite(t)
                targets = self.tensor([[1., 0., 0.], [0.2, 0.3, 0.5]], device)
                for function, args in [(kl, (t, s)), (soft_ce, (s, targets, mask))]:
                    combined = function(*args)
                    individual = sum(function(*(a[i:i+1] for a in args)).item() for i in range(2)) / 2
                    self.assert_scalar(combined, individual)

    def test_losses_are_invariant_to_rowwise_logit_offsets(self):
        for device in self.devices:
            with self.subTest(device=device):
                t = self.tensor([[1., 2., -math.inf], [-3., 2., 4.]], device)
                s = self.tensor([[3., -1., -math.inf], [2., -2., 0.]], device)
                shift = self.tensor([[1000.], [-1000.]], device)
                mask = torch.isfinite(t)
                targets = self.tensor([[0.2, 0.8, 0.], [0.2, 0.3, 0.5]], device)
                self.assert_scalar(kl(t + shift, s - shift), kl(t, s).item())
                self.assert_scalar(soft_ce(s + shift, targets, mask), soft_ce(s, targets, mask).item())

    def test_losses_large_logits_have_finite_values_and_gradients(self):
        for device in self.devices:
            with self.subTest(device=device):
                teacher = self.tensor([[1000., -1000.]], device)
                for loss_name in ("kl", "soft_ce"):
                    student = self.tensor([[-1000., 1000.]], device, True)
                    if loss_name == "kl":
                        loss = kl(teacher, student)
                    else:
                        loss = soft_ce(student, self.tensor([[1., 0.]], device), torch.ones_like(student, dtype=torch.bool))
                    self.assert_scalar(loss, 2000)
                    loss.backward()
                    torch.testing.assert_close(student.grad, self.tensor([[-1., 1.]], device), rtol=0, atol=1e-12)



if __name__ == "__main__":
    unittest.main()
