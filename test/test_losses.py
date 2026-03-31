"""Tests for src/losses.py (HingeLoss and FocalLoss)."""
import math
import pytest
import torch
from src.losses import HingeLoss, FocalLoss


# ── HingeLoss ─────────────────────────────────────────────────────────────────

class TestHingeLoss:
    def setup_method(self):
        self.loss_fn = HingeLoss(margin=1.0)

    def test_output_is_scalar(self):
        d = torch.tensor([0.5])
        t = torch.tensor([1.0])
        out = self.loss_fn(d, t)
        assert out.shape == torch.Size([])

    def test_positive_pair_zero_distance_zero_loss(self):
        """Same-track pair at distance 0 → no loss."""
        d = torch.tensor([0.0])
        t = torch.tensor([1.0])
        assert self.loss_fn(d, t).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_pair_large_distance_positive_loss(self):
        """Same-track pair at distance 5 → loss > 0."""
        d = torch.tensor([5.0])
        t = torch.tensor([1.0])
        assert self.loss_fn(d, t).item() > 0.0

    def test_negative_pair_beyond_margin_zero_loss(self):
        """Different-track pair at distance > margin → contribution is 0."""
        d = torch.tensor([2.0])  # > margin=1.0
        t = torch.tensor([0.0])
        assert self.loss_fn(d, t).item() == pytest.approx(0.0, abs=1e-6)

    def test_negative_pair_within_margin_positive_loss(self):
        """Different-track pair at distance 0 (< margin) → loss > 0."""
        d = torch.tensor([0.0])
        t = torch.tensor([0.0])
        assert self.loss_fn(d, t).item() > 0.0


# ── FocalLoss ─────────────────────────────────────────────────────────────────

class TestFocalLoss:
    def setup_method(self):
        self.loss_fn = FocalLoss(alpha=0.995, gamma=2.0)

    def test_output_is_scalar(self):
        pred = torch.tensor([0.5])
        t = torch.tensor([1.0])
        assert self.loss_fn(pred, t).shape == torch.Size([])

    def test_perfect_positive_prediction_near_zero_loss(self):
        """pred ≈ 1, label = 1 → almost no loss."""
        pred = torch.tensor([0.999])
        t = torch.tensor([1.0])
        assert self.loss_fn(pred, t).item() < 0.01

    def test_wrong_positive_prediction_high_loss(self):
        """pred ≈ 0, label = 1 → large loss."""
        pred = torch.tensor([0.001])
        t = torch.tensor([1.0])
        assert self.loss_fn(pred, t).item() > 0.1

    def test_perfect_negative_prediction_near_zero_loss(self):
        pred = torch.tensor([0.001])
        t = torch.tensor([0.0])
        assert self.loss_fn(pred, t).item() < 0.01

    def test_no_nan_with_extreme_predictions(self):
        """Predictions at exactly 0 or 1 (clamped inside) should not NaN."""
        pred = torch.tensor([0.0, 1.0])
        t = torch.tensor([1.0, 0.0])
        out = self.loss_fn(pred, t)
        assert not torch.isnan(out)

    def test_gamma_zero_equals_weighted_bce(self):
        """FocalLoss(gamma=0) must reduce to weighted BCE (up to alpha scaling)."""
        torch.manual_seed(0)
        pred = torch.rand(50).clamp(1e-6, 1 - 1e-6)
        t = (torch.rand(50) > 0.5).float()

        alpha = 0.7
        fl = FocalLoss(alpha=alpha, gamma=0.0)
        fl_loss = fl(pred, t).item()

        eps = 1e-7
        p = pred.clamp(eps, 1 - eps)
        bce = -(alpha * t * torch.log(p) + (1 - alpha) * (1 - t) * torch.log(1 - p))
        bce_loss = bce.mean().item()

        assert fl_loss == pytest.approx(bce_loss, rel=1e-4)

    def test_alpha_weighting(self):
        """With high alpha (up-weight positives), FN penalty >> FP penalty."""
        loss_fn = FocalLoss(alpha=0.99, gamma=2.0)
        # False negative: label=1, pred=0.1
        fn_loss = loss_fn(torch.tensor([0.1]), torch.tensor([1.0])).item()
        # False positive: label=0, pred=0.9
        fp_loss = loss_fn(torch.tensor([0.9]), torch.tensor([0.0])).item()
        assert fn_loss > fp_loss
