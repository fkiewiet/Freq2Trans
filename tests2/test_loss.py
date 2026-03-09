"""tests/test_loss.py — run: python -m pytest tests/test_loss.py -v"""
import pytest
import torch
import numpy as np


def make_field(B=2, H=32, W=32, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, 2, H, W, dtype=torch.float32)


class TestComplexMSE:
    def test_identical_zero(self):
        from src2.loss import complex_mse
        x = make_field()
        assert complex_mse(x, x).item() == pytest.approx(0.0, abs=1e-7)

    def test_positive(self):
        from src2.loss import complex_mse
        assert complex_mse(make_field(seed=0), make_field(seed=1)).item() > 0

    def test_scalar(self):
        from src2.loss import complex_mse
        assert complex_mse(make_field(), make_field(seed=1)).shape == torch.Size([])

    def test_equal_re_im_weight(self):
        from src2.loss import complex_mse
        target = torch.zeros(1, 2, 8, 8)
        re = torch.zeros(1, 2, 8, 8); re[:, 0] = 1.0
        im = torch.zeros(1, 2, 8, 8); im[:, 1] = 1.0
        assert complex_mse(re, target).item() == pytest.approx(
               complex_mse(im, target).item(), rel=1e-5)

    def test_fp32_output(self):
        from src2.loss import complex_mse
        result = complex_mse(make_field().float(), make_field(seed=1).float())
        assert result.dtype == torch.float32


class TestRelativeL2:
    def test_identical_zero(self):
        from src2.loss import relative_l2
        x = make_field()
        assert relative_l2(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_double_target_is_one(self):
        from src2.loss import relative_l2
        t = make_field()
        assert relative_l2(2.0 * t, t).item() == pytest.approx(1.0, rel=1e-4)

    def test_no_div_zero(self):
        from src2.loss import relative_l2
        assert torch.isfinite(relative_l2(make_field(), torch.zeros(2, 2, 32, 32)))


class TestPhysicsResidual:
    def test_plane_wave_near_zero(self):
        from src2.loss import physics_residual
        k = 2.0 * np.pi / 16.0
        H = W = 64
        x = torch.linspace(0, W-1, W).unsqueeze(0).expand(H, -1)
        pred = torch.cat([torch.cos(k*x).unsqueeze(0).unsqueeze(0),
                          torch.sin(k*x).unsqueeze(0).unsqueeze(0)], dim=1).float()
        assert physics_residual(pred, k=k, dx=1.0).item() < 0.05

    def test_random_field_nonzero(self):
        from src2.loss import physics_residual
        assert physics_residual(make_field(B=1), k=1.0, dx=1.0).item() > 0


class TestCombinedLoss:
    def test_lambda_zero_equals_mse(self):
        from src2.loss import combined_loss, complex_mse
        pred, target = make_field(seed=0), make_field(seed=1)
        total, comp = combined_loss(pred, target, lambda_residual=0.0)
        assert comp["total"] == pytest.approx(complex_mse(pred, target).item(), rel=1e-4)

    def test_keys_present(self):
        from src2.loss import combined_loss
        _, comp = combined_loss(make_field(), make_field(seed=1))
        assert {"mse", "residual", "total"} <= comp.keys()

    def test_total_has_grad(self):
        from src2.loss import combined_loss
        pred = make_field().requires_grad_(True)
        total, _ = combined_loss(pred, make_field(seed=1))
        assert total.requires_grad

    def test_components_are_floats(self):
        from src2.loss import combined_loss
        _, comp = combined_loss(make_field(), make_field(seed=1))
        for k, v in comp.items():
            assert isinstance(v, float), f"components['{k}'] is {type(v)}"
