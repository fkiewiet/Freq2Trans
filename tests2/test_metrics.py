"""tests/test_metrics.py — run: python -m pytest tests/test_metrics.py -v"""
import pytest
import torch
import numpy as np


def make_field(B=4, H=32, W=32, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, 2, H, W, dtype=torch.float32)

def make_mask(B=4, H=32, W=32, pml=4):
    m = torch.zeros(1, 1, H, W)
    m[:, :, :pml, :] = 1; m[:, :, -pml:, :] = 1
    m[:, :, :, :pml] = 1; m[:, :, :, -pml:] = 1
    return m


class TestPhaseError:
    def test_identical_zero(self):
        from src2.metrics import phase_error
        x = make_field()
        assert phase_error(x, x).item() == pytest.approx(0.0, abs=1e-5)

    def test_negated_is_pi(self):
        from src2.metrics import phase_error
        x = make_field()
        assert phase_error(-x, x).item() == pytest.approx(np.pi, rel=0.01)

    def test_in_range(self):
        from src2.metrics import phase_error
        r = phase_error(make_field(seed=0), make_field(seed=1)).item()
        assert 0.0 <= r <= np.pi


class TestPMLSplitMSE:
    def test_keys(self):
        from src2.metrics import pml_split_mse
        r = pml_split_mse(make_field(), make_field(seed=1), make_mask())
        assert {"interior_mse", "boundary_mse"} <= r.keys()

    def test_identical_zero(self):
        from src2.metrics import pml_split_mse
        x = make_field()
        r = pml_split_mse(x, x, make_mask())
        assert r["interior_mse"].item() == pytest.approx(0.0, abs=1e-6)
        assert r["boundary_mse"].item() == pytest.approx(0.0, abs=1e-6)

    def test_error_only_in_boundary(self):
        from src2.metrics import pml_split_mse
        target = make_field(); mask = make_mask()
        pred = target.clone() + mask * 1.0
        r = pml_split_mse(pred, target, mask)
        assert r["interior_mse"].item() == pytest.approx(0.0, abs=1e-5)
        assert r["boundary_mse"].item() > 0


class TestPercentileRelL2:
    def test_p95_geq_mean(self):
        from src2.metrics import percentile_rel_l2
        from src2.loss import relative_l2
        pred, target = make_field(seed=0), make_field(seed=1)
        assert percentile_rel_l2(pred, target, 95) >= relative_l2(pred, target).item()

    def test_identical_zero(self):
        from src2.metrics import percentile_rel_l2
        x = make_field()
        assert percentile_rel_l2(x, x, 95) == pytest.approx(0.0, abs=1e-5)


class TestSourceBinRelL2:
    def test_correct_means(self):
        from src2.metrics import source_bin_rel_l2
        errors  = torch.tensor([0.1, 0.3, 0.5, 0.7])
        bin_ids = np.array([0, 0, 1, 1])
        r = source_bin_rel_l2(errors, bin_ids)
        assert r[0] == pytest.approx(0.2, rel=1e-4)
        assert r[1] == pytest.approx(0.6, rel=1e-4)


class TestAllMetrics:
    def test_flat_dict_of_floats(self):
        from src2.metrics import all_metrics
        r = all_metrics(make_field(seed=0), make_field(seed=1))
        assert isinstance(r, dict)
        for k, v in r.items():
            assert isinstance(v, float), f"['{k}'] is {type(v)}"

    def test_required_keys(self):
        from src2.metrics import all_metrics
        r = all_metrics(make_field(seed=0), make_field(seed=1))
        for key in ["rel_l2_mean", "rel_l2_p95", "mse_re", "mse_im", "phase_error_rad"]:
            assert key in r, f"Missing: {key}"
