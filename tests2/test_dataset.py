"""tests/test_dataset.py — run: python -m pytest tests/test_dataset.py -v"""
import pytest
import torch
import numpy as np


def make_sample(seed=0, g=16):
    rng = np.random.default_rng(seed)
    interior = g // 4
    return {
        "u_source_re":      rng.standard_normal((g, g)).astype(np.float32),
        "u_source_im":      rng.standard_normal((g, g)).astype(np.float32),
        "u_target_re":      rng.standard_normal((g, g)).astype(np.float32),
        "u_target_im":      rng.standard_normal((g, g)).astype(np.float32),
        "pml_mask":         np.zeros((g, g), dtype=np.float32),
        "source_xy":        (interior + seed % interior, interior + seed % interior),
        "source_amplitude": float(1.0 + rng.random()),
    }

def make_samples(n=20, g=16):
    return [make_sample(seed=i, g=g) for i in range(n)]


class TestChannelNormaliser:
    def test_zero_mean_unit_std(self):
        from src2.dataset import ChannelNormaliser, HelmholtzDataset
        samples = make_samples(50, g=16)
        ds = HelmholtzDataset(samples, normaliser=None, grid_size=16)
        raw = [ds[i][0] for i in range(len(ds))]
        norm = ChannelNormaliser(); norm.fit(raw)
        stacked = torch.stack([norm.transform(x) for x in raw])
        assert torch.allclose(stacked.mean(dim=(0,2,3)), torch.zeros(8), atol=0.1)
        assert torch.allclose(stacked.std(dim=(0,2,3)),  torch.ones(8),  atol=0.1)

    def test_inverse_recovers_original(self):
        from src2.dataset import ChannelNormaliser, HelmholtzDataset
        samples = make_samples(10, g=16)
        ds = HelmholtzDataset(samples, normaliser=None, grid_size=16)
        raw = [ds[i][0] for i in range(len(ds))]
        norm = ChannelNormaliser(); norm.fit(raw)
        x = raw[0]
        assert torch.allclose(norm.inverse_transform(norm.transform(x)), x, atol=1e-5)


class TestHelmholtzDataset:
    def test_output_shapes(self):
        from src2.dataset import HelmholtzDataset
        ds = HelmholtzDataset(make_samples(5, g=16), grid_size=16)
        x, y = ds[0]
        assert x.shape == (8, 16, 16)
        assert y.shape == (2, 16, 16)

    def test_no_nan(self):
        from src2.dataset import HelmholtzDataset
        ds = HelmholtzDataset(make_samples(5, g=16), grid_size=16)
        for i in range(len(ds)):
            x, y = ds[i]
            assert not torch.isnan(x).any()
            assert not torch.isnan(y).any()

    def test_meshgrid_range(self):
        from src2.dataset import HelmholtzDataset
        ds = HelmholtzDataset(make_samples(3, g=16), grid_size=16)
        x, _ = ds[0]
        assert x[2].min() >= -1.0 - 1e-5 and x[2].max() <= 1.0 + 1e-5
        assert x[3].min() >= -1.0 - 1e-5 and x[3].max() <= 1.0 + 1e-5


class TestMakeSplits:
    def _cfg(self):
        return {"data": {"train_frac": 0.8, "val_frac": 0.1, "seed": 42,
                         "grid_size": 16, "pml_cells": 4,
                         "source_gaussian_sigma": 2.0, "stratify_grid": [5, 5]}}

    def test_sizes(self):
        from src2.dataset import make_splits
        tr, va, te = make_splits(make_samples(100, g=16), self._cfg())
        assert len(tr) == 80 and len(va) == 10 and len(te) == 10

    def test_no_overlap(self):
        from src2.dataset import make_splits
        tr, va, te = make_splits(make_samples(60, g=16), self._cfg())
        tk = {s["source_xy"] for s in tr.samples}
        vk = {s["source_xy"] for s in va.samples}
        xk = {s["source_xy"] for s in te.samples}
        assert not tk & vk and not tk & xk and not vk & xk

    def test_shared_normaliser(self):
        from src2.dataset import make_splits
        tr, va, te = make_splits(make_samples(30, g=16), self._cfg())
        assert tr.normaliser is va.normaliser is te.normaliser
