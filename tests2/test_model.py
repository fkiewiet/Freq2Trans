"""tests/test_model.py — run: python -m pytest tests/test_model.py -v"""
import pytest
import torch

BASE = {
    "in_channels": 8, "out_channels": 2, "width": 64, "kernel_size": 3,
    "dilation_pattern": [1, 1, 2, 4, 8, 16, 32, 64, 1, 1],
    "activation_body": "relu", "activation_integration": "gelu",
}
SMALL = {**BASE, "width": 16}


class TestDilatedCNN:
    def test_output_shape(self):
        from src2.model import DilatedCNN
        y = DilatedCNN(SMALL)(torch.randn(2, 8, 64, 64))
        assert y.shape == (2, 2, 64, 64)

    def test_output_shape_512(self):
        from src2.model import DilatedCNN
        y = DilatedCNN(SMALL)(torch.randn(1, 8, 512, 512))
        assert y.shape == (1, 2, 512, 512)

    def test_no_nan(self):
        from src2.model import DilatedCNN
        y = DilatedCNN(SMALL)(torch.randn(2, 8, 64, 64))
        assert not torch.isnan(y).any()

    def test_no_inf(self):
        from src2.model import DilatedCNN
        y = DilatedCNN(SMALL)(torch.randn(2, 8, 64, 64))
        assert not torch.isinf(y).any()

    def test_param_count(self):
        from src2.model import DilatedCNN
        n = DilatedCNN(BASE).count_parameters()
        assert 200_000 <= n <= 500_000, f"Unexpected param count: {n:,}"

    def test_no_output_activation(self):
        from src2.model import DilatedCNN
        y = DilatedCNN(SMALL)(torch.randn(1, 8, 32, 32) * 10.0)
        assert (y.abs() > 1.0).any(), "Output appears bounded — check output_proj activation"

    def test_backward(self):
        from src2.model import DilatedCNN
        model = DilatedCNN(SMALL)
        x = torch.randn(1, 8, 32, 32)
        model(x).mean().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad: {name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad: {name}"

    def test_resolution_preserved(self):
        from src2.model import DilatedCNN
        model = DilatedCNN(SMALL)
        for size in [32, 64, 128]:
            y = model(torch.randn(1, 8, size, size))
            assert y.shape[-2:] == (size, size)

    def test_layer_count(self):
        from src2.model import DilatedCNN
        model = DilatedCNN(SMALL)
        assert len(model.layers) == len(SMALL["dilation_pattern"])

    def test_width_128(self):
        from src2.model import DilatedCNN
        y = DilatedCNN({**BASE, "width": 128})(torch.randn(1, 8, 32, 32))
        assert y.shape == (1, 2, 32, 32)
