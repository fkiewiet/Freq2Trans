import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def complex_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff       = (pred - target).reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    return (diff.norm(dim=1) / (target_flat.norm(dim=1) + eps)).mean()


def physics_residual(pred: torch.Tensor, k: float, dx: float,
                     source_field: torch.Tensor = None,
                     eps: float = 1e-8) -> torch.Tensor:
    kernel = torch.tensor([[0., 1., 0.],
                           [1.,-4., 1.],
                           [0., 1., 0.]], device=pred.device, dtype=pred.dtype)
    kernel = kernel.view(1, 1, 3, 3).expand(2, 1, 3, 3) / (dx ** 2)
    lap    = F.conv2d(pred, kernel, padding=1, groups=2)
    residual = lap + (k ** 2) * pred
    # Exclude 1-pixel border where zero-padding creates fake Laplacian errors
    interior = (slice(None), slice(None), slice(1, -1), slice(1, -1))
    if source_field is not None:
        residual = residual - source_field
        return residual[interior].norm() / (source_field[interior].norm() + eps)
    return residual[interior].norm() / (pred[interior].norm() + eps)


def combined_loss(pred: torch.Tensor, target: torch.Tensor,
                  k: float = None, dx: float = None,
                  source_field: torch.Tensor = None,
                  lambda_residual: float = 0.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    mse = complex_mse(pred, target)
    if lambda_residual > 0 and k is not None and dx is not None:
        res = physics_residual(pred, k, dx, source_field)
    else:
        res = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    total = mse + lambda_residual * res
    return total, {"mse": mse.item(), "residual": res.item(), "total": total.item()}
