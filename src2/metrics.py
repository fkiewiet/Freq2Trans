import torch
import numpy as np
from typing import Dict, Optional
from src2.loss import relative_l2 as _rel_l2


def phase_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ap = torch.atan2(pred[:,1],   pred[:,0])
    at = torch.atan2(target[:,1], target[:,0])
    diff = torch.remainder(ap - at + np.pi, 2 * np.pi) - np.pi
    return diff.abs().mean()


def pml_split_mse(pred: torch.Tensor, target: torch.Tensor,
                  pml_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    interior = 1.0 - pml_mask
    sq       = (pred - target) ** 2
    int_mse  = (sq * interior).sum() / (interior.sum() * pred.shape[1] + 1e-8)
    bnd_mse  = (sq * pml_mask).sum()  / (pml_mask.sum()  * pred.shape[1] + 1e-8)
    return {"interior_mse": int_mse, "boundary_mse": bnd_mse}


def source_bin_rel_l2(per_sample_rel_l2: torch.Tensor,
                      bin_ids: np.ndarray) -> Dict[int, float]:
    return {int(b): per_sample_rel_l2[np.where(bin_ids == b)[0]].mean().item()
            for b in np.unique(bin_ids)}


def percentile_rel_l2(pred: torch.Tensor, target: torch.Tensor,
                      p: float = 95.0, eps: float = 1e-8) -> float:
    diff   = (pred - target).reshape(pred.shape[0], -1)
    t_flat = target.reshape(target.shape[0], -1)
    vals   = (diff.norm(dim=1) / (t_flat.norm(dim=1) + eps)).detach().cpu().numpy()
    return float(np.percentile(vals, p))


def spectral_error_profile(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    error   = (pred - target)[:, 0]
    power   = torch.fft.fftshift(torch.fft.fft2(error)).abs().pow(2).mean(0).detach().cpu().numpy()
    H, W    = power.shape
    cx, cy  = H // 2, W // 2
    radii   = np.sqrt((np.arange(H)[:,None] - cx)**2 + (np.arange(W)[None,:] - cy)**2)
    profile = np.array([power[((radii >= r) & (radii < r+1))].mean()
                        if ((radii >= r) & (radii < r+1)).any() else 0.0
                        for r in range(H // 2)])
    return profile


def all_metrics(pred: torch.Tensor, target: torch.Tensor,
                pml_mask=None, bin_ids=None) -> Dict[str, float]:
    sq  = (pred - target) ** 2
    out = {
        "rel_l2_mean":     _rel_l2(pred, target).item(),
        "rel_l2_p95":      percentile_rel_l2(pred, target, 95),
        "mse_re":          sq[:, 0].mean().item(),
        "mse_im":          sq[:, 1].mean().item(),
        "phase_error_rad": phase_error(pred, target).item(),
    }
    if pml_mask is not None:
        s = pml_split_mse(pred, target, pml_mask)
        out["interior_mse"] = s["interior_mse"].item()
        out["boundary_mse"] = s["boundary_mse"].item()
    if bin_ids is not None:
        diff  = (pred - target).reshape(pred.shape[0], -1)
        t_f   = target.reshape(target.shape[0], -1)
        per_s = diff.norm(dim=1) / (t_f.norm(dim=1) + 1e-8)
        for bid, val in source_bin_rel_l2(per_s, bin_ids).items():
            out[f"bin_{bid}_rel_l2"] = val
    return out
