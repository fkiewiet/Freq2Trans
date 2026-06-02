"""
Small, intuition-first skeleton for a flexible GMRES-style interface.

This file is intentionally simple. It is not yet a production solver.
Its purpose is to keep the input/output contract clear:

    residual -> preconditioner -> correction

for the problem:

    A u = f
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Protocol

import numpy as np


Array = np.ndarray


class LinearOperator(Protocol):
    def __call__(self, x: Array) -> Array:
        ...


class FlexiblePreconditioner(Protocol):
    def __call__(self, residual: Array, iteration: int) -> Array:
        ...


@dataclass
class FlexibleHistory:
    residual_norms: List[float] = field(default_factory=list)


def l2_norm(x: Array) -> float:
    return float(np.linalg.norm(x.ravel()))


def residual(A: LinearOperator, u: Array, f: Array) -> Array:
    return f - A(u)


def identity_preconditioner(residual: Array, iteration: int) -> Array:
    del iteration
    return residual.copy()


def staged_preconditioner(
    early_pc: FlexiblePreconditioner,
    late_pc: FlexiblePreconditioner,
    switch_iteration: int,
) -> FlexiblePreconditioner:
    def apply(residual: Array, iteration: int) -> Array:
        if iteration < switch_iteration:
            return early_pc(residual, iteration)
        return late_pc(residual, iteration)

    return apply


def toy_flexible_iteration(
    A: LinearOperator,
    f: Array,
    preconditioner: FlexiblePreconditioner,
    u0: Array | None = None,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> tuple[Array, FlexibleHistory]:
    """
    Minimal correction loop to keep the math visible.

    This is not full FGMRES yet. It is the preconditioner contract written in
    code:

    1. compute residual
    2. apply current preconditioner
    3. update iterate

    A true FGMRES implementation would additionally build and orthogonalize a
    Krylov basis and solve a small least-squares problem at each restart cycle.
    """

    u = np.zeros_like(f) if u0 is None else u0.copy()
    f_norm = max(l2_norm(f), 1e-30)
    history = FlexibleHistory()

    for iteration in range(max_iter):
        r = residual(A, u, f)
        rel_res = l2_norm(r) / f_norm
        history.residual_norms.append(rel_res)

        if rel_res < tol:
            break

        z = preconditioner(r, iteration)
        u = u + z

    return u, history


def explain_iteration_step() -> str:
    return (
        "At iteration k, the solver forms r_k = f - A u_k, sends r_k into the "
        "current preconditioner, receives a correction z_k, and uses z_k to "
        "improve the current solution."
    )


if __name__ == "__main__":
    print(explain_iteration_step())
