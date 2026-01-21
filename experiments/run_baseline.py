from __future__ import annotations
from pathlib import Path
import numpy as np

from freq2transfer.helmholtz.config import Grid2D, PMLConfig, HelmholtzConfig
from freq2transfer.helmholtz.cases import make_default_cases
from freq2transfer.helmholtz.medium import build_medium
from freq2transfer.helmholtz.rhs import assemble_rhs
from freq2transfer.helmholtz.assemble import assemble_helmholtz_matrix, ppw_gate
from freq2transfer.helmholtz.solve import solve_linear_system, compute_residual
from freq2transfer.helmholtz.diagnostics import save_npz, plot_field, plot_spectrum, pml_energy_proxy


def run_case(cfg: HelmholtzConfig, case, outdir: Path, solver: str = "direct") -> dict[str, float]:
    outdir.mkdir(parents=True, exist_ok=True)

    X, Y = cfg.grid.mesh()
    c = build_medium(cfg, case, X, Y)
    gate = ppw_gate(cfg, c)

    A = assemble_helmholtz_matrix(cfg, c)
    f = assemble_rhs(cfg, case, X, Y)

    u = solve_linear_system(A, f, method=solver)
    r = compute_residual(A, u, f)

    metrics = {
        "ppw": float(gate["ppw"]),
        "pml_energy_frac": float(pml_energy_proxy(cfg, u)),
        "rel_residual": float(np.linalg.norm(r) / (np.linalg.norm(f) + 1e-30)),
    }

    save_npz(outdir / "fields" / "solution_and_residual.npz",
             u=u, r=r, f=f, c=c, omega=np.array(cfg.omega))

    plot_field(cfg.grid.nx, cfg.grid.ny, u, f"{case.name} u", outdir / "figs" / "u.png")
    plot_field(cfg.grid.nx, cfg.grid.ny, r, f"{case.name} residual", outdir / "figs" / "residual.png")
    plot_spectrum(cfg.grid.nx, cfg.grid.ny, r, f"{case.name} residual", outdir / "figs" / "residual_spectrum.png")

    save_npz(outdir / "metrics.npz", **{k: np.array(v) for k, v in metrics.items()})
    return metrics


def main() -> None:
    base_out = Path("outputs")
    cases = make_default_cases()

    grid = Grid2D(nx=201, ny=201, lx=1.0, ly=1.0)
    pml = PMLConfig(thickness=20, strength=50.0, power=2.0)
    cfg = HelmholtzConfig(omega=80.0, grid=grid, pml=pml, ppw_target=10.0)

    for name, case in cases.items():
        outdir = base_out / f"case_{name}" / f"omega_{cfg.omega:.1f}"
        metrics = run_case(cfg, case, outdir, solver="direct")
        print(name, metrics)


if __name__ == "__main__":
    main()
