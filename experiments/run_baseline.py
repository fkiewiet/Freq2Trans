from pathlib import Path
import numpy as np

from numerics.config import Grid2D, HelmholtzConfig
from numerics.cases import make_default_cases
from numerics.medium import build_medium
from numerics.rhs import assemble_rhs
from numerics.assemble import assemble_helmholtz_matrix
from numerics.solve import solve_linear_system, compute_residual
from numerics.diagnostics import save_npz, plot_field, plot_spectrum


def main():
    grid = Grid2D(101, 101, 1.0, 1.0)
    cfg = HelmholtzConfig(omega=40.0, grid=grid)
    cases = make_default_cases()

    for name, case in cases.items():
        X, Y = grid.mesh()
        c = build_medium(cfg, case, X, Y)
        A = assemble_helmholtz_matrix(cfg, c)
        f = assemble_rhs(cfg, case, X, Y)

        u = solve_linear_system(A, f)
        r = compute_residual(A, u, f)

        outdir = Path("outputs") / name
        figs = outdir / "figs"
        figs.mkdir(parents=True, exist_ok=True)

        save_npz(outdir / "solution.npz", u=u, r=r, f=f, omega=np.array(cfg.omega))

        plot_field(grid.nx, grid.ny, u, f"{name}: |u|", figs / "u.png")
        plot_field(grid.nx, grid.ny, r, f"{name}: |residual|", figs / "residual.png")
        plot_spectrum(grid.nx, grid.ny, r, f"{name}: residual", figs / "residual_spectrum.png")

        print("finished", name)


if __name__ == "__main__":
    main()
