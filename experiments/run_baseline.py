
from pathlib import Path
from numerics.config import Grid2D, HelmholtzConfig
from numerics.cases import make_default_cases
from numerics.medium import build_medium
from numerics.rhs import assemble_rhs
from numerics.assemble import assemble_helmholtz_matrix
from numerics.solve import solve_linear_system, compute_residual
from numerics.diagnostics import save_npz

def main():
    grid = Grid2D(11,11,1.0,1.0)
    cfg = HelmholtzConfig(omega=10.0, grid=grid)
    cases = make_default_cases()

    for name, case in cases.items():
        X,Y = grid.mesh()
        c = build_medium(cfg, case, X, Y)
        A = assemble_helmholtz_matrix(cfg, c)
        f = assemble_rhs(cfg, case, X, Y)
        u = solve_linear_system(A, f)
        r = compute_residual(A, u, f)
        save_npz(Path("outputs")/name/"solution.npz", u=u, r=r)
        print("finished", name)

if __name__ == "__main__":
    main()
