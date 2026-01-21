import numpy as np

from numerics.config import Grid2D, PMLConfig, HelmholtzConfig
from numerics.cases import make_default_cases
from numerics.medium import build_medium
from numerics.rhs import assemble_rhs
from numerics.assemble import assemble_helmholtz_matrix, ppw_gate
from numerics.solve import solve_linear_system, compute_residual
from numerics.diagnostics import save_npz, plot_field, plot_spectrum, pml_energy_proxy



def test_assemble_shape():
    grid = Grid2D(nx=51, ny=51, lx=1.0, ly=1.0)
    cfg = HelmholtzConfig(omega=40.0, grid=grid, pml=None)
    case = make_default_cases()["const_point"]
    X, Y = grid.mesh()
    c = build_medium(cfg, case, X, Y)
    A = assemble_helmholtz_matrix(cfg, c)
    assert A.shape == (grid.nx * grid.ny, grid.nx * grid.ny)
    assert np.iscomplexobj(A.data)
