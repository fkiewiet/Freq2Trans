import numpy as np
from freq2transfer.helmholtz.config import Grid2D, HelmholtzConfig
from freq2transfer.helmholtz.cases import make_default_cases
from freq2transfer.helmholtz.medium import build_medium
from freq2transfer.helmholtz.assemble import assemble_helmholtz_matrix


def test_assemble_shape():
    grid = Grid2D(nx=51, ny=51, lx=1.0, ly=1.0)
    cfg = HelmholtzConfig(omega=40.0, grid=grid, pml=None)
    case = make_default_cases()["const_point"]
    X, Y = grid.mesh()
    c = build_medium(cfg, case, X, Y)
    A = assemble_helmholtz_matrix(cfg, c)
    assert A.shape == (grid.nx * grid.ny, grid.nx * grid.ny)
    assert np.iscomplexobj(A.data)
