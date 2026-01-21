
from numerics.config import Grid2D, HelmholtzConfig
from numerics.cases import make_default_cases
from numerics.medium import build_medium
from numerics.assemble import assemble_helmholtz_matrix

def test_assemble():
    grid = Grid2D(5,5,1.0,1.0)
    cfg = HelmholtzConfig(omega=5.0, grid=grid)
    case = make_default_cases()["const"]
    X,Y = grid.mesh()
    c = build_medium(cfg, case, X, Y)
    A = assemble_helmholtz_matrix(cfg, c)
    assert A.shape[0] == grid.nx*grid.ny
