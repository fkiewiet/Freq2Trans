
import numpy as np
import scipy.sparse as sp
from .grid import idx
from .config import HelmholtzConfig

def ppw_gate(cfg: HelmholtzConfig, c):
    cmin = c.min()
    lam = 2*np.pi*cmin/cfg.omega
    h = max(cfg.grid.hx, cfg.grid.hy)
    return {"ppw": lam/h}

def assemble_helmholtz_matrix(cfg: HelmholtzConfig, c):
    nx, ny = cfg.grid.nx, cfg.grid.ny
    N = nx*ny
    rows, cols, data = [], [], []
    for i in range(nx):
        for j in range(ny):
            p = idx(i,j,ny)
            rows.append(p); cols.append(p); data.append(-1.0)
    return sp.coo_matrix((data,(rows,cols)), shape=(N,N)).tocsr()
