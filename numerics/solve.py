
import scipy.sparse.linalg as spla

def solve_linear_system(A, f):
    return spla.spsolve(A, f)

def compute_residual(A, u, f):
    return f - A @ u
