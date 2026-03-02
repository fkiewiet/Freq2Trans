from .preconditioner import LearnedPreconditioner
from .iterative import RefinementLoop, ConvergenceTracker, RefinementHistory

def get_preconditioner(mode, model, omega_low, n_tot, phys_config):
    """
    Helper to quickly instantiate the right solver mode for Krylov methods.
    """
    return LearnedPreconditioner(
        model=model, 
        omega_low=omega_low, 
        n_tot=n_tot, 
        mode=mode,
        config=phys_config
    )

def get_refinement_solver(cfg_high, c_phys, f_phys, u_true=None):
    """
    Helper to initialize the Iterative Refinement framework.
    """
    return RefinementLoop(
        cfg_high=cfg_high,
        c_phys=c_phys,
        f_phys=f_phys,
        u_true=u_true
    )

__all__ = [
    "LearnedPreconditioner", 
    "get_preconditioner", 
    "RefinementLoop", 
    "get_refinement_solver",
    "ConvergenceTracker",
    "RefinementHistory"
]