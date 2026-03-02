class HelmholtzPreconditioner:
    """
    Logic to handle the selection between:
    - 'none': Standard GMRES
    - 'identity': Physics-only (L_low)
    - 'cnn': Physics + Local CNN
    - 'unet': Physics + Global U-Net
    """
    def __init__(self, mode='unet', model=None, LU_low=None):
        self.mode = mode
        self.model = model
        self.LU_low = LU_low

    def apply(self, r):
        # 1. Solve coarse physics (Always happens unless 'none')
        e_coarse = self.LU_low.solve(r)
        
        if self.mode == 'identity':
            return e_coarse
        
        # 2. Apply Neural Correction
        # (Neural logic goes here...)
        return corrected_e