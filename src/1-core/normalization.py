import torch

class PhysicsNormalizer:
    @staticmethod
    def scale_output(tensor, omega, inverse=False):
        """
        Applies 1/sqrt(omega) scaling to the wavefield.
        This compensates for the amplitude decay of the Hankel function at high frequencies.
        """
        factor = torch.sqrt(torch.tensor(omega, dtype=torch.float32))
        if inverse:
            return tensor * factor
        return tensor / factor

    @staticmethod
    def z_score(tensor):
        """Standard Unit Variance normalization for CNN inputs."""
        mean = tensor.mean()
        std = tensor.std() + 1e-8
        return (tensor - mean) / std