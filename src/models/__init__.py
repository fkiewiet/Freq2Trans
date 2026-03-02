from .unet import TransferUNet
from .cnn import PlainCNN

# This allows you to do: from src.models import TransferUNet, PlainCNN
__all__ = ["TransferUNet", "PlainCNN"]