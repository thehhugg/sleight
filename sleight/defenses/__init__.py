"""Defense mechanism implementations."""

from .adversarial_training import adversarial_train
from .input_transforms import jpeg_compression, spatial_smoothing, bit_depth_reduction
from .distillation import defensive_distillation

__all__ = [
    "adversarial_train",
    "jpeg_compression",
    "spatial_smoothing",
    "bit_depth_reduction",
    "defensive_distillation",
]
