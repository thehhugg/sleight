"""Adversarial attack implementations."""

from .fgsm_attack import fgsm_attack
from .pgd_attack import pgd_attack

__all__ = ["fgsm_attack", "pgd_attack"]
