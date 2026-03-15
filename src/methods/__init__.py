"""
CoT Vector methods package.

Available methods:
- ExtractedCoTVector: Statistical aggregation (Eq. 4-5)
- LearnableCoTVector: Teacher-student gradient optimization (Eq. 6)
- UACoTVector: Bayesian shrinkage with uncertainty-aware gating
- ABCCoTVector: Adaptive Bayesian CoT Vector with variational inference
"""

from .base import BaseCoTVectorMethod
from .extracted import ExtractedCoTVector
from .learnable import LearnableCoTVector
from .ua_vector import UACoTVector
from .abc_vector import ABCCoTVector

__all__ = [
    "BaseCoTVectorMethod",
    "ExtractedCoTVector",
    "LearnableCoTVector",
    "UACoTVector",
    "ABCCoTVector",
]
