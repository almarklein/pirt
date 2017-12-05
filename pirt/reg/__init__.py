# flake8: noqa
"""
The reg module implements the various registration algorithms.
"""

from .reg_base import AbstractRegistration, NullRegistration, BaseRegistration, GDGRegistration
from .reg_gravity import GravityRegistration
from .reg_demons import OriginalDemonsRegistration, DiffeomorphicDemonsRegistration
from .reg_elastix import (ElastixRegistration, ElastixGroupwiseRegistration,
                          ElastixRegistration_rigid, ElastixRegistration_affine)
