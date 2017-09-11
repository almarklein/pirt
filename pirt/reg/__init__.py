# flake8: noqa

""" Registration objects
"""

from .reg_base import AbstractRegistration, NullRegistration, BaseRegistration, GDGRegistration
from .reg_gravity import GravityRegistration
from .reg_demons import OriginalDemonsRegistration, DiffeomorphicDemonsRegistration
from .reg_elastix import ElastixRegistration, ElastixGroupwiseRegistration
