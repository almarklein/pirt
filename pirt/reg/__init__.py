""" Registration objects
"""

from __future__ import absolute_import, print_function, division 

from .reg_base import AbstractRegistration, NullRegistration, BaseRegistration, GDGRegistration
from .reg_gravity import GravityRegistration
from .reg_demons import OriginalDemonsRegistration, DiffeomorphicDemonsRegistration
from .reg_elastix import ElastixRegistration, ElastixGroupwiseRegistration
