""" pirt.utils

Utilities related to deformations and registration.

Most are not directly needed for pirt to work, but are convenient for
users.

Submodules:
  * deformvis
  * experiment
  * randomdeformations
  
"""

# todo: replace these with better versions ...
from visvis import Aarray
from visvis import Point, Pointset
from visvis import ssdf

from .randomdeformations import RandomDeformations, create_random_deformation
from . import experiment
from . import deformvis
