# flake8: noqa
""" 
The deform module implements classes to represent deformations: The
``DeformationGrid`` represents a deformation in world coordinates using
a spline grid, and the ``DeformationField`` represents a deformation
in world coordinates using an array for each dimension; it describes
the deformation for each pixel/voxel.

The aforementioned classes are actually base classes; one should use
``DeformationFieldBackward``, ``DeformationFieldForward``,
``DeformationGridBackward`` or ``DeformationGridForward``.

"""

from ._deformbase import Deformation
from ._deformgrid import DeformationGrid
from ._deformfield import DeformationField
from ._subs import (DeformationIdentity,
                    DeformationGridForward, DeformationGridBackward,
                    DeformationFieldForward, DeformationFieldBackward)
