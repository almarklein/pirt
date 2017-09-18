# flake8: noqa
""" Module pirt.deform

This moduls implements two classes to represent deformations:

  * DeformationGrid: represents a deformation in world coordinates using
    a spline grid.
  * DeformationField: represents a deformation in world coordinates using
    an array for each dimension; it describes the deformation for each
    pixel/voxel.

Note that these are actually base classes, one should use
DeformationFieldBackward, DeformationFieldForward,
DeformationGridBackward and DeformationGridForward.

"""

from ._deformbase import Deformation
from ._deformgrid import DeformationGrid
from ._deformfield import DeformationField
from ._subs import (DeformationIdentity,
                    DeformationGridForward, DeformationGridBackward,
                    DeformationFieldForward, DeformationFieldBackward)
