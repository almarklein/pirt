# flake8: noqa
""" Pirt - Python Image Registration Toolkit """

__version__ = '2.1.0'


# Check compat
import sys
if sys.version_info < (3, 4):
    raise RuntimeError('Pirt requires at least Python 3.4')

# Imports 

from ._utils import Aarray, PointSet, Parameters

from .gaussfun import (gaussiankernel, gfilter, gfilter2,
                       diffusionkernel, diffuse, diffuse2)

from .pyramid import ScaleSpacePyramid

from .interp import (get_cubic_spline_coefs, meshgrid,
                     warp, project, awarp, aproject,
                     deform_backward, deform_forward,
                     resize, imresize, zoom, imzoom,
                     make_samples_absolute,
                     SliceInVolume)

from .splinegrid import GridInterface, SplineGrid, GridContainer, FieldDescription, FD

from .deform import (Deformation, DeformationIdentity, 
                     DeformationGrid, DeformationField,
                     DeformationGridForward, DeformationFieldForward,
                     DeformationGridBackward, DeformationFieldBackward)

from .reg import *


# Some utils need to be imported to use them
from .randomdeformations import RandomDeformations, create_random_deformation
# from . import experiment
# from . import deformvis
# from . import testing

# Clean up
del sys
