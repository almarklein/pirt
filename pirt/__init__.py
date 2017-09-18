# flake8: noqa
""" Pirt - Python Image Registration Toolkit """

__version__ = '2.0.1'


# Check compat
import sys
if sys.version_info < (3, 4):
    raise RuntimeError('Pirt requires at least Python 3.4')

# Imports 

# todo: replace these with better versions ...
from visvis import Aarray
from visvis import Point, Pointset
from visvis import ssdf

from .gaussfun import (gaussiankernel, gfilter, gfilter2,
                       diffusionkernel, diffuse, diffuse2)

from .pyramid import ScaleSpacePyramid

from .interp import (get_cubic_spline_coefs, meshgrid,
                     warp, project, awarp, aproject,
                     deform_backward, deform_forward,
                     resize, imresize, zoom, imzoom,
                     SliceInVolume)

from pirt.splinegrid import SplineGrid, GridContainer, FieldDescription, FD

from pirt.deform import (Deformation, DeformationIdentity, 
                         DeformationGrid, DeformationField,
                         DeformationGridForward, DeformationFieldForward,
                         DeformationGridBackward, DeformationFieldBackward)

from . import reg


# Some utils need to be imported to use them
from .randomdeformations import RandomDeformations, create_random_deformation
# from . import experiment
# from . import deformvis
# from . import testing

# Clean up
del sys
