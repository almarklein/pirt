""" Module interp

Exposes several functions for interpolation, implemented in Cython.
This code is part of pirt, but written to be easily made stand-alone,
or be made a subpackage of another package.

Copyright 2014-2017(C) Almar Klein

"""

from __future__ import absolute_import, print_function, division 

import os

# Need some stuff from pirt utils. These can easily be transfered if necessary
from pirt.utils import Point, Aarray
from pirt.gaussfun import diffuse

from ._cubiclut import get_cubic_spline_coefs, get_lut, get_coef, get_coef_linear
from ._backward import warp, awarp
from ._forward import project, aproject
from ._misc import meshgrid, uglyRoot


# Import user-friendly functions
from ._func import deform_backward, deform_forward
from ._func import resize, imresize
from ._func import zoom, imzoom

from ._sliceinvolume import SliceInVolume
