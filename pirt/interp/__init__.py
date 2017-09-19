# flake8: noqa
# Copyright 2014-2017(C) Almar Klein
""" 
The interp module implements several functions for interpolation,
implemented in Numba.
"""

# More low level functions
from ._cubic import get_cubic_spline_coefs

from ._misc import meshgrid

from ._backward import warp, awarp
from ._forward import project, aproject
from ._misc import make_samples_absolute  #, uglyRoot

# More higher level functions
from ._func import deform_backward, deform_forward
from ._func import resize, imresize
from ._func import zoom, imzoom

# Special kinds of functionality
from ._sliceinvolume import SliceInVolume

# Aliases
interp = warp  
