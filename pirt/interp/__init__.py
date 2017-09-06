""" Module interp

Exposes several functions for interpolation, implemented in Numba.
This code is part of pirt, but written to be easily made stand-alone,
or be made a subpackage of another package.

Copyright 2014-2017(C) Almar Klein

"""

# More low level functions
from ._cubic import get_cubic_spline_coefs
from ._backward import warp, awarp
from ._forward import project, aproject
from ._misc import meshgrid, uglyRoot

# More higher level functions
from ._func import deform_backward, deform_forward
from ._func import resize, imresize
from ._func import zoom, imzoom

# Special kinds of functionality
from ._sliceinvolume import SliceInVolume

# Aliases
interp = warp  
