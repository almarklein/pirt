PIRT - Python Image Registration Toolkit.

Introduction
------------

Pirt provides functionality for image registration. It is the result
of my PhD. There is functionality for a variety of image registration
algorothms. Most notably pirt provides an easy way to use the Elastix
toolkit. Further it implements some algorithms in Cython (Demons and
Gravity).

Image registration itself requires several image processing techniques
and data types, which are also included in this package:

  * pirt.gaussfun - function for Gaussian smoothing and
    derivatives, image pyramid class
  * pirt.interp - interpolation of 1D, 2D and 3D data (nearest, linear,
    and various spline interpolants)
  * pirt.splinegrid - defines a B-spline grid class (for data up to
    three dimensions) and a class to describe a deformation grid
    (consisting of a B-spline grid for each dimension)
  
The registration algoriths are in `pirt.reg`.


Dependencies and installation
-----------------------------

Pirt runtime dependencies:

  * numpy
  * scipy
  * visvis (for now)

Build dependencies:

  * Cython
  * A working C compiler

To install:

  * `pip install pirt`  (need Cython)
  * `conda install pirt -c pyzo` (probably Windows only)


Status and licensing
--------------------

Pirt should be considered alpha status. The API may change. There are no
sphynx docs. There are little unit tests.

Pirt is pretty much research code that I tried to make more or less
user-friendly. It needs much more work to turn it into a proper package
suitable for broad adoption. Unfortunately, I currently lack the
resources to do that. If you have an interest in moving Pirt further,
be my guest!

Pirt is BSD licensed, see LICENSE.txt for more information.
