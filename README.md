PIRT - Python Image Registration Toolkit.

Introduction
------------

Pirt is the "Python image registration toolkit". It is a library for
(elastic, i.e. non-regid) image registration of 2D and 3D images with
support for groupwise registration. It has support to constrain the 
deformations to be "diffeomorphic", i.e. without folding or shearing, and 
thus invertable.

Pirt is written in pure Python and uses Numba for speed. It depends on
Numpy, Scipy, Numba. It has an optional dependency on Visvis for visualization,
and on pyelastix for the Elastix registration algorithm.

Pirt implements its own interpolation functions, which, incidentally,
are faster than the corresponding functions in scipy and scikit-image
(after Numba's JIT warmup).

Pirt is hosted on [Bitbucket](https://bitbucket.org/almarklein/pirt)
and has [docs on rtd](http://pirt.readthedocs.io/).

Overview
--------

Image registration itself requires several image processing techniques
and data types, which are also included in this package:

  * pirt.gaussfun - function for Gaussian smoothing and
    derivatives, image pyramid class
  * pirt.interp - interpolation of 1D, 2D and 3D data (nearest, linear,
    and various spline interpolants)
  * pirt.splinegrid - defines a B-spline grid class (for data up to
    three dimensions) and a class to describe a deformation grid
    (consisting of a B-spline grid for each dimension)
  * pirt.deform - defines classes to represent and compose deformations
  * pirt.reg - the actual registration algorithms


Dependencies and installation
-----------------------------

Pirt dependencies:

  * numpy
  * scipy
  * numba
  * visvis (optional)

To install:

  * `pip install pirt`
  * or install from the hg repo


Status and licensing
--------------------

Pirt should be considered alpha/beta status. The API may change. The
core parts are pretty well tested though!

Pirt is BSD licensed, see LICENSE.txt for more information.


History
-------

Pirt was developed during my PhD. It more or less scratched my itch and
was not well tested. It was written in Cython. In 2017 the code has been
refactored to move to Numba, and many tests were added.
