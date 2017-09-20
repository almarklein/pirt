Welcome to Pirt's documentation!
================================

Pirt is the "Python image registration toolkit". It is a library for
(elastic, i.e. non-regid) image registration of 2D and 3D images with
support for groupwise registration. It has support to constrain the 
deformations to be "diffeomorphic", i.e. without folding or shearing, and 
thus invertable.

Pirt is written in pure Python and uses Numba for speed. It depends on
Numpy, Scipy, Numba. It has an optional dependency on Visvis for
visualization.

Pirt implements its own interpolation functions, which, incidentally,
are faster than the corresponding functions in scipy and scikit-image
(after Numba's JIT warmup).

The functionality inside Pirt is implemented over a series of submodules,
but (almost) all functions and classes are available in the main namespace.

The code lives on `Bitbucket <https://bitbucket.org/almarklein/pirt>`_.
Also check out the `examples <https://bitbucket.org/almarklein/pirt/src/tip/examples/>`_
or the `docs <http://pirt.readthedocs.io>`_.

.. toctree::
   :maxdepth: 1
   :caption: Reference:
   
   gaussfun
   interp
   deform
   splinegrid
   reg
   aarrays_and_pointset
   apps


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
