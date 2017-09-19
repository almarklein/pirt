# flake8: noqa
"""
The splinegrid module implements functionality for spline grids. A
spline grid is used to representa field (i.e. data) using a grid. 
Essentially, spline grids allow the interpolation of sparse data in an
optimally smooth way by adopting a multiscale approach. The only type of
spline that makes sense for this purpose is the B-spline.

In Pirt, spline grids are used to represent deformations, but any kind
of data can be represented, like e.g. image data.

The ``GridInterface`` class is the base class that enables basic grid
properties and functionality. The ``SplineGrid`` class implements an
actual grid that represents a scalar field. The ``GridContainer`` class
can be used to wrap multiple ``SplineGrid`` instances in order to
represent a vector/tensor field (such as color channels or
deformations).
"""

from ._splinegridclasses import GridInterface, SplineGrid, GridContainer, FieldDescription, FD
from ._splinegridclasses import calculate_multiscale_sampling
