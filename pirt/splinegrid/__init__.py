# flake8: noqa
""" Module pirt.splinegrid

This module implements functionality for spline grids. The GridInterface
class is the base class that enables basic grid properties and 
functionality. The SplineGrid class implements an actual grid that 
represents a scalar field. The GridContainer class can be used to
wrap multiple SplineGrid instances in order to represent a vector/tensor
field (such as color or deformations).

"""
from .splinegrid import GridInterface, SplineGrid, GridContainer, FieldDescription, FD
from .splinegrid import calculate_multiscale_sampling