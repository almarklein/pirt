
from __future__ import absolute_import, print_function, division 

import numpy as np

from . import Point, Aarray

from .interpolation_ import interp, project, ainterp, aproject
from .interpolation_ import make_samples_absolute, fix_samples_edges
from .interpolation_ import get_cubic_spline_coefs, meshgrid


def get_span_vectors(normal, c, d):
    """ get_span_vectors(normal, prevA, prevB) -> (a,b)
    
    Given a normal, return two orthogonal vectors which are both
    orthogonal to the normal. The vectors are calculated so they match
    as much as possible the previous vectors.
    """
    
    # Calculate a from previous b
    a1 = d.cross(normal)
    
    if a1.norm() < 0.001:
        # The normal and  d point in same or reverse direction
        # -> Calculate b from previous a
        b1 = c.cross(normal)
        a1 = b1.cross(normal)
    
    # Consider the opposite direction
    a2 = -1 * a1
    if c.distance(a1) > c.distance(a2):
        a1 = a2
    
    # Ok, calculate b
    b1 = a1.cross(normal)
    
    # Consider the opposite
    b2 = -1 * b1
    if d.distance(b1) > d.distance(b2):
        b1 = b2

    # Done
    return a1.normalize(), b1.normalize()


class SliceInVolume:
    """ SliceInVolume(self, pos, normal=None, previous=None)
    Defines a slice in a volume. 
    
    The two span vectors are in v and u respectively. In other words,
    vec1 is up, vec2 is right.
    """
    
    def __init__(self, pos, normal=None, previous=None):
        
        # Init vectors
        self._pos = pos
        self._normal = None
        self._vec1 = None
        self._vec2 = None
        
        # Calculate normal
        if normal is None and previous is None:
            self._normal = Point(0,0,-1)
        elif normal is None:
            # Normal is defined by difference in position,
            # and the previous normal
            self._normal = pos - previous._pos            
            self._normal += previous._normal
            self._normal = self._normal.normalize()
        elif normal is not None:
            self._normal = normal.normalize()
        
        # Calculate vec1 and vec2
        if previous is None:
            # Use arbitrary vector
            arbitrary = Point(1,0,0)
            self._vec1 = arbitrary.cross(self._normal)
            if self._vec1.norm() < 0.0001:
                arbitrary = Point(0,1,0)
                self._vec1 = arbitrary.cross(self._normal)
            # second
            self._vec2 = self._vec1.cross(self._normal)
        else:
            # Use previous
            tmp = get_span_vectors(self._normal, previous._vec1, previous._vec2)
            self._vec1, self._vec2 = tmp
        # Normalize
        self._vec1 = self._vec1.normalize()
        self._vec2 = self._vec2.normalize()
    
    def get_slice(self, volume, N=128, spacing=1.0):
        vec1 = self._vec1 * spacing
        vec2 = self._vec2 * spacing
        im = interpolation_.slice_from_volume(volume, self._pos, vec1, vec2, N)
        return Aarray(im, (spacing,spacing))
    
    
    def convert_local_to_global(self, p2d, p3d):
        """ convert_local_to_global(p2d, p3d)
        Convert local 2D points to global 3D points.
        UNTESTED
        """
        pos = self._pos.copy()
        #
        pos.x += p2d.y * self._vec1.x + p2d.x * self._vec2.x
        pos.y += p2d.y * self._vec1.y + p2d.x * self._vec2.y
        pos.z += p2d.y * self._vec1.z + p2d.x * self._vec2.z
        #
        return pos
