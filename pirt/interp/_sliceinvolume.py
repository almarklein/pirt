""" Functionality for sampling a 2D slice from a 3D volume.
"""

import numpy as np
import numba

from .. import PointSet, Aarray
from ._backward import warp


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
        
        if isinstance(pos, (tuple, list)):
            pos = PointSet(pos)
        elif isinstance(pos, np.ndarray):
            pos = PointSet(pos)
        elif hasattr(pos, '_is_Point'):  # visvis.Point
            pos = PointSet(pos.data)
        
        assert pos.ndim == 2 and pos.shape == (1, 3)
        
        # Init vectors
        self._pos = pos
        self._normal = None
        self._vec1 = None
        self._vec2 = None
        
        # Calculate normal
        if normal is None and previous is None:
            self._normal = PointSet((0,0,-1))
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
            arbitrary = PointSet((1,0,0))
            self._vec1 = arbitrary.cross(self._normal)
            if self._vec1.norm() < 0.0001:
                arbitrary = PointSet((0,1,0))
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
        im = slice_from_volume(volume, self._pos, vec1, vec2, N)
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


def slice_from_volume(data, pos, vec1, vec2, Npatch, order=3):
    """ slice_from_volume(data, pos, vec1, vec2, Npatch, order=3)
    Samples a square 2D slice from a 3D volume, using a center position
    and two vectors that span the patch. The length of the vectors
    specify the sample distance for the patch.
    """
    
    # Prepare
    sampling, origin = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    if hasattr(data, 'sampling'):
        sampling, origin = data.sampling, data.origin
    
    # Generate sample positions
    x = _slice_samples_from_volume(data, sampling, origin, tuple(pos.flat),
                                   tuple(vec1.flat), tuple(vec2.flat),
                                   Npatch)
    samplesx, samplesy, samplesz = x
                        
    # Sample in 3D volume
    return warp(data, [samplesx, samplesy, samplesz], order)


@numba.jit(nopython=True, nogil=True)
def _slice_samples_from_volume(data, sampling, origin, pos, vec1, vec2, Npatch, order=3):
    
    # Init sample arrays
    samplesx = np.empty((Npatch, Npatch), dtype=np.float32)
    samplesy = np.empty((Npatch, Npatch), dtype=np.float32)
    samplesz = np.empty((Npatch, Npatch), dtype=np.float32)
    
    Npatch2 = Npatch / 2.0
    
    # Set start position
    x, y, z = pos
    
    # Get anisotropy factors
    sam_x = 1.0 / sampling[2]  # Do the division here
    sam_y = 1.0 / sampling[1]
    sam_z = 1.0 / sampling[0]
    ori_x = origin[2]
    ori_y = origin[1]
    ori_z = origin[0]
    
    # Make vectors quick
    v1x, v1y, v1z = vec1
    v2x, v2y, v2z = vec2
    
    # Loop
    for v in range(Npatch):
        vd = v - Npatch2
        for u in range(Npatch):
            ud = u - Npatch2
            
            # Determine sample positions
            samplesx[v, u] = ( (x + vd*v1x + ud*v2x) - ori_x) * sam_x
            samplesy[v, u] = ( (y + vd*v1y + ud*v2y) - ori_y) * sam_y
            samplesz[v, u] = ( (z + vd*v1z + ud*v2z) - ori_z) * sam_z
    
    return samplesx, samplesy, samplesz
