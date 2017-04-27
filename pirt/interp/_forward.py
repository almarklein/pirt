import numpy as np
import numba
from numba import cuda

from ._cubic import spline_type_to_id, set_cubic_spline_coefs


def project(data, deltas):
    """ project(data, deltas)
    
    Interpolate data according to the deformations specified in deltas.
    Deltas should be a tuple of numpy arrays similar to 'samples' in
    the warp() function. They represent the relative sample positions.
    
    Applies forward deformations, moving the pixels/voxels to the given
    locations, rather than getting the pixel values from the given 
    locations.
    
    """
    
    # Check
    if len(deltas) != data.ndim:
        tmp = "Samples must contain as many arrays as data has dimensions."
        raise ValueError(tmp)
    
    if data.dtype == np.float32:
        if data.ndim == 1:
            return project1_32(data, *deltas)
        elif data.ndim == 2:
            return project2_32(data, *deltas)
        elif data.ndim == 3:
            return project3_32(data, *deltas)
        else:
            raise RuntimeError('deform_forward not implemented for that dimension.')
    
    elif data.dtype == np.float64:
        if data.ndim == 1:
            return project1_64(data, *deltas)
        elif data.ndim == 2:
            return project2_64(data, *deltas)
        elif data.ndim == 3:
            return project3_64(data, *deltas)
        else:
            raise RuntimeError('deform_forward not implemented for that dimension.')
    else:
        raise RuntimeError('deform_forward not implemented for that data type.')



def aproject(data, samples, *args, **kwargs):
    """ aproject(data, samples, order='linear', spline_type=0.0)
    
    Interpolation in anisotropic array. Like project(), but the
    samples are expressed in world coordimates.    
    
    """
    
    # Check
    if not (hasattr(data, 'sampling') and hasattr(data, 'origin')):
        raise ValueError('aproject() needs the data to be an Aarray.')
    
    # Correct samples
    samples2 = []
    ndim = len(samples)
    for i in range(ndim):
        d = ndim-i-1
        origin = data.origin[d]
        sampling = data.sampling[d]
        samples2.append( (samples[i]-origin) / sampling )
    
    # Interpolate
    return project(data, samples2, *args, **kwargs)
