import numpy as np

from ..gaussfun import diffuse
from .. import Aarray

from ._backward import warp, awarp  # noqa
from ._forward import project, aproject  # noqa
from ._misc import make_samples_absolute, meshgrid


## Deformations


def deform_backward(data, deltas, order=1, spline_type=0.0):
    """ deform_backward(data, deltas, order=1, spline_type=0.0)
    
    Interpolate data according to the deformations specified in deltas.
    Deltas should be a tuple of numpy arrays similar to 'samples' in
    the warp() function. They represent the relative sample positions 
    expressed in world coordinates.
    
    """
    
    # todo: this function assumes that the shape and sampling of data and deltas is equal.
    # Either document, enforce, or fix that (e.g. by using awarp when needed?)
    
    # Check
    if len(deltas) != data.ndim:
        tmp = "Samples must contain as many arrays as data has dimensions."
        raise ValueError(tmp)
    
    # Create samples
    samples = make_samples_absolute(deltas)
    
    # Interpolate
    result = warp(data, samples, order, spline_type)
    
    # Make Aarray
    # todo: is this necessary? can the Aarray not do this itself?
    if hasattr(data, 'sampling'):
        result = Aarray(result, data.sampling)
        if hasattr(data, 'origin'):
            result.origin = data.origin
    
    # Done
    return result


def deform_forward(data, deltas):
    """deform_forward(data, deltas)
    
    Like deform_backward(), but applied to project (forward deformation).
    
    """
    
    # Check
    if len(deltas) != data.ndim:
        tmp = "Samples must contain as many arrays as data has dimensions."
        raise ValueError(tmp)
    
    # Create samples
    samples = make_samples_absolute(deltas)
    
    # Interpolate
    result = project(data, samples)
    
    # Make Aarray
    # todo: is this necessary? can the Aarray not do this itself?
    if hasattr(data, 'sampling'):
        result = Aarray(result, data.sampling)
        if hasattr(data, 'origin'):
            result.origin = data.origin
    
    # Done
    return result


## Resize and zoom

def resize(data, new_shape, order=3, spline_type=0.0, prefilter=False, extra=False):
    """ resize(data, new_shape, order=3, spline_type=0.0, prefilter=False, extra=False)
    
    Resize the data to the specified new shape.
    
    Parameters
    ----------
    data : numpy array 
        The data to rezize.
    new_shape : tuple 
        The new shape of the data (z-y-x order).
    order : {0,1,3} or {'nearest', 'linear', 'cubic'}
        The interpolation order to use.
    spline_type : float or string
        Only for cubic interpolation. Specifies the type of spline. 
        Can be 'Basis', 'Hermite', 'Cardinal', 'Catmull-rom', 'Lagrange', 
        'Lanczos', 'quadratic', or a float, specifying the tension parameter
        for the Cardinal spline. See the docs of get_cubic_spline_coefs()
        for more information.
    prefilter : bool
       Whether to apply (discrete Gaussian diffusion) anti-aliasing 
       (when downampling). Default False.
    extra : bool
        Whether to extrapolate the data a bit. In this case, each datapoint
        is seen as spanning a space equal to the distance between the data
        points. This is the method used when you resize an image using 
        e.g. paint.net or photoshop. If False, the first and last datapoint
        are exactly on top of the original first and last datapoint (like
        scipy.ndimage.zoom). Default False.
    
    Notes on extrapolating
    ----------------------
    For the sake of simplicity, assume that the new shape is exactly
    twice that of the original.
    When extra if False, the sampling between the pixels is not a factor 2 
    of the original. When extra is True, the sampling decreases with a 
    factor of 2, but the data now has an offset. Additionally, extrapolation
    is performed, which is less accurate than interpolation
    
    """
    
    # Check new_shape
    if not isinstance(new_shape, (tuple,list)):
        raise ValueError('new_shape must be a tuple or list.')
    elif not len(new_shape) == len(data.shape):
        raise ValueError('new_shape must contain as much values as data has dimensions.')
    new_shape = [int(round(n)) for n in new_shape]
    
    # Get shape, sampling and origin
    shape = data.shape
    sampling = [1.0 for s in shape]
    origin = [0.0 for s in shape]
    if hasattr(data, 'sampling'):
        sampling = data.sampling
    if hasattr(data, 'origin'):
        origin = data.origin
    
    # Init lists
    ranges = []
    sampling2 = []
    origin2 = []
    
    for s, n, sam, ori in zip(shape, new_shape, sampling, origin):        
        
        if extra:
            
            # Get full range (expressed in "pixels")
            dmin, dmax = -0.5, s-0.5
            drange = dmax-dmin # == s
            
            # Step size (n-1 steps in between the pixels, and 0.5 steps at each side)
            dstep = float(drange) / n  
            dstep2 = 0.5 * dstep
            
            # Get sampling and offset
            sampling2.append( dstep*sam )
            origin2.append( ori+(dmin+dstep2)*sam )
            
            # Get range
            r = np.linspace(dmin+dstep2, dmax-dstep2, n)
            ranges.append(r)
        
        else:
            
            # Get full range (expressed in "pixels")
            dmin, dmax = 0, s-1
            drange = dmax-dmin # == s
            
            # Step size (the outer points are exactly at the dmin and dmax)
            dstep = float(drange) / (n-1)
            
            # Get sampling and offset   
            sampling2.append( dstep*sam )
            origin2.append( ori )
            
            # Get range
            r = np.linspace(dmin, dmax, n)
            ranges.append(r)
    
    
    # Anti-aliasing
    def foo(x):
        if x < 1.0: return 0.8/x # 1.0 is better, but people like sharp images
        else: return 0.0
    factors = [float(s1)/s2 for s1,s2 in zip(new_shape, shape)]
    sigmas = [foo(f) for f in factors]
    if prefilter and sum(sigmas):
        data = diffuse(data, sigmas)
    
    # Interpolate (first make ranges x-y-z)
    ranges.reverse()
    grids = meshgrid(ranges)
    data2 = warp(data, grids, order, spline_type)
    
    # Make Aarray
    return Aarray(data2, sampling2, origin2)


def imresize(data, new_shape, order=3):
    """ imzoom(data, factor, order=3)
    
    Convenience function to resize the image data (1D, 2D or 3D).
    
    This function uses pirt.resize() with 'prefilter' and 'extra' set to True.
    This makes it more suitble for generic image resizing. Use pirt.resize()
    for more fine-grained control.
    
    Parameters
    ----------
    data : numpy array 
        The data to rezize.
    new_shape : tuple 
        The new shape of the data (z-y-x order).
    order : {0,1,3} or {'nearest', 'linear', 'cubic'}
        The interpolation order to use.
    
    """
    return resize(data, new_shape, order, 0.0, True, True)


def zoom(data, factor, order=3, spline_type=0.0, prefilter=False, extra=False):
    """ zoom(data, factor, order=3, spline_type=0.0, prefilter=False, extra=False)
    
    Resize the data with the specified factor. The default behavior is
    the same as scipy.ndimage.zoom(), but three times faster.
    
    Parameters
    ----------
    data : numpy array 
        The data to rezize.
    factor : scalar or tuple 
        The resize factor, optionally for each dimension (z-y-z order).
    order : {0,1,3} or {'nearest', 'linear', 'cubic'}
        The interpolation order to use.
    spline_type : float or string
        Only for cubic interpolation. Specifies the type of spline. 
        Can be 'Basis', 'Hermite', 'Cardinal', 'Catmull-rom', 'Lagrange', 
        'Lanczos', 'quadratic', or a float, specifying the tension parameter
        for the Cardinal spline. See the docs of get_cubic_spline_coefs()
        for more information.
    prefilter : bool
       Whether to apply (discrete Gaussian diffusion) anti-aliasing 
       (when downampling). Default False.
    extra : bool
        Whether to extrapolate the data a bit. In this case, each datapoint
        is seen as spanning a space equal to the distance between the data
        points. This is the method used when you resize an image using 
        e.g. paint.net or photoshop. If False, the first and last datapoint
        are exactly on top of the original first and last datapoint (like
        numpy.zoom). Default False.
    
    Notes on extrapolating
    ----------------------
    For the sake of simplicity, assume a resize factor of 2.
    When extra if False, the sampling between the pixels is not a factor 2
    of the original. When extra is True, the sampling decreases with a
    factor of 2, but the data now has an offset. Additionally, extrapolation
    is performed, which is less accurate than interpolation
    
    """
   
    # Process factor
    if isinstance(factor, np.ndarray) and factor.size == 1:
        factor = float(factor)
    if isinstance(factor, (float, int)):
        factor = [factor for i in data.shape]
    
    # Check factor
    if not isinstance(factor, (list, tuple)):
        raise ValueError('Factor must be a float or tuple/list.')
    if len(factor) != data.ndim:
        raise ValueError('Factor len does not match ndim of data.')
    
    # Calculate new shape
    new_shape = [float(f)*s for f,s in zip(factor, data.shape)]
    new_shape = [int(round(s)) for s in new_shape]
    
    # Resize
    return resize(data, new_shape, order, spline_type, prefilter, extra)


def imzoom(data, factor, order=3):
    """ imzoom(data, factor, order=3)
    
    Convenience function to resize the image data (1D, 2D or 3D) with the
    specified factor.
    
    This function uses pirt.interp.resize() with 'prefilter' and 'extra'
    set to True. This makes it more suitble for generic image resizing.
    Use pirt.resize() for more fine-grained control.
    
    Parameters
    ----------
    data : numpy array 
        The data to rezize.
    factor : scalar or tuple 
        The resize factor, optionally for each dimension (z-y-x order).
    order : {0,1,3} or {'nearest', 'linear', 'cubic'}
        The interpolation order to use.
    
    """
    return zoom(data, factor, order, 0.0, True, True)
