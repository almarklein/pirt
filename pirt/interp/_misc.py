import numpy as np
import numba


def meshgrid(*args):
    """ Meshgrid implementation for 1D, 2D, 3D and beyond. 
    
    meshgrid(nx, ny) will create meshgrids with the specified shapes.
    
    meshgrid(ndarray) will create meshgrids corresponding to the shape
    of the given array (which must have 2 or 3 dimension).
    
    meshgrid([2,3,4], [1,2,3], [4,1,2]) uses the supplied values to 
    create the grids. These lists can also be numpy arrays.
    
    Returns a tuple with the grids in x-y-z order, with arrays of type float32.
    """
    
    # Test args
    if len(args) == 1:
        args = args[0]
    if isinstance(args, np.ndarray) and args.ndim in [2,3]:
        args = tuple(reversed(args.shape))
    if not isinstance(args, (list, tuple)):
        raise ValueError('Invalid argument for meshgrid.')
    iterators = []
    for arg in args:
        if isinstance(arg, int):
            iterators.append( np.arange(arg, dtype=np.float32) )
        elif isinstance(arg, list):
            iterators.append( np.array(arg, dtype=np.float32) )
        elif isinstance(arg, np.ndarray):
            iterators.append( arg )
        else:
            raise ValueError('Invalid argument for meshgrid.')
    
    # Swizzle iterators because Numpy's meshgrid behaves different
    iterators.reverse()
    if len(iterators) > 1:
        iterators[0], iterators[1] = iterators[1], iterators[0]
    
    # Use Numpy
    res = np.meshgrid(*iterators, indexing='xy')
    res = [a.astype(np.float32) for a in res]
    
    # Swizzle back and return
    if len(iterators) > 1:
        res[0], res[1] = res[1], res[0]
    return tuple(reversed(res))


# todo: this seems not to be used anymore
@numba.jit(nopython=True, nogil=True)
def uglyRoot(n):
    """ uglyRoot(n)
    Calculates an approximation of the square root using
    (a few) Newton iterations.
    """
    x = 1.0    
    x = x - (x * x - n) / (2.0 * x)
    x = x - (x * x - n) / (2.0 * x)
    x = x - (x * x - n) / (2.0 * x)
    return x


def make_samples_absolute(samples):
    """ make_samples_absolute(samples)
    
    Note: this function is intended for sampes that represent a 
    deformation; the number of dimensions of each array should 
    match the number of arrays.
    
    Given a tuple of arrays that represent a relative deformation 
    expressed in world coordinates (x,y,z order), returns a tuple
    of sample arrays that represents the absolute sample locations in pixel
    coordinates. It is assumed that the sampling of the data is the same
    as for the sample arrays. The origin property is not used.
    
    This process can also be done with relative ease by adding a meshgrid
    and then using awarp() or aproject(). But by combining it in 
    one step, this process becomes faster and saves memory. Note that
    the deform_*() functions use this function.
    
    """
    
    ndim = len(samples)
    absolute_samples = []
    for i in range(ndim):
        
        # Get array and check
        sample_array = samples[i]
        if sample_array.ndim != ndim:
            raise ValueError("make_samples_absolute: the number of dimensions"+
                " of each array should  match the number of arrays.")
        
        # Get dimension corresponding to this sampling array
        d = ndim-i-1
        
        # Get sampling
        sampling = 1.0
        if hasattr(sample_array, 'sampling'):
            sampling = sample_array.sampling[d]
        
        # Instantiate result
        result = np.empty_like(sample_array)
        absolute_samples.append(result)
        
        # Apply
        if sample_array.ndim == 1:
            make_samples_absolute1(sample_array, result, sampling, d)
        if sample_array.ndim == 2:
            make_samples_absolute2(sample_array, result, sampling, d)
        if sample_array.ndim == 3:
            make_samples_absolute3(sample_array, result, sampling, d)
    
    # Done
    return tuple(absolute_samples)


@numba.jit(nopython=True, nogil=True)
def make_samples_absolute1(samples_, result_, sampling, dim=0):
    
    # Define variables
    sampling_i = 1.0/sampling
    Nx = samples_.shape[0]
    
    if dim == 0:
        for x in range(Nx):
            result_[x] = x +  samples_[x] * sampling_i


@numba.jit(nopython=True, nogil=True)
def make_samples_absolute2(samples_, result_, sampling, dim=0):
    
    sampling_i = 1.0/sampling
    Ny = samples_.shape[0]
    Nx = samples_.shape[1]
    
    if dim == 0:
        for y in range(Ny):
            for x in range(Nx):
                result_[y,x] = y +  samples_[y,x] * sampling_i
    elif dim == 1:
        for y in range(Ny):
            for x in range(Nx):
                result_[y,x] = x +  samples_[y,x] * sampling_i


@numba.jit(nopython=True, nogil=True)
def make_samples_absolute3(samples_, result_, sampling, dim=0):
    
    # Define variables
    sampling_i = 1.0/sampling
    Nz = samples_.shape[0]
    Ny = samples_.shape[1]
    Nx = samples_.shape[2]
    
    if dim == 0:
        for z in range(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    result_[z,y,x] = z +  samples_[z,y,x] * sampling_i
    elif dim == 1:
        for z in range(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    result_[z,y,x] = y +  samples_[z,y,x] * sampling_i
    elif dim == 2:
        for z in range(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    result_[z,y,x] = x +  samples_[z,y,x] * sampling_i
