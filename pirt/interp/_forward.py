""" Low level project() function implemented with Numba to make it super fast.
"""

import numpy as np
import numba


@numba.jit(nopython=True, nogil=True)
def floor(i):
    if i >= 0:
        return int(i)
    else:
        return int(i) - 1


@numba.jit(nopython=True, nogil=True)
def ceil(i):
    if i >= 0:
        return int(i + 0.9999999999999999)
    else:
        return int(i)


def project(data, samples):
    """ project(data, samples)
    
    Interpolate data to the positions specified by samples (pixel coordinates).
    
    In contrast to warp(), the project() function applies forward
    deformations, moving the pixels/voxels to the given locations,
    rather than getting the pixel values from the given locations.
    Although this may feel closer to how one would like to think about
    deformations, this function is slower and has no options to determine
    the interpolation, because there is no interpolation, but splatting.
    
    Parameters
    ----------
    data : array (float32 or float64)
        Data to interpolate, can be 1D, 2D or 3D.
    samples : tuple with numpy arrays
        Each array specifies the sample position for one dimension (in 
        x-y-z order).  In contrast to warp(), each array must have the same
        shape as data. Can also be a stacked array as in skimage's warp()
        (in z-y-x order).
    
    Returns
    -------
    result : array 
        The result is of the same type and shape as the data array.
    """
    
    # Check data
    if not isinstance(data, np.ndarray):
        raise ValueError('data must be a numpy array.')
    elif data.ndim > 3:
        raise ValueError('can not interpolate data with such many dimensions.')
    # With Numba we can allow any dtype!
    
    # Check samples
    if isinstance(samples, tuple):
        pass
    elif isinstance(samples, list):
        samples = tuple(samples)
    elif isinstance(samples, np.ndarray) and samples.shape[0] == data.ndim and samples[0].ndim > 0:
        # skimage API, note that this is z-y-x order!
        samples = tuple(reversed([samples[i] for i in range(samples.shape[0])]))
    elif data.ndim==1:
        samples = (samples,)
    else:
        raise ValueError("samples must be a tuple of arrays.")
    
    if len(samples) != data.ndim:
        tmp = "samples must contain as many arrays as data has dimensions."
        raise ValueError(tmp)
    for s in samples:
        if not isinstance(s, np.ndarray):
            raise ValueError("values in samples must all be numpy arrays.")
        if s.shape != data.shape:  # note that this is quite a bit more restrictive than in warp()
            raise ValueError("sample arrays must all have the same shape as the data.")
        # With Numba we can allow any dtype for the samples!
    
    # Prepare empty result array - important to use zeros(), not empty()!
    result = np.zeros(samples[0].shape, data.dtype)  # shape of samples, dtype of data
    
    # Go
    if data.ndim == 1:
        project1(data, result, samples[0])
    elif data.ndim == 2:
        project2(data, result, samples[0], samples[1])
    elif data.ndim == 3:
        project3(data, result, samples[0], samples[1], samples[2])
    
    # Done
    return result


def aproject(data, samples):
    """ aproject(data, samples)
    
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
    return project(data, samples2)


@numba.jit(nopython=True, nogil=True)
def project1(data_, result_, deformx_):
    
    Nx = data_.shape[0]
    
    coeff_ = np.zeros(data_.shape, dtype=np.float32)
    
    for ix1 in range(0, Nx):
        
        # Calculate location to map to
        x2 = deformx_[ix1]
        
        # Select where the surrounding pixels map to.
        # This defines the region that we should fill in. This region
        # is overestimated as it is assumed rectangular in the destination,
        # which is not true in general.
        x2_min = x2
        x2_max = x2        
        for ix3 in range(-1,2):
            if ix3==0:
                continue
            if ix1+ix3 < 0:
                x2_min = x2 - 1000000.0 # Go minimal
                continue
            if ix1+ix3 >= Nx:
                x2_max = x2 + 1000000.0 # Go maximal
                continue
            # x border
            val = deformx_[ix1+ix3]
            x2_min = min(x2_min, val)
            x2_max = max(x2_max, val)
        
        
        # Limit to bounds and make integer
        x2_min = max(0, min(Nx-1, x2_min ))
        x2_max = max(0, min(Nx-1, x2_max ))
        #
        ix2_min = max(0, floor(x2_min) )   # use max/min again to be sure
        ix2_max = min(Nx-1, ceil(x2_max) )
        
        # Calculate max range to determine kernel size
        rangeMax = 0.1
        rangeMax = max( rangeMax, max(abs(x2_min-x2), abs(x2_max-x2)) )
        rangeMax = 1.0 / rangeMax # pre-divide
        
        # Sample value
        val = data_[ix1]
        
        # Splat value in destination
        for ix2 in range(ix2_min, ix2_max+1):
            
            # Calculate weights and make sure theyre > 0
            wx = 1.0 - rangeMax * abs(ix2 - x2)
            w = max(0.0, wx)
            
            # Assign values
            result_[ix2  ] += val * w
            coeff_[ ix2  ] +=       w
    
    
    # Divide by the coeffeicients
    for ix2 in range(Nx):
        
        c = coeff_[ix2]
        if c>0:
            result_[ix2] = result_[ix2] / c


@numba.jit(nopython=True, nogil=True)
def project2(data_, result_, deformx_, deformy_):
    
    Ny = data_.shape[0]
    Nx = data_.shape[1]
    
    coeff_ = np.zeros(data_.shape, dtype=np.float32)
    
    for iy1 in range(0, Ny):
        for ix1 in range(0, Nx):
            
            # Calculate location to map to
            y2 = deformy_[iy1,ix1]
            x2 = deformx_[iy1,ix1]
            
            # Select where the surrounding pixels map to.
            # This defines the region that we should fill in. This region
            # is overestimated as it is assumed rectangular in the destination,
            # which is not true in general.
            x2_min = x2
            x2_max = x2
            y2_min = y2
            y2_max = y2
            for iy3 in range(-1,2):
                for ix3 in range(-1,2):
                    if iy3*ix3==0:
                        continue
                    if iy1+iy3 < 0:
                        y2_min = y2 - 1000000.0 # Go minimal
                        continue
                    if ix1+ix3 < 0:
                        x2_min = x2 - 1000000.0 # Go minimal
                        continue
                    if iy1+iy3 >= Ny:
                        y2_max = y2 + 1000000.0 # Go maximal
                        continue
                    if ix1+ix3 >= Nx:
                        x2_max = x2 + 1000000.0 # Go maximal
                        continue
                    # x border
                    val = deformx_[iy1+iy3,ix1+ix3]
                    x2_min = min(x2_min, val)
                    x2_max = max(x2_max, val)
                    # y border
                    val = deformy_[iy1+iy3,ix1+ix3]
                    y2_min = min(y2_min, val)
                    y2_max = max(y2_max, val)
            
            # Limit to bounds and make integer
            x2_min = max(0, min(Nx-1, x2_min ))
            x2_max = max(0, min(Nx-1, x2_max ))
            y2_min = max(0, min(Ny-1, y2_min ))
            y2_max = max(0, min(Ny-1, y2_max ))
            #
            ix2_min = max(0, floor(x2_min) )   # use max/min again to be sure
            ix2_max = min(Nx-1, ceil(x2_max) )
            iy2_min = max(0, floor(y2_min) )
            iy2_max = min(Ny-1, ceil(y2_max) )
            
            # Calculate max range to determine kernel size
            rangeMax = 0.1
            rangeMax = max( rangeMax, max(abs(y2_min-y2), abs(y2_max-y2)) )
            rangeMax = max( rangeMax, max(abs(x2_min-x2), abs(x2_max-x2)) )
            
            #rangeMax_sum += rangeMax
            rangeMax = 1.0 / rangeMax # pre-divide
            
            # Sample value
            val = data_[iy1,ix1]
            
            # Splat value in destination
            for iy2 in range(iy2_min, iy2_max+1):
                for ix2 in range(ix2_min, ix2_max+1):
                    
                    # Calculate weights and make sure theyre > 0
                    wy = 1.0 - rangeMax * abs(iy2 - y2)
                    wx = 1.0 - rangeMax * abs(ix2 - x2)
                    w = max(0.0, wy) * max(0.0, wx)
                    
                    # Assign values
                    result_[iy2  ,ix2  ] += val * w
                    coeff_[ iy2  ,ix2  ] +=       w
    
    # Divide by the coeffeicients
    for iy2 in range(Ny):
        for ix2 in range(Nx):
            
            c = coeff_[iy2,ix2]
            if c>0:
                result_[iy2,ix2] /= c


@numba.jit(nopython=True, nogil=True)
def project3(data_, result_, deformx_, deformy_, deformz_):
    
    Nz = data_.shape[0]
    Ny = data_.shape[1]
    Nx = data_.shape[2]
    
    coeff_ = np.zeros(data_.shape, dtype=np.float32)
    
    for iz1 in range(0, Nz):
        for iy1 in range(0, Ny):
            for ix1 in range(0, Nx):
                
                # Calculate location to map to
                z2 = deformz_[iz1,iy1,ix1]
                y2 = deformy_[iz1,iy1,ix1]
                x2 = deformx_[iz1,iy1,ix1]
                
                # Select where the surrounding pixels map to.
                # This defines the region that we should fill in. This region
                # is overestimated as it is assumed rectangular in the destination,
                # which is not true in general.
                x2_min = x2
                x2_max = x2
                y2_min = y2
                y2_max = y2
                z2_min = z2
                z2_max = z2
                for iz3 in range(-1,2):
                    for iy3 in range(-1,2):
                        for ix3 in range(-1,2):
                            if iz3*iy3*ix3==0:
                                continue
                            if iz1+iz3 < 0:
                                z2_min = z2 - 1000000.0 # Go minimal
                                continue
                            if iy1+iy3 < 0:
                                y2_min = y2 - 1000000.0 # Go minimal
                                continue
                            if ix1+ix3 < 0:
                                x2_min = x2 - 1000000.0 # Go minimal
                                continue
                            if iz1+iz3 >= Nz:
                                z2_max = z2 + 1000000.0 # Go maximal
                                continue
                            if iy1+iy3 >= Ny:
                                y2_max = y2 + 1000000.0 # Go maximal
                                continue
                            if ix1+ix3 >= Nx:
                                x2_max = x2 + 1000000.0 # Go maximal
                                continue
                            # x border
                            val = deformx_[iz1+iz3,iy1+iy3,ix1+ix3]
                            x2_min = min(x2_min, val)
                            x2_max = max(x2_max, val)
                            # y border
                            val = deformy_[iz1+iz3,iy1+iy3,ix1+ix3]
                            y2_min = min(y2_min, val)
                            y2_max = max(y2_max, val)
                            # z border
                            val = deformz_[iz1+iz3,iy1+iy3,ix1+ix3]
                            z2_min = min(z2_min, val)
                            z2_max = max(z2_max, val)
                
                # Limit to bounds and make integer
                x2_min = max(0, min(Nx-1, x2_min ))
                x2_max = max(0, min(Nx-1, x2_max ))
                y2_min = max(0, min(Ny-1, y2_min ))
                y2_max = max(0, min(Ny-1, y2_max ))
                z2_min = max(0, min(Nz-1, z2_min ))
                z2_max = max(0, min(Nz-1, z2_max ))
                #
                ix2_min = max(0, floor(x2_min) )   # use max/min again to be sure
                ix2_max = min(Nx-1, ceil(x2_max) )
                iy2_min = max(0, floor(y2_min) )
                iy2_max = min(Ny-1, ceil(y2_max) )
                iz2_min = max(0, floor(z2_min) )
                iz2_max = min(Nz-1, ceil(z2_max) )
                
                # Calculate max range to determine kernel size
                rangeMax = 0.1
                rangeMax = max( rangeMax, max(abs(z2_min-z2), abs(z2_max-z2)) )
                rangeMax = max( rangeMax, max(abs(y2_min-y2), abs(y2_max-y2)) )
                rangeMax = max( rangeMax, max(abs(x2_min-x2), abs(x2_max-x2)) )
                rangeMax = 1.0 / rangeMax # pre-divide
                
                # Sample value
                val = data_[iz1,iy1,ix1]
                
                # Splat value in destination
                for iz2 in range(iz2_min, iz2_max+1):
                    for iy2 in range(iy2_min, iy2_max+1):
                        for ix2 in range(ix2_min, ix2_max+1):
                            
                            # Calculate weights and make sure theyre > 0
                            wz = 1.0 - rangeMax * abs(iz2 - z2)
                            wy = 1.0 - rangeMax * abs(iy2 - y2)
                            wx = 1.0 - rangeMax * abs(ix2 - x2)
                            w = max(0.0, wz) * max(0.0, wy) * max(0.0, wx)
                            
                            # Assign values
                            result_[iz2, iy2, ix2] += val * w
                            coeff_[ iz2, iy2, ix2] +=       w
    
    
    # Divide by the coeffeicients
    for iz2 in range(Nz):
        for iy2 in range(Ny):
            for ix2 in range(Nx):
                
                c = coeff_[iz2,iy2,ix2]
                if c>0:
                    result_[iz2,iy2,ix2] = result_[iz2,iy2,ix2] / c
