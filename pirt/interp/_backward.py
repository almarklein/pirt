# from math import floor

import numpy as np
import numba
from numba import cuda

from ._cubic import spline_type_to_id, set_cubic_spline_coefs
from ._cubic import cubicsplinecoef_catmullRom, cubicsplinecoef_cardinal, cubicsplinecoef_quadratic


@numba.jit(nopython=True, nogil=True)
def floor(i):
    if i >= 0:
        return int(i)
    else:
        return int(i) - 1


def warp(data, samples, order=1, spline_type=0.0):
    """ warp(data, samples, order='linear', spline_type=0.0)
    
    Interpolate (sample) data at the positions specified by samples 
    (pixel coordinates).
    
    Parameters
    ----------
    data : array (float32 or float64)
        Data to interpolate, can be 1D, 2D or 3D.
    sample : tuple with numpy arrays
        Each array specifies the sample position for one dimension (in 
        x-y-z order). 
    order : integer or string
        Order of interpolation. Can be 0:'nearest', 1:'linear', 2: 'quadratic',
        3:'cubic'. 
    spline_type : float or string
        Only for cubic interpolation. Specifies the type of spline.
        Can be 'Basis', 'Hermite', 'Cardinal', 'Catmull-rom', 'Lagrange', 
        'Lanczos', 'quadratic', or a float, specifying the tension 
        parameter for the Cardinal spline. See the docs of 
        get_cubic_spline_coefs() for more information.
    
    Returns
    -------
    result : array 
        The result is of the same shape of the samples arrays, which 
        can be of any shape. This flexibility makes this function suitable
        as a base function used by many of the other high-level functions
        defined in this module.
    
    Notes
    -----------
    The input data should be of float32 or float64 and can have up to 
    three dimensions.
    
    An order of interpolation of 2 would naturally correspond to
    quadratic interpolation. However, due to its uneven coefficients
    it reques the same support (and speed) as a cubic interpolant.
    This implementation adds the two quadratic polynomials. Note that
    you can probably better use order=3 with a Catmull-Rom spline, which
    corresponds to the linear interpolation of the two quadratic polynomials.
    
    It can be shown (see Thevenaz et al. 2000 "Interpolation Revisited") 
    that interpolation using a Cardinal spline is equivalent to 
    interpolating-B-spline interpolation.
    """
    
    # Check data
    if not isinstance(data, np.ndarray):
        raise ValueError('data must be a numpy array.')
    elif data.ndim > 3:
        raise ValueError('can not interpolate data with such many dimensions.')
    elif data.dtype not in [np.float32, np.float64, np.int16]:
        raise ValueError('data must be float32 or float64.')
        # todo: this dtype requirement is not strictly necessary anymore
    
    # todo: I think we can support skimage-like samples out of the box
    
    # Check samples
    if isinstance(samples, tuple):
        pass
    elif isinstance(samples, list):
        samples = tuple(samples)
    elif data.ndim==1:
        samples = (samples,)
    else:
        raise ValueError("samples must be a tuple of arrays.")
    if len(samples) != data.ndim:
        tmp = "samples must contain as many arrays as data has dimensions."
        raise ValueError(tmp)
    for s in samples:
        if not isinstance(data, np.ndarray):
            raise ValueError("values in samples must all be numpy arrays.")
        if s.shape != samples[0].shape:
            raise ValueError("sample arrays must all have the same shape.")
        if s.dtype != np.float32:
            raise ValueError("sample arrays must be of type float32.")
    
    # Check order
    orders = {'nearest': 0, 'linear': 1, 'quadratic': 2, 'cubic': 3}
    if isinstance(order, str):
        try:
            order = orders[order]
        except KeyError:
            raise ValueError('Unknown order of interpolation.')
    if order not in [0, 1, 2, 3]:
        raise ValueError('Invalid order of interpolation.')
    if order == 2:
        order = 3
        spline_type = 'quadratic'
    
    # Prepare spline_id and empty result array
    spline_id = spline_type_to_id(spline_type)
    result = np.empty(samples[0].shape, data.dtype)
    
    # Some code snippets to do it on Cuda, would need to be incorporated to make it work
    gpu = False
    if gpu:
        threadsperblock = 32
        blockspergrid = (samplesx_.size + (threadsperblock - 1)) # threadperblock
        interp2_cuda[blockspergrid, threadsperblock](image_, result_, samplesx_, samplesy_, order, coeffx.N, cuda.to_device(coeffx._LUT.astype('float64')))
        result_.copy_to_host(result.ravel())  # only this array is copied back
    
    # Go
    if data.ndim == 1:
        warp1(data, result.ravel(), samples[0].ravel(), order, spline_id)
    elif data.ndim == 2:
        warp2(data, result.ravel(), samples[0].ravel(), samples[1].ravel(), order, spline_id)
    elif data.ndim == 3:
        warp3(data, result.ravel(), samples[0].ravel(), samples[1].ravel(), samples[2].ravel(), order, spline_id)
    
    # Make Anisotropic array if input data was too
    # --> No: We do not know what the sample points are
    
    # Done
    return result


def awarp(data, samples, *args, **kwargs):
    """ awarp(data, samples, order='linear', spline_type=0.0)
    
    Interpolation in anisotropic array. Like warp(), but the
    samples are expressed in world coordimates.    
    
    """
    
    # Check
    if not (hasattr(data, 'sampling') and hasattr(data, 'origin')):
        raise ValueError('awarp() needs the data to be an Aarray.')
    
    # Correct samples
    samples2 = []
    ndim = len(samples)
    for i in range(ndim):
        d = ndim-i-1
        origin = data.origin[d]
        sampling = data.sampling[d]
        samples2.append( (samples[i]-origin) / sampling )
    
    # Interpolate
    return warp(data, samples2, *args, **kwargs)


@numba.jit(nopython=True, nogil=True)
def warp1(data_, result_, samplesx_, order, spline_id):
    
    Ni = samplesx_.size
    Nx = data_.shape[0]
    
    ccx = np.empty((4, ), np.float64)
    
    if order == 3:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            
            if ix >= 1 and ix < Nx-2:
                # Cubic interpolation
                
                # Get coefficients.
                if spline_id <= 1.0:  # By far most common, make it fast!
                    cubicsplinecoef_cardinal(tx, ccx, spline_id)  # tension=spline_id
                elif spline_id == 99.0:
                    cubicsplinecoef_quadratic(tx, ccx)
                else:  # Select spline type, slower, but ok, because these make less sense, or are slow anyway
                    set_cubic_spline_coefs(tx, spline_id, ccx)
                
                val =  data_[ix-1] * ccx[0]
                val += data_[ix  ] * ccx[1]
                val += data_[ix+1] * ccx[2]
                val += data_[ix+2] * ccx[3]
                result_[i] = val
            
            elif dx>=-0.5 and dx<=Nx-0.5:
                # Edge effects
                
                # Get coefficients. Slower, but only needed at edges.
                set_cubic_spline_coefs(tx, spline_id, ccx)
                
                # Correct stuff: calculate offset (max 2)
                cx1, cx2 = 0, 4
                #
                if ix<1: cx1+=1-ix;
                if ix>Nx-3: cx2+=(Nx-3)-ix;
                
                # Correct coefficients, so that the sum is one
                val = 0.0
                for cx in range(cx1, cx2):  val += ccx[cx]
                val = 1.0/val
                for cx in range(cx1, cx2):  ccx[cx] *= val
                
                # Combine elements
                val = 0.0
                for cx in range(cx1, cx2):
                    val += data_[ix+cx-1] * ccx[cx]
                result_[i] = val
                
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 1:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            
            if ix >= 0 and ix < Nx-1:
                # Linear interpolation
                val =  data_[ix] * (1.0-tx)
                val += data_[ix+1] * tx
                result_[i] = val
            elif dx>=-0.5 and dx<=Nx-0.5:
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                # Linear interpolation (edges)
                val =  data_[ix] * (1.0-tx)
                val += data_[ix+1] * tx
                result_[i] = val
            else:
                # Out of range
                result_[i] = 0.0
    
    else:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location
            dx = samplesx_[i]; ix = floor(dx+0.5)
            
            if ix >= 0 and ix < Nx:
                # Nearest neighbour interpolation
                result_[i] = data_[ix]
            else:
                # Out of range
                result_[i] = 0.0


@numba.jit(nopython=True, nogil=True)
def warp2(data_, result_, samplesx_, samplesy_, order, spline_id):
    
    Ni = samplesx_.size
    Ny = data_.shape[0]
    Nx = data_.shape[1]
    
    ccx = np.empty((4, ), np.float64)
    ccy = np.empty((4, ), np.float64)
    
    if order == 3:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            
            if (    ix >= 1 and ix < Nx-2 and 
                    iy >= 1 and iy < Ny-2       ):
                # Cubic interpolation
                
                # Get coefficients.
                if spline_id <= 1.0:  # By far most common, make it fast!
                    cubicsplinecoef_cardinal(tx, ccx, spline_id)  # tension=spline_id
                    cubicsplinecoef_cardinal(ty, ccy, spline_id)
                elif spline_id == 99.0:
                    cubicsplinecoef_quadratic(tx, ccx)
                    cubicsplinecoef_quadratic(ty, ccy)
                else:  # Select spline type, slower, but ok, because these make less sense, or are slow anyway
                    set_cubic_spline_coefs(tx, spline_id, ccx)
                    set_cubic_spline_coefs(ty, spline_id, ccy)
                
                # Apply
                val = 0.0
                for cy in range(4):
                    for cx in range(4):
                        val += data_[iy+cy-1,ix+cx-1] * ccy[cy] * ccx[cx]
                result_[i] = val
            
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5     ):
                # Edge effects
                
                # Get coefficients. Slower, but only needed at edges.
                set_cubic_spline_coefs(tx, spline_id, ccx)
                set_cubic_spline_coefs(ty, spline_id, ccy)
                
                # Correct stuff: calculate offset (max 2)
                cx1, cx2 = 0, 4
                cy1, cy2 = 0, 4
                #
                if ix<1: cx1+=1-ix;
                if ix>Nx-3: cx2+=(Nx-3)-ix;
                #
                if iy<1: cy1+=1-iy;
                if iy>Ny-3: cy2+=(Ny-3)-iy;
                
                # Correct coefficients, so that the sum is one
                val = 0.0
                for cx in range(cx1, cx2):  val += ccx[cx]
                val = 1.0/val
                for cx in range(cx1, cx2):  ccx[cx] *= val
                #
                val = 0.0
                for cy in range(cy1, cy2):  val += ccy[cy]
                val = 1.0/val
                for cy in range(cy1, cy2):  ccy[cy] *= val
                
                # Combine elements
                # No need to pre-calculate indices: the compiler is well
                # capable of making these optimizations.
                val = 0.0
                for cy in range(cy1, cy2):
                    for cx in range(cx1, cx2):
                        val += data_[iy+cy-1,ix+cx-1] * ccy[cy] * ccx[cx]
                result_[i] = val
            
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 1:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            
            if (    ix >= 0 and ix < Nx-1 and
                    iy >= 0 and iy < Ny-1     ):
                # Linear interpolation
                val =  data_[iy,  ix  ] * (1.0-ty) * (1.0-tx)
                val += data_[iy,  ix+1] * (1.0-ty) *      tx
                val += data_[iy+1,ix  ] *      ty  * (1.0-tx)
                val += data_[iy+1,ix+1] *      ty  *      tx
                result_[i] = val
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5     ):
                # Edge effects
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                #
                if iy<0: ty+=iy; iy=0; 
                if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
                # Linear interpolation (edges)
                val =  data_[iy,  ix  ] * (1.0-ty) * (1.0-tx)
                val += data_[iy,  ix+1] * (1.0-ty) *      tx
                val += data_[iy+1,ix  ] *      ty  * (1.0-tx)
                val += data_[iy+1,ix+1] *      ty  *      tx
                result_[i] = val
            else:
                # Out of range
                result_[i] = 0.0
    
    else:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location
            dx = samplesx_[i]; ix = floor(dx+0.5)
            dy = samplesy_[i]; iy = floor(dy+0.5)
            
            if (    ix >= 0 and ix < Nx and
                    iy >= 0 and iy < Ny     ):
                # Nearest neighbour interpolation
                result_[i] = data_[iy,ix]
            else:
                # Out of range
                result_[i] = 0.0


@numba.jit(nopython=True, nogil=True)
def warp3(data_, result_, samplesx_, samplesy_, samplesz_, order, spline_id):
    
    Ni = samplesx_.size
    Nz = data_.shape[0]
    Ny = data_.shape[1]
    Nx = data_.shape[2]
    
    ccx = np.empty((4, ), np.float64)
    ccy = np.empty((4, ), np.float64)
    ccz = np.empty((4, ), np.float64)
    
    if order == 3:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            dz = samplesz_[i]; iz = floor(dz); tz = dz-iz
            
            if (    ix >= 1 and ix < Nx-2 and 
                    iy >= 1 and iy < Ny-2 and
                    iz >= 1 and iz < Nz-2       ):
                # Cubic interpolation
                
                # Get coefficients.
                if spline_id <= 1.0:  # By far most common, make it fast!
                    cubicsplinecoef_cardinal(tx, ccx, spline_id)  # tension=spline_id
                    cubicsplinecoef_cardinal(ty, ccy, spline_id)
                    cubicsplinecoef_cardinal(tz, ccz, spline_id)
                elif spline_id == 99.0:
                    cubicsplinecoef_quadratic(tx, ccx)
                    cubicsplinecoef_quadratic(ty, ccy)
                    cubicsplinecoef_quadratic(tz, ccz)
                else:  # Select spline type, slower, but ok, because these make less sense, or are slow anyway
                    set_cubic_spline_coefs(tx, spline_id, ccx)
                    set_cubic_spline_coefs(ty, spline_id, ccy)
                    set_cubic_spline_coefs(tz, spline_id, ccz)
                
                # Apply
                val = 0.0
                for cz in range(4):
                    for cy in range(4):
                        for cx in range(4):
                            val += data_[iz+cz-1,iy+cy-1,ix+cx-1] * (
                                            ccz[cz] * ccy[cy] * ccx[cx] )
                result_[i] = val
            
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5 and
                    dz>=-0.5 and dz<=Nz-0.5     ):
                # Edge effects
                
                # Get coefficients. Slower, but only needed at edges.
                set_cubic_spline_coefs(tx, spline_id, ccx)
                set_cubic_spline_coefs(ty, spline_id, ccy)
                set_cubic_spline_coefs(tz, spline_id, ccz)
                
                # Correct stuff: calculate offset (max 2)
                cx1, cx2 = 0, 4
                cy1, cy2 = 0, 4
                cz1, cz2 = 0, 4
                #
                if ix<1: cx1+=1-ix;
                if ix>Nx-3: cx2+=(Nx-3)-ix;
                #
                if iy<1: cy1+=1-iy;
                if iy>Ny-3: cy2+=(Ny-3)-iy;
                #
                if iz<1: cz1+=1-iz;
                if iz>Nz-3: cz2+=(Nz-3)-iz;
                
                # Correct coefficients, so that the sum is one
                val = 0.0
                for cx in range(cx1, cx2):  val += ccx[cx]
                val = 1.0/val
                for cx in range(cx1, cx2):  ccx[cx] *= val
                #
                val = 0.0
                for cy in range(cy1, cy2):  val += ccy[cy]
                val = 1.0/val
                for cy in range(cy1, cy2):  ccy[cy] *= val
                #
                val = 0.0
                for cz in range(cz1, cz2):  val += ccz[cz]
                val = 1.0/val
                for cz in range(cz1, cz2):  ccz[cz] *= val
                
                # Combine elements 
                # No need to pre-calculate indices: the C compiler is well
                # capable of making these optimizations.
                val = 0.0
                for cz in range(cz1, cz2):
                    for cy in range(cy1, cy2):
                        for cx in range(cx1, cx2):
                            val += data_[iz+cz-1,iy+cy-1,ix+cx-1] * ccz[cz] * ccy[cy] * ccx[cx]
                result_[i] = val
            
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 1:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            dz = samplesz_[i]; iz = floor(dz); tz = dz-iz
            
            if (    ix >= 0 and ix < Nx-1 and
                    iy >= 0 and iy < Ny-1 and
                    iz >= 0 and iz < Nz-1       ):
                # Linear interpolation
                val =  data_[iz  ,iy,  ix  ] * (1.0-tz) * (1.0-ty) * (1.0-tx)
                val += data_[iz  ,iy,  ix+1] * (1.0-tz) * (1.0-ty) *      tx
                val += data_[iz  ,iy+1,ix  ] * (1.0-tz) *      ty  * (1.0-tx)
                val += data_[iz  ,iy+1,ix+1] * (1.0-tz) *      ty  *      tx
                #
                val += data_[iz+1,iy,  ix  ] *      tz  * (1.0-ty) * (1.0-tx)
                val += data_[iz+1,iy,  ix+1] *      tz  * (1.0-ty) *      tx
                val += data_[iz+1,iy+1,ix  ] *      tz  *      ty  * (1.0-tx)
                val += data_[iz+1,iy+1,ix+1] *      tz  *      ty  *      tx
                result_[i] = val
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5 and
                    dz>=-0.5 and dz<=Nz-0.5    ):
                # Edge effects
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                #
                if iy<0: ty+=iy; iy=0; 
                if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
                #
                if iz<0: tz+=iz; iz=0; 
                if iz>Nz-2: tz+=iz-(Nz-2); iz=Nz-2; 
                # Linear interpolation (edges)
                val =  data_[iz  ,iy,  ix  ] * (1.0-tz) * (1.0-ty) * (1.0-tx)
                val += data_[iz  ,iy,  ix+1] * (1.0-tz) * (1.0-ty) *      tx
                val += data_[iz  ,iy+1,ix  ] * (1.0-tz) *      ty  * (1.0-tx)
                val += data_[iz  ,iy+1,ix+1] * (1.0-tz) *      ty  *      tx
                #
                val += data_[iz+1,iy,  ix  ] *      tz  * (1.0-ty) * (1.0-tx)
                val += data_[iz+1,iy,  ix+1] *      tz  * (1.0-ty) *      tx
                val += data_[iz+1,iy+1,ix  ] *      tz  *      ty  * (1.0-tx)
                val += data_[iz+1,iy+1,ix+1] *      tz  *      ty  *      tx
                result_[i] = val
            else:
                # Out of range
                result_[i] = 0.0
    
    else:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location
            dx = samplesx_[i]; ix = floor(dx+0.5)
            dy = samplesy_[i]; iy = floor(dy+0.5)
            dz = samplesz_[i]; iz = floor(dz+0.5)
            
            if (    ix >= 0 and ix < Nx and
                    iy >= 0 and iy < Ny and
                    iz >= 0 and iz < Nz     ):
                # Nearest neighbour interpolation
                result_[i] = data_[iz,iy,ix]
            else:
                # Out of range
                result_[i] = 0.0


# todo: Cuda!
# Seemed just a bit faster when I tried it on 2D data and using Luts, but could
# be more performant with direct coefficients and on 3D!

@cuda.jit
def warp2_cuda(data_, result_, samplesx_, samplesy_, order, lutn, lut):
    
    coefsx = cuda.local.array((4, ), numba.float64)
    coefsy = cuda.local.array((4, ), numba.float64)
    Ni = samplesx_.size
    Ny = data_.shape[0]
    Nx = data_.shape[1]
    
    
    if order == 3:
        
        # Iterate over all samples
        #for i in range(0, Ni):
        i = cuda.grid(1)
        if i < Ni:
            
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            
            if (    ix >= 1 and ix < Nx-2 and 
                    iy >= 1 and iy < Ny-2       ):
                # Cubic interpolation
                # ccx = get_coef(lut, lutn, tx)
                # ccy = get_coef(lut, lutn, ty)
                i1 = floor( tx * lutn + 0.5) * 4; ccx = lut[i1:i1 + 4]
                i1 = floor( ty * lutn + 0.5) * 4; ccy = lut[i1:i1 + 4]
                val = 0.0
                for cy in range(4):
                    for cx in range(4):
                        val += data_[iy+cy-1,ix+cx-1] * ccy[cy] * ccx[cx]
                result_[i] = val
            
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5     ):
                # Edge effects
                
                # Get coefficients, make copy, because we need to edit tem
                i1 = floor( tx * lutn + 0.5) * 4; ccx = lut[i1:i1 + 4]
                i1 = floor( ty * lutn + 0.5) * 4; ccy = lut[i1:i1 + 4]
                for i in range(4):
                    coefsx[i] = ccx[i]
                    coefsy[i] = ccy[i]
                
                # Correct stuff: calculate offset (max 2)
                cx1, cx2 = 0, 4
                cy1, cy2 = 0, 4
                #
                if ix<1: cx1+=1-ix;
                if ix>Nx-3: cx2+=(Nx-3)-ix;
                #
                if iy<1: cy1+=1-iy;
                if iy>Ny-3: cy2+=(Ny-3)-iy;
                
                # Correct coefficients, so that the sum is one
                val = 0.0
                for cx in range(cx1, cx2):  val += coefsx[cx]
                val = 1.0/val
                for cx in range(cx1, cx2):  coefsx[cx] *= val
                #
                val = 0.0
                for cy in range(cy1, cy2):  val += coefsy[cy]
                val = 1.0/val
                for cy in range(cy1, cy2):  coefsy[cy] *= val
                
                # Combine elements
                # No need to pre-calculate indices: the C compiler is well
                # capable of making these optimizations.
                val = 0.0
                for cy in range(cy1, cy2):
                    for cx in range(cx1, cx2):
                        val += data_[iy+cy-1,ix+cx-1] * coefsy[cy] * coefsx[cx]
                result_[i] = val
            
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 1:
        # Iterate over all samples
        #for i in range(0, Ni):
        i = cuda.grid(1)
        if i < Ni:
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            
            if (    ix >= 0 and ix < Nx-1 and
                    iy >= 0 and iy < Ny-1     ):
                # Linear interpolation
                val =  data_[iy,  ix  ] * (1.0-ty) * (1.0-tx)
                val += data_[iy,  ix+1] * (1.0-ty) *      tx
                val += data_[iy+1,ix  ] *      ty  * (1.0-tx)
                val += data_[iy+1,ix+1] *      ty  *      tx
                result_[i] = val
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5     ):
                # Edge effects
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                #
                if iy<0: ty+=iy; iy=0; 
                if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
                # Linear interpolation (edges)
                val =  data_[iy,  ix  ] * (1.0-ty) * (1.0-tx)
                val += data_[iy,  ix+1] * (1.0-ty) *      tx
                val += data_[iy+1,ix  ] *      ty  * (1.0-tx)
                val += data_[iy+1,ix+1] *      ty  *      tx
                result_[i] = val
            else:
                # Out of range
                result_[i] = 0.0

