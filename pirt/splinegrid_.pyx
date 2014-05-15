""" Cython module splineGrid_

Provides functionality for (cubic B-spline) grids. 

Copyright 2010 (C) Almar Klein, University of Twente.

A Note on pointsets:
In this module, all pointsets are 2D numpy arrays shaped (N, ndim), 
which express world coordinates. The coordinates they represent 
should be expressed using doubles wx,wy,wz, in contrast to pixel
coordinates, which are expressed using integers x, y, z. 

"""

from pirt.interp.interpolation_ cimport (
    CoefLut, AccurateCoef, FLOAT32_T, FLOAT64_T, SAMPLE_T)

# Cython specific imports
import numpy as np
cimport numpy as np
import cython 

cdef double pi = 3.1415926535897931
cdef extern from "math.h":
    double sin(double val)

# Type defs, we support float32 and float64
ctypedef np.float64_t GRID_T
FLOAT32 = np.float32
FLOAT64 = np.float64
SAMPLE = np.float32
GRID = np.float64

# Floor operator (deal with negative numbers)
cdef inline int floor(double a): return <int>a if a>=0.0 else (<int>a)-1


## Functions to obtain the field that the grid represents


def get_field(grid, spline_type='B'):
    """ get_field(grid)
    Sample the grid at all the pixels of the underlying field.
    """
    
    # Decide what function to call
    if grid.ndim == 1:
        return _get_field1(grid, spline_type)
    elif grid.ndim == 2: 
        return _get_field2(grid, spline_type)
    elif grid.ndim == 3: 
        return _get_field3(grid, spline_type)
    else:
        tmp = 'Grid interpolation not suported for this dimension.'
        raise RuntimeError(tmp)


def get_field_sparse(grid, pp, spline_type='B'):
    """ get_field_sparse(grid, pp)
    
    Sparsely sample the grid at a specified set of points (which are in
    world coordinates).
    
    Also see get_field_at(). 
    
    """
    
    # Test dimensions
    if grid.ndim != pp.shape[1]:
        raise ValueError('Dimension of grid and pointset do not match.')
    
    # Create samples
    samples = []
    for i in range(pp.shape[1]):
        samples.append(pp[:,i])
    
    # Decide what function to call
    if grid.ndim == 1:
        return _get_field_at1(grid, *samples, inPixels=False, spline_type=spline_type)
    elif grid.ndim == 2: 
        return _get_field_at2(grid, *samples, inPixels=False, spline_type=spline_type)
    elif grid.ndim == 3: 
        return _get_field_at3(grid, *samples, inPixels=False, spline_type=spline_type)
    else:
        tmp = 'Grid interpolation not suported for this dimension.'
        raise RuntimeError(tmp)


def get_field_at(grid, samples, spline_type='B'):
    """ get_field_at(grid, samples)
    
    Sample the grid at specified sample locations (in pixels), similar to 
    pirt.interp.interp().
    
    Also see get_field_sparse().
    
    """
    
    # Test dimensions
    if not isinstance(samples, (tuple, list)):
        raise ValueError('Samples must be list or tuple.')
    if len(samples) != grid.ndim:
        raise ValueError('Samples must contain one element per dimension.')
    sample0 = samples[0]
    for sample in samples:
        if sample0.shape != sample.shape:
            raise ValueError('Elements in samples must all have the same shape.')
    
    # Decide what function to call
    if grid.ndim == 1:
        return _get_field_at1(grid, *samples, inPixels=True, spline_type=spline_type)
    elif grid.ndim == 2: 
        return _get_field_at2(grid, *samples, inPixels=True, spline_type=spline_type)
    elif grid.ndim == 3: 
        return _get_field_at3(grid, *samples, inPixels=True, spline_type=spline_type)
    else:
        tmp = 'Grid interpolation not suported for this dimension.'
        raise RuntimeError(tmp)


## Workhorse functions to get the field. The result is always 32 bit

@cython.boundscheck(False)
@cython.wraparound(False)
def _get_field1(grid, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp
    
    # Create and init cubic interpolator    
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    
    # Get shape and test dims
    shape = grid.field_shape
    if grid.ndim != 1:
        raise ValueError('This function can only sample 1D grids.')
    
    # Init resulting array, make it float32, since for deformation-grids
    # the result is a sampler to be used in interp().
    cdef np.ndarray[SAMPLE_T, ndim=1] result = np.zeros(shape, dtype=SAMPLE)
    
    # Grid stuff
    cdef double grid_xSpacing = grid.grid_sampling_in_pixels[0]
    cdef np.ndarray[GRID_T, ndim=1] knots = grid.knots
    
    
    # For each pixel ...    
    for x in range(shape[0]):
        
        # Note that we cannot put this bit inside a cdef function,
        # because we cannot cast the knots array as an argument...
        
        # Calculate what is the leftmost (reference) knot on the grid,
        # and the ratio between closest and second closest knot.
        # Note the +1 to correct for padding.        
        tmp = (<double>x / grid_xSpacing ) + 1
        gx = <int>tmp
        tx	= tmp - <double>gx
        
        # Get coefficients
        ccx = coeffx.get_coef(tx)
        
        # Init value
        val = 0.0
        
        # For each knot ...	        
        ii = gx - 1  # x-location of first knot
        for i in range(4):
            # Calculate interpolated value.
            val += ccx[i] * knots[ii]
            ii+=1
        
        # Store value in result array
        result[x] = val
    
    # Done
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_field2(grid, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    cdef AccurateCoef coeffy = AccurateCoef(lut)
    
    # Get shape and test dims
    shape = grid.field_shape
    if grid.ndim != 2:
        raise ValueError('This function can only sample 2D grids.')
    
    # Init resulting array, make it float32, since for deformation-grids
    # the result is a sampler to be used in interp().
    cdef np.ndarray[SAMPLE_T, ndim=2] result = np.zeros(shape, dtype=SAMPLE)
    
    # Grid stuff
    cdef double grid_ySpacing = grid.grid_sampling_in_pixels[0]
    cdef double grid_xSpacing = grid.grid_sampling_in_pixels[1]
    cdef np.ndarray[GRID_T, ndim=2] knots = grid.knots
    
    
    # For each pixel ...
    for y in range(shape[0]):
        for x in range(shape[1]):
            
            # Note that we cannot put this bit inside a cdef function,
            # because we cannot cast the knots array as an argument...
            
            # Calculate what is the leftmost (reference) knot on the grid,
            # and the ratio between closest and second closest knot.
            # Note the +1 to correct for padding.
            tmp = (<double>y / grid_ySpacing ) + 1
            gy = <int>tmp
            ty	= tmp - <double>gy
            #
            tmp = (<double>x / grid_xSpacing ) + 1
            gx = <int>tmp
            tx	= tmp - <double>gx
            
            # Get coefficients
            ccy = coeffy.get_coef(ty)
            ccx = coeffx.get_coef(tx)
            
            # Init value
            val = 0.0
            
            # For each knot ...	
            jj = gy - 1  # y-location of first knot
            for j in range(4):
                ii = gx - 1  # x-location of first knot
                for i in range(4):
                    # Calculate interpolated value.
                    val += ccy[j] * ccx[i] * knots[jj,ii]
                    ii+=1
                jj+=1
            
            # Store value in result array
            result[y,x] = val
    
    # Done
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_field3(grid, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    cdef AccurateCoef coeffy = AccurateCoef(lut)
    cdef AccurateCoef coeffz = AccurateCoef(lut)
    
    # Get shape and test dims
    shape = grid.field_shape
    if grid.ndim != 3:
        raise ValueError('This function can only sample 3D grids.')
    
    # Init resulting array, make it float32, since for deformation-grids
    # the result is a sampler to be used in interp().
    cdef np.ndarray[SAMPLE_T, ndim=3] result = np.zeros(shape, dtype=SAMPLE)
    
    # Grid stuff
    cdef double grid_zSpacing = grid.grid_sampling_in_pixels[0]
    cdef double grid_ySpacing = grid.grid_sampling_in_pixels[1]
    cdef double grid_xSpacing = grid.grid_sampling_in_pixels[2]
    cdef np.ndarray[GRID_T, ndim=3] knots = grid.knots
    
    
    # For each pixel ...
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                
                # Note that we cannot put this bit inside a cdef function,
                # because we cannot cast the knots array as an argument...
                
                # Calculate what is the leftmost (reference) knot on the grid,
                # and the ratio between closest and second closest knot.
                # Note the +1 to correct for padding.
                tmp = (<double>z / grid_zSpacing ) + 1
                gz = <int>tmp
                tz	= tmp - <double>gz
                #
                tmp = (<double>y / grid_ySpacing ) + 1
                gy = <int>tmp
                ty	= tmp - <double>gy
                #
                tmp = (<double>x / grid_xSpacing ) + 1
                gx = <int>tmp
                tx	= tmp - <double>gx
                
                # Get coefficients
                ccz = coeffz.get_coef(tz)
                ccy = coeffy.get_coef(ty)
                ccx = coeffx.get_coef(tx)
                
                # Init value
                val = 0.0
                
                # For each knot ...	
                kk = gz -1  # z-location of first knot
                for k in range(4):
                    jj = gy - 1  # y-location of first knot
                    for j in range(4):
                        ii = gx - 1  # x-location of first knot
                        for i in range(4):
                            # Calculate interpolated value.
                            val += ccz[k] * ccy[j] * ccx[i] * knots[kk,jj,ii]
                            ii+=1
                        jj+=1
                    kk+=1
                
                # Store value in result array
                result[z,y,x] = val
    
    # Done
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_field_at1(grid, samplesx, inPixels=True, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp
    cdef double grid_xSpacing, grid_ySpacing, grid_zSpacing
    
    # Sample locations
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesx_ = samplesx.ravel()
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    
    # Grid stuff
    if inPixels:
        grid_xSpacing = grid.grid_sampling_in_pixels[0]
    else:
        grid_xSpacing = grid_ySpacing = grid_zSpacing = grid.grid_sampling
    #
    cdef double wGridSampling = grid.grid_sampling
    cdef np.ndarray[GRID_T, ndim=1] knots = grid.knots
    cdef int gridShapex = knots.shape[0]
    
    # Init resulting array, make it float32, since for deformation-grids
    # the result is a sampler to be used in interp().
    result = np.zeros_like(samplesx)
    cdef np.ndarray[SAMPLE_T, ndim=1] result_ = result.ravel()
    
    
    # For each point in the set
    for p in range(samplesx.size):
        
        # Calculate wx
        wx = samplesx_[p]
        
        # Note that we cannot put this bit inside a cdef function,
        # because we cannot cast the knots array as an argument...
        
        # Calculate what is the leftmost (reference) knot on the grid,
        # and the ratio between closest and second closest knot.
        # Note the +1 to correct for padding.
        tmp = (wx / grid_xSpacing ) + 1
        gx = <int>tmp
        tx	= tmp - <double>gx
        
        # Check if within bounds of interpolatable domain
        if (    (gx<1 or gx >= gridShapex-2) ):
            result_[p] = 0.0
            continue
        
        # Get coefficients
        ccx = coeffx.get_coef(tx)
        
        # Init value
        val = 0.0
        
        # For each knot ...	
        ii = gx - 1  # x-location of first knot
        for i in range(4):
            # Calculate interpolated value.
            val += ccx[i] * knots[ii]
            ii+=1
        
        # Store
        result_[p] = val
    
    # Done
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_field_at2(grid, samplesx, samplesy, inPixels=True, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp
    cdef double grid_xSpacing, grid_ySpacing, grid_zSpacing
    
    # Sample locations
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesx_ = samplesx.ravel()
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesy_ = samplesy.ravel()
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    cdef AccurateCoef coeffy = AccurateCoef(lut)
    
    # Grid stuff
    if inPixels:
        grid_ySpacing = grid.grid_sampling_in_pixels[0]
        grid_xSpacing = grid.grid_sampling_in_pixels[1]
    else:
        grid_ySpacing = grid_xSpacing = grid.grid_sampling
    #
    cdef np.ndarray[GRID_T, ndim=2] knots = grid.knots
    cdef int gridShapey = knots.shape[0]
    cdef int gridShapex = knots.shape[1]
    
    # Init resulting array, make it float32, since for deformation-grids
    # the result is a sampler to be used in interp().
    result = np.zeros_like(samplesx)
    cdef np.ndarray[SAMPLE_T, ndim=1] result_ = result.ravel()
    
    
    # For each point in the set
    for p in range(samplesx.size):
        
        # Calculate wx and wy
        wx = samplesx_[p]
        wy = samplesy_[p]
        
        # Note that we cannot put this bit inside a cdef function,
        # because we cannot cast the knots array as an argument...
        
        # Calculate what is the leftmost (reference) knot on the grid,
        # and the ratio between closest and second closest knot.
        # Note the +1 to correct for padding.
        tmp = (wy / grid_ySpacing ) + 1
        gy = <int>tmp
        ty	= tmp - <double>gy
        #
        tmp = (wx / grid_xSpacing ) + 1
        gx = <int>tmp
        tx	= tmp - <double>gx
        
        # Check if within bounds of interpolatable domain
        if (    (gy<1 or gy >= gridShapey-2) or
                (gx<1 or gx >= gridShapex-2) ):
            result_[p] = 0.0
            continue
        
        # Get coefficients
        ccy = coeffy.get_coef(ty)
        ccx = coeffx.get_coef(tx)
        
        # Init value
        val = 0.0
        
        # For each knot ...	
        jj = gy - 1  # y-location of first knot
        for j in range(4):
            ii = gx - 1  # x-location of first knot
            for i in range(4):
                # Calculate interpolated value.
                val += ccy[j] * ccx[i] * knots[jj,ii]
                ii+=1
            jj+=1
        
        # Store
        result_[p] = val
    
    # Done
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_field_at3(grid, samplesx, samplesy, samplesz, inPixels=True, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp
    cdef double grid_xSpacing, grid_ySpacing, grid_zSpacing
    
    # Sample locations
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesx_ = samplesx.ravel()
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesy_ = samplesy.ravel()
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesz_ = samplesz.ravel()
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    cdef AccurateCoef coeffy = AccurateCoef(lut)
    cdef AccurateCoef coeffz = AccurateCoef(lut)
    
    # Grid stuff
    if inPixels:
        grid_zSpacing = grid.grid_sampling_in_pixels[0]
        grid_ySpacing = grid.grid_sampling_in_pixels[1]
        grid_xSpacing = grid.grid_sampling_in_pixels[2]
    else:
        grid_xSpacing = grid_ySpacing = grid_zSpacing = grid.grid_sampling
    #
    cdef np.ndarray[GRID_T, ndim=3] knots = grid.knots
    cdef int gridShapez = knots.shape[0]
    cdef int gridShapey = knots.shape[1]
    cdef int gridShapex = knots.shape[2]
    
    # Init resulting array, make it float32, since for deformation-grids
    # the result is a sampler to be used in interp().
    result = np.zeros_like(samplesx)
    cdef np.ndarray[SAMPLE_T, ndim=1] result_ = result.ravel()
    
    
    # For each point in the set
    for p in range(samplesx.size):
        
        # Calculate wx and wy
        wx = samplesx_[p]
        wy = samplesy_[p]
        wz = samplesz_[p]
        
        # Note that we cannot put this bit inside a cdef function,
        # because we cannot cast the knots array as an argument...
        
        # Calculate what is the leftmost (reference) knot on the grid,
        # and the ratio between closest and second closest knot.
        # Note the +1 to correct for padding.
        tmp = (wz / grid_zSpacing ) + 1
        gz = <int>tmp
        tz	= tmp - <double>gz
        #
        tmp = (wy / grid_ySpacing ) + 1
        gy = <int>tmp
        ty	= tmp - <double>gy
        #
        tmp = (wx / grid_xSpacing ) + 1
        gx = <int>tmp
        tx	= tmp - <double>gx
        
         # Check if within bounds of interpolatable domain
        if (    (gx<1 or gx >= gridShapex-2) or
                (gy<1 or gy >= gridShapey-2) or
                (gz<1 or gz >= gridShapez-2)):
            result_[p] = 0.0
            continue
        
        # Get coefficients
        ccz = coeffz.get_coef(tz)
        ccy = coeffy.get_coef(ty)
        ccx = coeffx.get_coef(tx)
        
        # Init value
        val = 0.0
        
        # For each knot ...	
        kk = gz - 1  # z-location of first knot
        for k in range(4):
            jj = gy - 1  # y-location of first knot
            for j in range(4):
                ii = gx - 1  # x-location of first knot
                for i in range(4):
                    # Calculate interpolated value.
                    val += ccz[k] * ccy[j] * ccx[i] * knots[kk,jj,ii]
                    ii+=1
                jj+=1
            kk+=1
        
        # Store
        result_[p] = val
    
    # Done
    return result


## Functions to set the grid using a field

@cython.boundscheck(False)
@cython.wraparound(False)
def _set_field_using_num_and_dnum(grid, num, dnum):
    
    # Cast flat arrays
    cdef np.ndarray[GRID_T, ndim=1] knots = grid.knots.ravel()
    cdef np.ndarray[GRID_T, ndim=1] num_ = num.ravel()
    cdef np.ndarray[GRID_T, ndim=1] dnum_ = dnum.ravel()
    
    # Loop
    cdef int i
    cdef double n
    for i in range(knots.size):
        n = dnum_[i]
        if n > 0.0:
            knots[i] = num_[i] / n
        else:
            n = 0.0


def set_field(grid, field, weights, spline_type='B'):
    """ set_field(grid, pp)
    Set the grid using the specified field (and optional weights).
    """
    # Test dimensions
    if grid.field_shape != field.shape:
        raise ValueError('Dimension of grid-field and field do not match.')
    
    # Test dtype
    if field.dtype != weights.dtype:
        raise ValueError('Field and weights must be of the same type.')
    
    # Apply proper function
    if field.dtype.name == 'float32':
        if grid.ndim == 1:
            num, dnum = _set_field1_32(grid, field, weights, spline_type)
        elif grid.ndim == 2:
            num, dnum = _set_field2_32(grid, field, weights, spline_type)
        elif grid.ndim == 3:
            num, dnum = _set_field3_32(grid, field, weights, spline_type)
        else:
            tmp = 'This method does not support grids of that dimension.'
            raise RuntimeError(tmp)
    
    elif field.dtype.name == 'float64':
        if grid.ndim == 1:
            num, dnum = _set_field1_64(grid, field, weights, spline_type)
        elif grid.ndim == 2:
            num, dnum = _set_field2_64(grid, field, weights, spline_type)
        elif grid.ndim == 3:
            num, dnum = _set_field3_64(grid, field, weights, spline_type)
        else:
            tmp = 'This method does not support grids of that dimension.'
            raise RuntimeError(tmp)
    
    else:
        raise ValueError('This function only supports 32 bit and 64 bit floats.')
    
    # Apply
    _set_field_using_num_and_dnum(grid, num, dnum)


def set_field_sparse(grid, pp, values, spline_type='B'):
    """ set_field_sparse(grid, pp, values, spline_type='B')
    
    Set the grid by providing the field values at a set of points (wich
    are in world coordinates).  
    
    """
    
    # Test dimensions
    if grid.ndim != pp.shape[1]:
        raise ValueError('Dimension of grid and pointset do not match.')
    
    
    # Apply proper function
    if values.dtype.name == 'float32':
        # Apply proper function
        if grid.ndim == 1:
            num, dnum = _set_field_sparse1_32(grid, pp, values, spline_type)
        elif grid.ndim == 2:
            num, dnum = _set_field_sparse2_32(grid, pp, values, spline_type)
        elif grid.ndim == 3:
            num, dnum = _set_field_sparse3_32(grid, pp, values, spline_type)
        else:
            tmp = 'This method does not support grids of that dimension.'
            raise RuntimeError(tmp)
    
    elif values.dtype.name == 'float64':
        # Apply proper function
        if grid.ndim == 1:
            num, dnum = _set_field_sparse1_64(grid, pp, values, spline_type)
        elif grid.ndim == 2:
            num, dnum = _set_field_sparse2_64(grid, pp, values, spline_type)
        elif grid.ndim == 3:
            num, dnum = _set_field_sparse3_64(grid, pp, values, spline_type)
        else:
            tmp = 'This method does not support grids of that dimension.'
            raise RuntimeError(tmp)
    
    else:
        raise ValueError('This function only supports 32 bit and 64 bit floats.')
    
    # Apply
    _set_field_using_num_and_dnum(grid, num, dnum)


## 32 bit field set functions


@cython.boundscheck(False)
@cython.wraparound(False)
def _set_field1_32(grid, field, weights, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp, weight, omega, omega2, omsum
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    
    # Grid stuff    
    cdef double grid_xSpacing = grid.grid_sampling_in_pixels[0]
    
    # Cast input arrrays
    cdef np.ndarray[FLOAT32_T, ndim=1] field_ = field
    cdef np.ndarray[FLOAT32_T, ndim=1] weights_ = weights
    
    # Create temporary arrays the same size as the grid
    cdef np.ndarray[GRID_T, ndim=1] num_ = np.zeros_like(grid.knots)
    cdef np.ndarray[GRID_T, ndim=1] dnum_ = np.zeros_like(grid.knots)
    
    
    # For each pixel ...        
    for x in range(field_.shape[0]):
        
        # Get val and alpha
        val = field_[x]
        weight = weights_[x]
        
        # Evaluate this one?
        if weight <= 0.0:
            continue
        
        # Calculate what is the leftmost (reference) knot on the grid,
        # and the ratio between closest and second closest knot.
        # Note the +1 to correct for padding.
        tmp = (<double>x / grid_xSpacing ) + 1
        gx = <int>tmp
        tx	= tmp - <double>gx
        
        # Get coefficients
        ccx = coeffx.get_coef(tx)
        
        # Pre-normalize value
        omsum = 0.0        
        for i in range(4):
            omsum += ccx[i] * ccx[i]
        val_n = val / omsum
        
        # For each knot that this point influences
        # Following Lee et al. we update a numerator and a denumerator for
        # each knot.
        for i in range(4):
            ii = i+gx-1
            #
            omega = ccx[i]
            omega2 = weight * omega * omega
            num_[ii] += omega2 * ( val_n*omega)
            dnum_[ii] += omega2
    
    # Done
    return num_, dnum_


@cython.boundscheck(False)
@cython.wraparound(False)
def _set_field2_32(grid, field, weights, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, val_n, tmp, weight, omega, omega2, omsum
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    cdef AccurateCoef coeffy = AccurateCoef(lut)
    
    # Grid stuff
    cdef double grid_ySpacing = grid.grid_sampling_in_pixels[0]
    cdef double grid_xSpacing = grid.grid_sampling_in_pixels[1]
    
    # Cast input arrrays
    cdef np.ndarray[FLOAT32_T, ndim=2] field_ = field
    cdef np.ndarray[FLOAT32_T, ndim=2] weights_ = weights
    
    # Create temporary arrays the same size as the grid
    cdef np.ndarray[GRID_T, ndim=2] num_ = np.zeros_like(grid.knots)
    cdef np.ndarray[GRID_T, ndim=2] dnum_ = np.zeros_like(grid.knots)
    
    
    # For each pixel ...    
    for y in range(field_.shape[0]):
        for x in range(field_.shape[1]):
            
            # Get val and alpha
            val = field_[y, x]
            weight = weights_[y, x]
            
            # Evaluate this one?
            if weight <= 0.0:
                continue
            
            # Calculate what is the leftmost (reference) knot on the grid,
            # and the ratio between closest and second closest knot.
            # Note the +1 to correct for padding.
            tmp = (<double>y / grid_ySpacing ) + 1
            gy = <int>tmp
            ty	= tmp - <double>gy
            #
            tmp = (<double>x / grid_xSpacing ) + 1
            gx = <int>tmp
            tx	= tmp - <double>gx
            
            # Get coefficients
            ccy = coeffy.get_coef(ty)
            ccx = coeffx.get_coef(tx)
            
            # Pre-normalize value
            omsum = 0.0
            for j in range(4):
                for i in range(4):
                    omsum += ccy[j] * ccx[i] * ccy[j] * ccx[i]
            val_n = val / omsum
            
            # For each knot that this point influences
            # Following Lee et al. we update a numerator and a denumerator for
            # each knot.
            for j in range(4):
                jj = j+gy-1
                for i in range(4):
                    ii = i+gx-1
                    #
                    omega = ccy[j] * ccx[i]
                    omega2 = weight * omega * omega
                    num_[jj,ii] += omega2 * ( val_n*omega)
                    dnum_[jj,ii] += omega2
    
    # Done
    return num_, dnum_


@cython.boundscheck(False)
@cython.wraparound(False)
def _set_field3_32(grid, field, weights, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp, weight, omega, omega2, omsum
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    cdef AccurateCoef coeffy = AccurateCoef(lut)
    cdef AccurateCoef coeffz = AccurateCoef(lut)
    
    # Grid stuff
    cdef double grid_zSpacing = grid.grid_sampling_in_pixels[0]
    cdef double grid_ySpacing = grid.grid_sampling_in_pixels[1]
    cdef double grid_xSpacing = grid.grid_sampling_in_pixels[2]
    
    # Cast input arrrays
    cdef np.ndarray[FLOAT32_T, ndim=3] field_ = field
    cdef np.ndarray[FLOAT32_T, ndim=3] weights_ = weights
    
    # Create temporary arrays the same size as the grid
    cdef np.ndarray[GRID_T, ndim=3] num_ = np.zeros_like(grid.knots)
    cdef np.ndarray[GRID_T, ndim=3] dnum_ = np.zeros_like(grid.knots)
    
    
    # For each pixel ...    
    for z in range(field_.shape[0]):
        for y in range(field_.shape[1]):
            for x in range(field_.shape[2]):
                
                # Get val and alpha
                val = field_[z, y, x]
                weight = weights_[z, y, x]
                
                # Evaluate this one?
                if weight <= 0.0:
                    continue
                
                # Calculate what is the leftmost (reference) knot on the grid,
                # and the ratio between closest and second closest knot.
                # Note the +1 to correct for padding.
                tmp = (<double>z / grid_zSpacing ) + 1
                gz = <int>tmp
                tz	= tmp - <double>gz
                #
                tmp = (<double>y / grid_ySpacing ) + 1
                gy = <int>tmp
                ty	= tmp - <double>gy
                #
                tmp = (<double>x / grid_xSpacing ) + 1
                gx = <int>tmp
                tx	= tmp - <double>gx
                
                # Get coefficients
                ccz = coeffz.get_coef(tz)
                ccy = coeffy.get_coef(ty)
                ccx = coeffx.get_coef(tx)
                
                # Pre-normalize value
                omsum = 0.0
                for k in range(4):
                    for j in range(4):
                        for i in range(4):
                            omsum += ccz[k] * ccy[j] * ccx[i] * ccz[k] * ccy[j] * ccx[i]
                val_n = val / omsum
                
                # For each knot that this point influences
                # Following Lee et al. we update a numerator and a denumerator for
                # each knot.
                for k in range(4):
                    kk = k+gz-1
                    for j in range(4):
                        jj = j+gy-1
                        for i in range(4):
                            ii = i+gx-1
                            #
                            omega = ccz[k] * ccy[j] * ccx[i]
                            omega2 = weight * omega * omega
                            num_[kk,jj,ii] += omega2 * ( val_n*omega)
                            dnum_[kk,jj,ii] += omega2
    
    # Done
    return num_, dnum_


@cython.boundscheck(False)
@cython.wraparound(False)
def _set_field_sparse1_32(grid, pp, values, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp, weight, omega, omega2, omsum
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    
    # Grid stuff
    cdef double wGridSampling = grid.grid_sampling
    
    # Create num, dnum 
    cdef np.ndarray[GRID_T, ndim=1] num_ = np.zeros_like(grid.knots)
    cdef np.ndarray[GRID_T, ndim=1] dnum_ = np.zeros_like(grid.knots)
    
    # Cast pointset
    cdef np.ndarray[FLOAT32_T, ndim=2] pp_ = pp
    
    
    # For each point ...
    for p in range(pp_.shape[0]):
        
        # Get wx
        wx = pp_[p,0]
        
        # Calculate which is the closest point on the lattice to the top-left
        # corner and find ratio's of influence between lattice point.
        tmp = (wx / wGridSampling ) + 1
        gx = <int>tmp
        tx	= tmp - <double>gx
        
        # Get coefficients
        ccx = coeffx.get_coef(tx)
        
        # Precalculate omsum
        omsum = 0.0        
        for i in range(4):
            omsum += ccx[i] * ccx[i]
        
        # Get val
        val = values[p]
        
        # For each knot that this point influences
        # Following Lee et al. we update a numerator and a denumerator for
        # each knot.
        ii = gx - 1  # x-location of first knot
        for i in range(4):
            #
            omega = ccx[i]
            omega2 = omega*omega
            num_[ii] += omega2 * ( val*omega/omsum )
            dnum_[ii] += omega2
            #
            ii+=1
    
    # Done
    return num_, dnum_


@cython.boundscheck(False)
@cython.wraparound(False)
def _set_field_sparse2_32(grid, pp, values, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp, weight, omega, omega2, omsum
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    cdef AccurateCoef coeffy = AccurateCoef(lut)
    
    # Grid stuff
    cdef double wGridSampling = grid.grid_sampling
    
    # Create num, dnum 
    cdef np.ndarray[GRID_T, ndim=2] num_ = np.zeros_like(grid.knots)
    cdef np.ndarray[GRID_T, ndim=2] dnum_ = np.zeros_like(grid.knots)
    
    # Cast pointset
    cdef np.ndarray[FLOAT32_T, ndim=2] pp_ = pp
    
    
    # For each point ...
    for p in range(pp_.shape[0]):
        
        # Get wx and wy
        wx = pp_[p,0]
        wy = pp_[p,1]
        
        # Calculate which is the closest point on the lattice to the top-left
        # corner and find ratio's of influence between lattice point.
        tmp = (wy / wGridSampling ) + 1
        gy = <int>tmp
        ty	= tmp - <double>gy
        #
        tmp = (wx / wGridSampling ) + 1
        gx = <int>tmp
        tx	= tmp - <double>gx
        
        # Get coefficients
        ccy = coeffy.get_coef(ty)
        ccx = coeffx.get_coef(tx)
        
        # Precalculate omsum (denominator of eq 4 in Lee 1996)
        omsum = 0.0
        for j in range(4):
            for i in range(4):
                omsum += ccy[j] * ccx[i] * ccy[j] * ccx[i]
        
        # Get val
        val = values[p]
        
        # For each knot that this point influences
        # Following Lee et al. we update a numerator and a denumerator for
        # each knot.
        jj = gy - 1  # y-location of first knot
        for j in range(4):
            ii = gx - 1  # x-location of first knot
            for i in range(4):
                #
                omega = ccy[j] * ccx[i]
                omega2 = omega*omega
                num_[jj,ii] += omega2 * ( val*omega/omsum )
                dnum_[jj,ii] += omega2
                #
                ii+=1
            jj+=1
    
    # Done
    return num_, dnum_


@cython.boundscheck(False)
@cython.wraparound(False)
def _set_field_sparse3_32(grid, pp, values, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp, weight, omega, omega2, omsum
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    cdef AccurateCoef coeffy = AccurateCoef(lut)
    cdef AccurateCoef coeffz = AccurateCoef(lut)
    
    # Grid stuff
    cdef double wGridSampling = grid.grid_sampling
    
    # Create num, dnum 
    cdef np.ndarray[GRID_T, ndim=3] num_ = np.zeros_like(grid.knots)
    cdef np.ndarray[GRID_T, ndim=3] dnum_ = np.zeros_like(grid.knots)
    
    # Cast pointset
    cdef np.ndarray[FLOAT32_T, ndim=2] pp_ = pp
    
    
    # For each point ...
    for p in range(pp_.shape[0]):
        
        # Get wx, wy and wz
        wx = pp_[p,0]
        wy = pp_[p,1]
        wz = pp_[p,2]
        
        # Calculate which is the closest point on the lattice to the top-left
        # corner and find ratio's of influence between lattice point.
        tmp = (wz / wGridSampling ) + 1
        gz = <int>tmp
        tz	= tmp - <double>gz
        #
        tmp = (wy / wGridSampling ) + 1
        gy = <int>tmp
        ty	= tmp - <double>gy
        #
        tmp = (wx / wGridSampling ) + 1
        gx = <int>tmp
        tx	= tmp - <double>gx
        
        # Get coefficients
        ccz = coeffz.get_coef(tz)
        ccy = coeffy.get_coef(ty)
        ccx = coeffx.get_coef(tx)
        
        # Precalculate omsum
        omsum = 0.0
        for k in range(4):
            for j in range(4):
                for i in range(4):
                    omsum += ccz[k] * ccy[j] * ccx[i] * ccz[k] * ccy[j] * ccx[i]
        
        # Get val
        val = values[p]
        
        # For each knot that this point influences
        # Following Lee et al. we update a numerator and a denumerator for
        # each knot.
        kk = gz -1  # z-location of first knot
        for k in range(4):
            jj = gy - 1  # y-location of first knot
            for j in range(4):
                ii = gx - 1  # x-location of first knot
                for i in range(4):
                    #
                    omega = ccy[j] * ccx[i]
                    omega2 = omega*omega
                    num_[jj,ii] += omega2 * ( val*omega/omsum )
                    dnum_[jj,ii] += omega2
                    #
                    ii+=1
                jj+=1
            kk+=1
    
    # Done
    return num_, dnum_



## 64 bit field set functions
# Copy the whole above cell and replace all occurances of '32' with '64'


@cython.boundscheck(False)
@cython.wraparound(False)
def _set_field1_64(grid, field, weights, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp, weight, omega, omega2, omsum
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    
    # Grid stuff    
    cdef double grid_xSpacing = grid.grid_sampling_in_pixels[0]
    
    # Cast input arrrays
    cdef np.ndarray[FLOAT64_T, ndim=1] field_ = field
    cdef np.ndarray[FLOAT64_T, ndim=1] weights_ = weights
    
    # Create temporary arrays the same size as the grid
    cdef np.ndarray[GRID_T, ndim=1] num_ = np.zeros_like(grid.knots)
    cdef np.ndarray[GRID_T, ndim=1] dnum_ = np.zeros_like(grid.knots)
    
    
    # For each pixel ...        
    for x in range(field_.shape[0]):
        
        # Get val and alpha
        val = field_[x]
        weight = weights_[x]
        
        # Evaluate this one?
        if weight <= 0.0:
            continue
        
        # Calculate what is the leftmost (reference) knot on the grid,
        # and the ratio between closest and second closest knot.
        # Note the +1 to correct for padding.
        tmp = (<double>x / grid_xSpacing ) + 1
        gx = <int>tmp
        tx	= tmp - <double>gx
        
        # Get coefficients
        ccx = coeffx.get_coef(tx)
        
        # Pre-normalize value
        omsum = 0.0        
        for i in range(4):
            omsum += ccx[i] * ccx[i]
        val_n = val / omsum
        
        # For each knot that this point influences
        # Following Lee et al. we update a numerator and a denumerator for
        # each knot.
        for i in range(4):
            ii = i+gx-1
            #
            omega = ccx[i]
            omega2 = weight * omega * omega
            num_[ii] += omega2 * ( val_n*omega)
            dnum_[ii] += omega2
    
    # Done
    return num_, dnum_


@cython.boundscheck(False)
@cython.wraparound(False)
def _set_field2_64(grid, field, weights, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, val_n, tmp, weight, omega, omega2, omsum
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    cdef AccurateCoef coeffy = AccurateCoef(lut)
    
    # Grid stuff
    cdef double grid_ySpacing = grid.grid_sampling_in_pixels[0]
    cdef double grid_xSpacing = grid.grid_sampling_in_pixels[1]
    
    # Cast input arrrays
    cdef np.ndarray[FLOAT64_T, ndim=2] field_ = field
    cdef np.ndarray[FLOAT64_T, ndim=2] weights_ = weights
    
    # Create temporary arrays the same size as the grid
    cdef np.ndarray[GRID_T, ndim=2] num_ = np.zeros_like(grid.knots)
    cdef np.ndarray[GRID_T, ndim=2] dnum_ = np.zeros_like(grid.knots)
    
    
    # For each pixel ...    
    for y in range(field_.shape[0]):
        for x in range(field_.shape[1]):
            
            # Get val and alpha
            val = field_[y, x]
            weight = weights_[y, x]
            
            # Evaluate this one?
            if weight <= 0.0:
                continue
            
            # Calculate what is the leftmost (reference) knot on the grid,
            # and the ratio between closest and second closest knot.
            # Note the +1 to correct for padding.
            tmp = (<double>y / grid_ySpacing ) + 1
            gy = <int>tmp
            ty	= tmp - <double>gy
            #
            tmp = (<double>x / grid_xSpacing ) + 1
            gx = <int>tmp
            tx	= tmp - <double>gx
            
            # Get coefficients
            ccy = coeffy.get_coef(ty)
            ccx = coeffx.get_coef(tx)
            
            # Pre-normalize value
            omsum = 0.0
            for j in range(4):
                for i in range(4):
                    omsum += ccy[j] * ccx[i] * ccy[j] * ccx[i]
            val_n = val / omsum
            
            # For each knot that this point influences
            # Following Lee et al. we update a numerator and a denumerator for
            # each knot.
            for j in range(4):
                jj = j+gy-1
                for i in range(4):
                    ii = i+gx-1
                    #
                    omega = ccy[j] * ccx[i]
                    omega2 = weight * omega * omega
                    num_[jj,ii] += omega2 * ( val_n*omega)
                    dnum_[jj,ii] += omega2
    
    # Done
    return num_, dnum_


@cython.boundscheck(False)
@cython.wraparound(False)
def _set_field3_64(grid, field, weights, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp, weight, omega, omega2, omsum
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    cdef AccurateCoef coeffy = AccurateCoef(lut)
    cdef AccurateCoef coeffz = AccurateCoef(lut)
    
    # Grid stuff
    cdef double grid_zSpacing = grid.grid_sampling_in_pixels[0]
    cdef double grid_ySpacing = grid.grid_sampling_in_pixels[1]
    cdef double grid_xSpacing = grid.grid_sampling_in_pixels[2]
    
    # Cast input arrrays
    cdef np.ndarray[FLOAT64_T, ndim=3] field_ = field
    cdef np.ndarray[FLOAT64_T, ndim=3] weights_ = weights
    
    # Create temporary arrays the same size as the grid
    cdef np.ndarray[GRID_T, ndim=3] num_ = np.zeros_like(grid.knots)
    cdef np.ndarray[GRID_T, ndim=3] dnum_ = np.zeros_like(grid.knots)
    
    
    # For each pixel ...    
    for z in range(field_.shape[0]):
        for y in range(field_.shape[1]):
            for x in range(field_.shape[2]):
                
                # Get val and alpha
                val = field_[z, y, x]
                weight = weights_[z, y, x]
                
                # Evaluate this one?
                if weight <= 0.0:
                    continue
                
                # Calculate what is the leftmost (reference) knot on the grid,
                # and the ratio between closest and second closest knot.
                # Note the +1 to correct for padding.
                tmp = (<double>z / grid_zSpacing ) + 1
                gz = <int>tmp
                tz	= tmp - <double>gz
                #
                tmp = (<double>y / grid_ySpacing ) + 1
                gy = <int>tmp
                ty	= tmp - <double>gy
                #
                tmp = (<double>x / grid_xSpacing ) + 1
                gx = <int>tmp
                tx	= tmp - <double>gx
                
                # Get coefficients
                ccz = coeffz.get_coef(tz)
                ccy = coeffy.get_coef(ty)
                ccx = coeffx.get_coef(tx)
                
                # Pre-normalize value
                omsum = 0.0
                for k in range(4):
                    for j in range(4):
                        for i in range(4):
                            omsum += ccz[k] * ccy[j] * ccx[i] * ccz[k] * ccy[j] * ccx[i]
                val_n = val / omsum
                
                # For each knot that this point influences
                # Following Lee et al. we update a numerator and a denumerator for
                # each knot.
                for k in range(4):
                    kk = k+gz-1
                    for j in range(4):
                        jj = j+gy-1
                        for i in range(4):
                            ii = i+gx-1
                            #
                            omega = ccz[k] * ccy[j] * ccx[i]
                            omega2 = weight * omega * omega
                            num_[kk,jj,ii] += omega2 * ( val_n*omega)
                            dnum_[kk,jj,ii] += omega2
    
    # Done
    return num_, dnum_


@cython.boundscheck(False)
@cython.wraparound(False)
def _set_field_sparse1_64(grid, pp, values, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp, weight, omega, omega2, omsum
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    
    # Grid stuff
    cdef double wGridSampling = grid.grid_sampling
    
    # Create num, dnum 
    cdef np.ndarray[GRID_T, ndim=1] num_ = np.zeros_like(grid.knots)
    cdef np.ndarray[GRID_T, ndim=1] dnum_ = np.zeros_like(grid.knots)
    
    # Cast pointset
    cdef np.ndarray[FLOAT64_T, ndim=2] pp_ = pp
    
    
    # For each point ...
    for p in range(pp_.shape[0]):
        
        # Get wx
        wx = pp_[p,0]
        
        # Calculate which is the closest point on the lattice to the top-left
        # corner and find ratio's of influence between lattice point.
        tmp = (wx / wGridSampling ) + 1
        gx = <int>tmp
        tx	= tmp - <double>gx
        
        # Get coefficients
        ccx = coeffx.get_coef(tx)
        
        # Precalculate omsum
        omsum = 0.0        
        for i in range(4):
            omsum += ccx[i] * ccx[i]
        
        # Get val
        val = values[p]
        
        # For each knot that this point influences
        # Following Lee et al. we update a numerator and a denumerator for
        # each knot.
        ii = gx - 1  # x-location of first knot
        for i in range(4):
            #
            omega = ccx[i]
            omega2 = omega*omega
            num_[ii] += omega2 * ( val*omega/omsum )
            dnum_[ii] += omega2
            #
            ii+=1
    
    # Done
    return num_, dnum_


@cython.boundscheck(False)
@cython.wraparound(False)
def _set_field_sparse2_64(grid, pp, values, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp, weight, omega, omega2, omsum
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    cdef AccurateCoef coeffy = AccurateCoef(lut)
    
    # Grid stuff
    cdef double wGridSampling = grid.grid_sampling
    
    # Create num, dnum 
    cdef np.ndarray[GRID_T, ndim=2] num_ = np.zeros_like(grid.knots)
    cdef np.ndarray[GRID_T, ndim=2] dnum_ = np.zeros_like(grid.knots)
    
    # Cast pointset
    cdef np.ndarray[FLOAT64_T, ndim=2] pp_ = pp
    
    
    # For each point ...
    for p in range(pp_.shape[0]):
        
        # Get wx and wy
        wx = pp_[p,0]
        wy = pp_[p,1]
        
        # Calculate which is the closest point on the lattice to the top-left
        # corner and find ratio's of influence between lattice point.
        tmp = (wy / wGridSampling ) + 1
        gy = <int>tmp
        ty	= tmp - <double>gy
        #
        tmp = (wx / wGridSampling ) + 1
        gx = <int>tmp
        tx	= tmp - <double>gx
        
        # Get coefficients
        ccy = coeffy.get_coef(ty)
        ccx = coeffx.get_coef(tx)
        
        # Precalculate omsum
        omsum = 0.0
        for j in range(4):
            for i in range(4):
                omsum += ccy[j] * ccx[i] * ccy[j] * ccx[i]
        
        # Get val
        val = values[p]
        
        # For each knot that this point influences
        # Following Lee et al. we update a numerator and a denumerator for
        # each knot.
        jj = gy - 1  # y-location of first knot
        for j in range(4):
            ii = gx - 1  # x-location of first knot
            for i in range(4):
                #
                omega = ccy[j] * ccx[i]
                omega2 = omega*omega
                num_[jj,ii] += omega2 * ( val*omega/omsum )
                dnum_[jj,ii] += omega2
                #
                ii+=1
            jj+=1
    
    # Done
    return num_, dnum_


@cython.boundscheck(False)
@cython.wraparound(False)
def _set_field_sparse3_64(grid, pp, values, spline_type='B'):
    
    # Predefs
    cdef int x, y, z, gx, gy, gz 
    cdef double wx, wy, wz
    cdef double tx, ty, tz
    cdef int p, i, j, k, ii, jj, kk
    cdef double *ccx, *ccy, *ccz
    cdef double val, tmp, weight, omega, omega2, omsum
    
    # Create and init cubic interpolator
    cdef CoefLut lut = CoefLut.get_lut(spline_type)
    cdef AccurateCoef coeffx = AccurateCoef(lut)
    cdef AccurateCoef coeffy = AccurateCoef(lut)
    cdef AccurateCoef coeffz = AccurateCoef(lut)
    
    # Grid stuff
    cdef double wGridSampling = grid.grid_sampling
    
    # Create num, dnum 
    cdef np.ndarray[GRID_T, ndim=3] num_ = np.zeros_like(grid.knots)
    cdef np.ndarray[GRID_T, ndim=3] dnum_ = np.zeros_like(grid.knots)
    
    # Cast pointset
    cdef np.ndarray[FLOAT64_T, ndim=2] pp_ = pp
    
    
    # For each point ...
    for p in range(pp_.shape[0]):
        
        # Get wx, wy and wz
        wx = pp_[p,0]
        wy = pp_[p,1]
        wz = pp_[p,2]
        
        # Calculate which is the closest point on the lattice to the top-left
        # corner and find ratio's of influence between lattice point.
        tmp = (wz / wGridSampling ) + 1
        gz = <int>tmp
        tz	= tmp - <double>gz
        #
        tmp = (wy / wGridSampling ) + 1
        gy = <int>tmp
        ty	= tmp - <double>gy
        #
        tmp = (wx / wGridSampling ) + 1
        gx = <int>tmp
        tx	= tmp - <double>gx
        
        # Get coefficients
        ccz = coeffz.get_coef(tz)
        ccy = coeffy.get_coef(ty)
        ccx = coeffx.get_coef(tx)
        
        # Precalculate omsum
        omsum = 0.0
        for k in range(4):
            for j in range(4):
                for i in range(4):
                    omsum += ccz[k] * ccy[j] * ccx[i] * ccz[k] * ccy[j] * ccx[i]
        
        # Get val
        val = values[p]
        
        # For each knot that this point influences
        # Following Lee et al. we update a numerator and a denumerator for
        # each knot.
        kk = gz -1  # z-location of first knot
        for k in range(4):
            jj = gy - 1  # y-location of first knot
            for j in range(4):
                ii = gx - 1  # x-location of first knot
                for i in range(4):
                    #
                    omega = ccy[j] * ccx[i]
                    omega2 = omega*omega
                    num_[jj,ii] += omega2 * ( val*omega/omsum )
                    dnum_[jj,ii] += omega2
                    #
                    ii+=1
                jj+=1
            kk+=1
    
    # Done
    return num_, dnum_
