""" Low level functions for spline grids, implemented with Numba to make it super fast.

Provides functionality for (cubic B-spline) grids. 

Copyright 2010-2012 (C) Almar Klein, University of Twente.
Copyright 2012-2017 (C) Almar Klein

A Note on pointsets:
In this module, all pointsets are 2D numpy arrays shaped (N, ndim), 
which express world coordinates. The coordinates they represent 
should be expressed using doubles wx,wy,wz, in contrast to pixel
coordinates, which are expressed using integers x, y, z. 
"""

import numpy as np
import numba

from ..interp._cubic import cubicsplinecoef_basis


## Functions to obtain the field that the grid represents


def get_field(grid):
    """ get_field(grid)
    Sample the grid at all the pixels of the underlying field.
    """
    
    # Init resulting array, make it float32, since for deformation-grids
    # the result is a sampler to be used in interp().
    result = np.zeros(grid.field_shape, dtype=np.float32)
    
    # Decide what function to call
    if grid.ndim == 1:
        _get_field1(result, grid.grid_sampling_in_pixels, grid.knots)
    elif grid.ndim == 2: 
        _get_field2(result, grid.grid_sampling_in_pixels, grid.knots)
    elif grid.ndim == 3: 
        _get_field3(result, grid.grid_sampling_in_pixels, grid.knots)
    else:  # nocov
        tmp = 'Grid interpolation not suported for this dimension.'
        raise RuntimeError(tmp)
    
    return result


def get_field_sparse(grid, pp):
    """ get_field_sparse(grid, pp)
    
    Sparsely sample the grid at a specified set of points (which are in
    world coordinates).
    
    Also see get_field_at(). 
    
    """
    assert isinstance(pp, np.ndarray) and pp.ndim == 2
    
    # Test dimensions
    if grid.ndim != pp.shape[1]:  # nocov
        raise ValueError('Dimension of grid and pointset do not match.')
    
    # Create samples
    samples = []
    for i in range(pp.shape[1]):
        samples.append(pp[:,i])
    
    # Init result
    result = np.zeros_like(samples[0], dtype=np.float32)
    
    # Determine sampling
    grid_sampling_in_pixels = tuple([grid.grid_sampling for i in grid.grid_sampling_in_pixels])
    
    # Decide what function to call
    if grid.ndim == 1:
        _get_field_at1(result.ravel(), grid_sampling_in_pixels,
                       grid.knots, *[s.ravel() for s in samples])
    elif grid.ndim == 2: 
        _get_field_at2(result.ravel(), grid_sampling_in_pixels,
                       grid.knots, *[s.ravel() for s in samples])
    elif grid.ndim == 3: 
        _get_field_at3(result.ravel(), grid_sampling_in_pixels,
                       grid.knots, *[s.ravel() for s in samples])
    else:  # nocov
        tmp = 'Grid interpolation not suported for this dimension.'
        raise RuntimeError(tmp)
    
    return result


def get_field_at(grid, samples):
    """ get_field_at(grid, samples)
    
    Sample the grid at specified sample locations (in pixels, x-y-z order),
    similar to pirt.interp.interp().
    
    Also see get_field_sparse().
    
    """
    
    # Test dimensions
    if not isinstance(samples, (tuple, list)):  # nocov
        raise ValueError('Samples must be list or tuple.')
    if len(samples) != grid.ndim:  # nocov
        raise ValueError('Samples must contain one element per dimension.')
    sample0 = samples[0]
    for sample in samples:
        if sample0.shape != sample.shape:  # nocov
            raise ValueError('Elements in samples must all have the same shape.')
    
    # Init result
    result = np.zeros_like(samples[0], dtype=np.float32)
    
    # Determine sampling
    grid_sampling_in_pixels = grid.grid_sampling_in_pixels
    
    # Decide what function to call
    if grid.ndim == 1:
        _get_field_at1(result.ravel(), grid_sampling_in_pixels,
                       grid.knots, *[s.ravel() for s in samples])
    elif grid.ndim == 2: 
        _get_field_at2(result.ravel(), grid_sampling_in_pixels,
                       grid.knots, *[s.ravel() for s in samples])
    elif grid.ndim == 3: 
        _get_field_at3(result.ravel(), grid_sampling_in_pixels,
                       grid.knots, *[s.ravel() for s in samples])
    else:  # nocov
        tmp = 'Grid interpolation not suported for this dimension.'
        raise RuntimeError(tmp)
    
    return result


## Workhorse functions to get the field


@numba.jit(nopython=True, nogil=True)
def _get_field1(result, grid_sampling_in_pixels, knots):
    
    if result.ndim != 1:
        raise ValueError('This function can only sample 1D grids.')
    
    ccx = np.empty((4, ), np.float64)
    
    grid_xSpacing = grid_sampling_in_pixels[0]
    
    # For each pixel ...    
    for x in range(result.shape[0]):
        
        # Calculate what is the leftmost (reference) knot on the grid,
        # and the ratio between closest and second closest knot.
        # Note the +1 to correct for padding.        
        tmp = x / grid_xSpacing + 1
        gx = int(tmp)
        tx = tmp - gx
        
        # Get coefficients
        cubicsplinecoef_basis(tx, ccx)
        
        # Init value
        val = 0.0
        
        # For each knot ...         
        ii = gx - 1  # x-location of first knot
        for i in range(4):
            # Calculate interpolated value.
            val += ccx[i] * knots[ii]
            ii += 1
        
        # Store value in result array
        result[x] = val


@numba.jit(nopython=True, nogil=True)
def _get_field2(result, grid_sampling_in_pixels, knots):
    
    if result.ndim != 2:
        raise ValueError('This function can only sample 2D grids.')
    
    ccy = np.empty((4, ), np.float64)
    ccx = np.empty((4, ), np.float64)
    
    grid_ySpacing = grid_sampling_in_pixels[0]
    grid_xSpacing = grid_sampling_in_pixels[1]
    
    
    # For each pixel ...
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            
            # Calculate what is the leftmost (reference) knot on the grid,
            # and the ratio between closest and second closest knot.
            # Note the +1 to correct for padding.
            tmp = y / grid_ySpacing + 1
            gy = int(tmp)
            ty = tmp - gy
            #
            tmp = x / grid_xSpacing + 1
            gx = int(tmp)
            tx = tmp - gx
            
            # Get coefficients
            cubicsplinecoef_basis(ty, ccy)
            cubicsplinecoef_basis(tx, ccx)
            
            # Init value
            val = 0.0
            
            # For each knot ... 
            jj = gy - 1  # y-location of first knot
            for j in range(4):
                ii = gx - 1  # x-location of first knot
                for i in range(4):
                    # Calculate interpolated value.
                    val += ccy[j] * ccx[i] * knots[jj, ii]
                    ii += 1
                jj += 1
            
            # Store value in result array
            result[y, x] = val


@numba.jit(nopython=True, nogil=True)
def _get_field3(result, grid_sampling_in_pixels, knots):
    
    if result.ndim != 3:
        raise ValueError('This function can only sample 3D grids.')
    
    ccz = np.empty((4, ), np.float64)
    ccy = np.empty((4, ), np.float64)
    ccx = np.empty((4, ), np.float64)
    
    grid_zSpacing = grid_sampling_in_pixels[0]
    grid_ySpacing = grid_sampling_in_pixels[1]
    grid_xSpacing = grid_sampling_in_pixels[2]
    
    # For each pixel ...
    for z in range(result.shape[0]):
        for y in range(result.shape[1]):
            for x in range(result.shape[2]):
                
                # Calculate what is the leftmost (reference) knot on the grid,
                # and the ratio between closest and second closest knot.
                # Note the +1 to correct for padding.
                tmp = z / grid_zSpacing + 1
                gz = int(tmp)
                tz = tmp - gz
                #
                tmp = y / grid_ySpacing + 1
                gy = int(tmp)
                ty = tmp - gy
                #
                tmp = x / grid_xSpacing + 1
                gx = int(tmp)
                tx = tmp - gx
                
                # Get coefficients
                cubicsplinecoef_basis(tz, ccz)
                cubicsplinecoef_basis(ty, ccy)
                cubicsplinecoef_basis(tx, ccx)

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
                            val += ccz[k] * ccy[j] * ccx[i] * knots[kk, jj, ii]
                            ii += 1
                        jj += 1
                    kk += 1
                
                # Store value in result array
                result[z, y, x] = val


@numba.jit(nopython=True, nogil=True)
def _get_field_at1(result, grid_sampling_in_pixels, knots, samplesx_):
    
    assert samplesx_.ndim == 1
    
    ccx = np.empty((4, ), np.float64)
    
    grid_xSpacing = grid_sampling_in_pixels[0]
    
    gridShapex = knots.shape[0]
    
    # For each point in the set
    for p in range(samplesx_.size):
        
        # Calculate wx
        wx = samplesx_[p]
        
        # Calculate what is the leftmost (reference) knot on the grid,
        # and the ratio between closest and second closest knot.
        # Note the +1 to correct for padding.
        tmp = wx / grid_xSpacing + 1
        gx = int(tmp)
        tx = tmp - gx
        
        # Check if within bounds of interpolatable domain
        if gx < 1 or gx >= gridShapex-2:
            result[p] = 0.0
            continue
        
        # Get coefficients
        cubicsplinecoef_basis(tx, ccx)
        
        # Init value
        val = 0.0
        
        # For each knot ... 
        ii = gx - 1  # x-location of first knot
        for i in range(4):
            # Calculate interpolated value.
            val += ccx[i] * knots[ii]
            ii += 1
        
        # Store
        result[p] = val


@numba.jit(nopython=True, nogil=True)
def _get_field_at2(result_, grid_sampling_in_pixels, knots, samplesx_, samplesy_):
    
    assert samplesx_.ndim == 1
    assert samplesy_.ndim == 1
    
    ccy = np.empty((4, ), np.float64)
    ccx = np.empty((4, ), np.float64)
    
    grid_ySpacing = grid_sampling_in_pixels[0]
    grid_xSpacing = grid_sampling_in_pixels[1]
    
    gridShapey = knots.shape[0]
    gridShapex = knots.shape[1]
    
    # For each point in the set
    for p in range(samplesx_.size):
        
        # Calculate wx and wy
        wx = samplesx_[p]
        wy = samplesy_[p]
        
        # Calculate what is the leftmost (reference) knot on the grid,
        # and the ratio between closest and second closest knot.
        # Note the +1 to correct for padding.
        tmp = wy / grid_ySpacing + 1
        gy = int(tmp)
        ty = tmp - gy
        #
        tmp = wx / grid_xSpacing + 1
        gx = int(tmp)
        tx = tmp - gx
        
        # Check if within bounds of interpolatable domain
        if (    (gy < 1 or gy >= gridShapey - 2) or
                (gx < 1 or gx >= gridShapex - 2) ):
            result_[p] = 0.0
            continue
        
        # Get coefficients
        cubicsplinecoef_basis(ty, ccy)
        cubicsplinecoef_basis(tx, ccx)
        
        # Init value
        val = 0.0
        
        # For each knot ... 
        jj = gy - 1  # y-location of first knot
        for j in range(4):
            ii = gx - 1  # x-location of first knot
            for i in range(4):
                # Calculate interpolated value.
                val += ccy[j] * ccx[i] * knots[jj, ii]
                ii += 1
            jj += 1
        
        # Store
        result_[p] = val


@numba.jit(nopython=True, nogil=True)
def _get_field_at3(result_, grid_sampling_in_pixels, knots, samplesx_, samplesy_, samplesz_):

    assert samplesx_.ndim == 1
    assert samplesy_.ndim == 1
    assert samplesz_.ndim == 1
    
    ccz = np.empty((4, ), np.float64)
    ccy = np.empty((4, ), np.float64)
    ccx = np.empty((4, ), np.float64)
    
    grid_zSpacing = grid_sampling_in_pixels[0]
    grid_ySpacing = grid_sampling_in_pixels[1]
    grid_xSpacing = grid_sampling_in_pixels[2]
    
    gridShapez = knots.shape[0]
    gridShapey = knots.shape[1]
    gridShapex = knots.shape[2]
    
    # For each point in the set
    for p in range(samplesx_.size):
        
        # Calculate wx and wy
        wx = samplesx_[p]
        wy = samplesy_[p]
        wz = samplesz_[p]
        
        # Calculate what is the leftmost (reference) knot on the grid,
        # and the ratio between closest and second closest knot.
        # Note the +1 to correct for padding.
        tmp = wz / grid_zSpacing + 1
        gz = int(tmp)
        tz = tmp - gz
        #
        tmp = wy / grid_ySpacing + 1
        gy = int(tmp)
        ty = tmp - gy
        #
        tmp = wx / grid_xSpacing + 1
        gx = int(tmp)
        tx = tmp - gx
        
        # Check if within bounds of interpolatable domain
        if (    (gx < 1 or gx >= gridShapex - 2) or
                (gy < 1 or gy >= gridShapey - 2) or
                (gz < 1 or gz >= gridShapez - 2)):
            result_[p] = 0.0
            continue
        
        # Get coefficients
        cubicsplinecoef_basis(tz, ccz)
        cubicsplinecoef_basis(ty, ccy)
        cubicsplinecoef_basis(tx, ccx)
        
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
                    val += ccz[k] * ccy[j] * ccx[i] * knots[kk, jj, ii]
                    ii+=1
                jj += 1
            kk += 1
        
        # Store
        result_[p] = val


## Functions to set the grid using a field


@numba.jit(nopython=True, nogil=True)
def _set_field_using_num_and_dnum(knots_, num_, dnum_):
    
    for i in range(knots_.size):
        n = dnum_[i]
        if n > 0.0:
            knots_[i] = num_[i] / n


def set_field(grid, field, weights):
    """ set_field(grid, pp)
    Set the grid using the specified field (and optional weights).
    """
    # Test dimensions
    if grid.field_shape != field.shape:  # nocov
        raise ValueError('Dimension of grid-field and field do not match.')
    
    # Test dtype
    if field.dtype != weights.dtype:  # nocov
        raise ValueError('Field and weights must be of the same type.')
    
    # Apply proper function
    if grid.ndim == 1:
        num, dnum = _set_field1(grid.grid_sampling_in_pixels, grid.knots, field, weights)
    elif grid.ndim == 2:
        num, dnum = _set_field2(grid.grid_sampling_in_pixels, grid.knots, field, weights)
    elif grid.ndim == 3:
        num, dnum = _set_field3(grid.grid_sampling_in_pixels, grid.knots, field, weights)
    else:  # nocov
        tmp = 'This method does not support grids of that dimension.'
        raise RuntimeError(tmp)
    
    # Apply
    _set_field_using_num_and_dnum(grid.knots.ravel(), num.ravel(), dnum.ravel())


def set_field_sparse(grid, pp, values):
    """ set_field_sparse(grid, pp, values)
    
    Set the grid by providing the field values at a set of points (wich
    are in world coordinates).  
    
    """
    assert isinstance(pp, np.ndarray) and pp.ndim == 2
    
    # Test dimensions
    if grid.ndim != pp.shape[1]:  # nocov
        raise ValueError('Dimension of grid and pointset do not match.')
    
    # Apply proper function
    if grid.ndim == 1:
        num, dnum = _set_field_sparse1(grid.grid_sampling, grid.knots, pp, values)
    elif grid.ndim == 2:
        num, dnum = _set_field_sparse2(grid.grid_sampling, grid.knots, pp, values)
    elif grid.ndim == 3:
        num, dnum = _set_field_sparse3(grid.grid_sampling, grid.knots, pp, values)
    else:  # nocov
        tmp = 'This method does not support grids of that dimension.'
        raise RuntimeError(tmp)

    # Apply
    _set_field_using_num_and_dnum(grid.knots.ravel(), num.ravel(), dnum.ravel())


## Workhorse functions to set the field


@numba.jit(nopython=True, nogil=True)
def _set_field1(grid_sampling_in_pixels, knots, field, weights):
    
    ccx = np.empty((4, ), np.float64)
    
    grid_xSpacing = grid_sampling_in_pixels[0]
    
    # Create temporary arrays the same size as the grid
    num = np.zeros_like(knots)
    dnum = np.zeros_like(knots)
    
    # For each pixel ...        
    for x in range(field.shape[0]):
        
        # Get val and alpha
        val = field[x]
        weight = weights[x]
        
        # Evaluate this one?
        if weight <= 0.0:
            continue
        
        # Calculate what is the leftmost (reference) knot on the grid,
        # and the ratio between closest and second closest knot.
        # Note the +1 to correct for padding.
        tmp = x / grid_xSpacing + 1
        gx = int(tmp)
        tx = tmp - gx
        
        # Get coefficients
        cubicsplinecoef_basis(tx, ccx)
        
        # Pre-normalize value
        omsum = 0.0
        for i in range(4):
            omsum += ccx[i] * ccx[i]
        val_n = val / omsum
        
        # For each knot that this point influences
        # Following Lee et al. we update a numerator and a denumerator for
        # each knot.
        for i in range(4):
            ii = i + gx - 1
            #
            omega = ccx[i]
            omega2 = weight * omega * omega
            num[ii] += omega2 * (val_n * omega)
            dnum[ii] += omega2
    
    # Done
    return num, dnum


@numba.jit(nopython=True, nogil=True)
def _set_field2(grid_sampling_in_pixels, knots, field, weights):
    
    ccy = np.empty((4, ), np.float64)
    ccx = np.empty((4, ), np.float64)
    
    grid_ySpacing = grid_sampling_in_pixels[0]
    grid_xSpacing = grid_sampling_in_pixels[1]
    
    # Create temporary arrays the same size as the grid
    num = np.zeros_like(knots)
    dnum = np.zeros_like(knots)
    
    # For each pixel ...    
    for y in range(field.shape[0]):
        for x in range(field.shape[1]):
            
            # Get val and alpha
            val = field[y, x]
            weight = weights[y, x]
            
            # Evaluate this one?
            if weight <= 0.0:
                continue
            
            # Calculate what is the leftmost (reference) knot on the grid,
            # and the ratio between closest and second closest knot.
            # Note the +1 to correct for padding.
            tmp = y / grid_ySpacing + 1
            gy = int(tmp)
            ty = tmp - gy
            #
            tmp = x / grid_xSpacing + 1
            gx = int(tmp)
            tx = tmp - gx
            
            # Get coefficients
            cubicsplinecoef_basis(ty, ccy)
            cubicsplinecoef_basis(tx, ccx)
            
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
                jj = j + gy - 1
                for i in range(4):
                    ii = i + gx - 1
                    #
                    omega = ccy[j] * ccx[i]
                    omega2 = weight * omega * omega
                    num[jj,ii] += omega2 * ( val_n*omega )
                    dnum[jj,ii] += omega2
    
    # Done
    return num, dnum


@numba.jit(nopython=True, nogil=True)
def _set_field3(grid_sampling_in_pixels, knots, field, weights):
    
    ccz = np.empty((4, ), np.float64)
    ccy = np.empty((4, ), np.float64)
    ccx = np.empty((4, ), np.float64)
    
    grid_zSpacing = grid_sampling_in_pixels[0]
    grid_ySpacing = grid_sampling_in_pixels[1]
    grid_xSpacing = grid_sampling_in_pixels[2]
    
    # Create temporary arrays the same size as the grid
    num = np.zeros_like(knots)
    dnum = np.zeros_like(knots)
    
    # For each pixel ...    
    for z in range(field.shape[0]):
        for y in range(field.shape[1]):
            for x in range(field.shape[2]):
                
                # Get val and alpha
                val = field[z, y, x]
                weight = weights[z, y, x]
                
                # Evaluate this one?
                if weight <= 0.0:
                    continue
                
                # Calculate what is the leftmost (reference) knot on the grid,
                # and the ratio between closest and second closest knot.
                # Note the +1 to correct for padding.
                tmp = z / grid_zSpacing + 1
                gz = int(tmp)
                tz = tmp - gz
                #
                tmp = y / grid_ySpacing + 1
                gy = int(tmp)
                ty = tmp - gy
                #
                tmp = x / grid_xSpacing + 1
                gx = int(tmp)
                tx = tmp - gx
                
                # Get coefficients
                cubicsplinecoef_basis(tz, ccz)
                cubicsplinecoef_basis(ty, ccy)
                cubicsplinecoef_basis(tx, ccx)
                
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
                    kk = k + gz - 1
                    for j in range(4):
                        jj = j + gy - 1
                        for i in range(4):
                            ii = i + gx - 1
                            #
                            omega = ccz[k] * ccy[j] * ccx[i]
                            omega2 = weight * omega * omega
                            num[kk,jj,ii] += omega2 * ( val_n*omega )
                            dnum[kk,jj,ii] += omega2
    
    # Done
    return num, dnum


@numba.jit(nopython=True, nogil=True)
def _set_field_sparse1(grid_sampling, knots, pp, values):
    
    ccx = np.empty((4, ), np.float64)
    
    wGridSampling = grid_sampling
    
    # Create num, dnum 
    num = np.zeros_like(knots)
    dnum = np.zeros_like(knots)
    
    # For each point ...
    for p in range(pp.shape[0]):
        
        # Get wx
        wx = pp[p, 0]
        
        # Calculate which is the closest point on the lattice to the top-left
        # corner and find ratio's of influence between lattice point.
        tmp = wx / wGridSampling + 1
        gx = int(tmp)
        tx = tmp - gx
        
        # Get coefficients
        cubicsplinecoef_basis(tx, ccx)
        
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
            num[ii] += omega2 * ( val*omega/omsum )
            dnum[ii] += omega2
            #
            ii += 1
    
    # Done
    return num, dnum


@numba.jit(nopython=True, nogil=True)
def _set_field_sparse2(grid_sampling, knots, pp, values):
    
    ccy = np.empty((4, ), np.float64)
    ccx = np.empty((4, ), np.float64)
    
    wGridSampling = grid_sampling
    
    # Create num, dnum 
    num = np.zeros_like(knots)
    dnum = np.zeros_like(knots)
    
    # For each point ...
    for p in range(pp.shape[0]):
        
        # Get wx and wy
        wx = pp[p, 0]
        wy = pp[p, 1]
        
        # Calculate which is the closest point on the lattice to the top-left
        # corner and find ratio's of influence between lattice point.
        tmp = wy / wGridSampling + 1
        gy = int(tmp)
        ty = tmp - gy
        #
        tmp = wx / wGridSampling + 1
        gx = int(tmp)
        tx = tmp - gx
        
        # Get coefficients
        cubicsplinecoef_basis(ty, ccy)
        cubicsplinecoef_basis(tx, ccx)
        
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
                num[jj, ii] += omega2 * ( val*omega/omsum )
                dnum[jj, ii] += omega2
                #
                ii += 1
            jj += 1
    
    # Done
    return num, dnum


@numba.jit(nopython=True, nogil=True)
def _set_field_sparse3(grid_sampling, knots, pp, values):
    
    ccz = np.empty((4, ), np.float64)
    ccy = np.empty((4, ), np.float64)
    ccx = np.empty((4, ), np.float64)
    
    wGridSampling = grid_sampling
    
    # Create num, dnum 
    num = np.zeros_like(knots)
    dnum = np.zeros_like(knots)
    
    # For each point ...
    for p in range(pp.shape[0]):
        
        # Get wx, wy and wz
        wx = pp[p, 0]
        wy = pp[p, 1]
        wz = pp[p, 2]
        
        # Calculate which is the closest point on the lattice to the top-left
        # corner and find ratio's of influence between lattice point.
        tmp = wz / wGridSampling + 1
        gz = int(tmp)
        tz = tmp - gz
        #
        tmp = wy / wGridSampling + 1
        gy = int(tmp)
        ty = tmp - gy
        #
        tmp = wx / wGridSampling + 1
        gx = int(tmp)
        tx = tmp - gx
        
        # Get coefficients
        cubicsplinecoef_basis(tz, ccz)
        cubicsplinecoef_basis(ty, ccy)
        cubicsplinecoef_basis(tx, ccx)
        
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
                    omega = ccz[k] * ccy[j] * ccx[i]
                    omega2 = omega*omega
                    num[kk, jj, ii] += omega2 * ( val*omega/omsum )
                    dnum[kk, jj, ii] += omega2
                    #
                    ii += 1
                jj += 1
            kk += 1
    
    # Done
    return num, dnum
