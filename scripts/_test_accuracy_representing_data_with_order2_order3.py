import time
import numpy as np
import visvis as vv
import pirt

sigma = 10
factor = 2

def get_errors(sigma, factor, verbose=True, N=100000):
    
    if isinstance(factor, (list, tuple)):
        factor1, factor2 = factor
    else:
        factor1 = factor
        factor2 = factor1*2
    
    # Create smooth random data
    a00 = np.random.normal(0,1,(N,))
    a0 = pirt.diffuse(a00, sigma)
    a0 /= 1.0/sigma**0.5 #pirt.gaussiankernel(sigma).max()
    
    # Regularize with B-spline grid
    grid = pirt.SplineGrid.from_field_multiscale(a00, sigma)
    a0 = grid.get_field()
    
    # Decimate
    b1 = a0[::factor1]
    b2 = a0[::factor2]
    
    # Sample points 
    t1 = np.arange(0, len(b1), 1.0/factor1, dtype='float32')
    t2 = np.arange(0, len(b2), 1.0/factor2, dtype='float32')
    
    # Increase resolution again (using different methods)
    times = []
    t0 = time.time()
    a11 = pirt.interp(b1, t1, 'linear') 
    times.append(time.time()-t0); t0 = time.time();
    a12 = pirt.interp(b2, t2, 'linear')
    times.append(time.time()-t0); t0 = time.time();
    a31 = pirt.interp(b1, t1, 'cubic')
    times.append(time.time()-t0); t0 = time.time();
    a32 = pirt.interp(b2, t2, 'cubic')
    times.append(time.time()-t0); t0 = time.time();
    
    arrays = [a11, a12, a31, a32]
    factors = [factor1, factor2, factor1, factor2]
    interps = ['linear', 'linear', 'cubic ', 'cubic ']
    errors = []
    for i in range(len(arrays)):
        a = arrays[i][20:-20]
        err = (a0[20:len(a)+20]-a)**2        
        #perr = 100.0 * err.mean()# / (a0[20:-20]**2).mean()
        perr = 100.0 * ( (err**0.5).mean() + 3*(err**0.5).std())
        # todo: show with boxplots?
        errors.append(perr)
        if verbose:
            print ' factor %i, %s: %1.5f%% (%1.0f ms)' % (
                    factors[i], interps[i], float(perr), times[i]*1000.0 )
    
    if verbose:
        vv.figure(1); vv.clf()
        vv.plot(a0, ms='.', lc='k', lw=2, mc='k', mw=4)
        vv.plot(a11, ms='', mc='r', ls='-', lc='r')
        vv.plot(a32, ms='', mc='b', ls='-', lc='b')
    
    return errors

factor = 2,4
get_errors(5, factor,N=100000)
a = vv.gca()
a.SetLimits(rangeX=(100,200))

sigmas = np.arange(5, 15,0.5)
errors = [[],[],[],[]]
for sigma in sigmas:
    error = get_errors(float(sigma), factor, False)
    for i in range(4):
        errors[i].append(error[i])

vv.figure(2); vv.clf()
colors = ['r', 'r', 'b', 'b']
widths = [2,1,1,2]
for i in range(4):
    vv.plot(sigmas, errors[i], lc=colors[i], lw=widths[i])
a = vv.gca()
a.SetLimits(rangeY=(-0.001,0.005))

