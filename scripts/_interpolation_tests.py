""" Script with cells containing small tests.

Copyright 2010 (C) Almar Klein, University of Twente.

"""




## Error made by interpolating using a LUT
# Numerical precision of a float32 (eps): 1.19e-07
# Numerical precision of a float64 (eps): 2.22e-16
#
# Memory required for LUT: N*4*8: 2**15 -> 1MB
#
# Nearest Neighbour:
# For N= 128, the error is about 1%
# For N=1024, the error is below 0.14%
# For N=2048, the error is below 0.07%
#
# Linear C-splines:
# For N=4096 (2**12), the error is about 3.73e-08 (smaller then eps32)
# For N=32768 (2**15), the error is about 5.82e-10 (much smaller then eps32)
#
# Linear B-splines:
# For N=2048 (2**11), the error is about 5.96e-08 (smaller then eps32)
# For N=32768 (2**15), the error is about 2.32e-10 (much smaller then eps32)


# Init
import pirt
import visvis as vv

PP = range(7,16)
NN = [2**i for i in PP]
ST = 'C' # Spline Type

# Init error lists (for order 0:NN and 1:linear)
EE0 = []
EE1 = []

# Loop over values
for N in NN:
    # A range of 1.0 is divided in N steps, so the maximal error
    # in t (assuming the LUT simply performs round operator) 
    # is 0.5/N for NN. 
    # For linear interpolation in the LUT, the largest error is when
    # we are right in between two values (which are then combined each with
    # a factor 0.5)
    errors0 = []
    errors1 = []
    te = 0.5/N
    for t in [float(i)/N for i in range(N-1)]:            
        cc_real = pirt.get_cubic_spline_coefs(t, ST)
        cc_wrong1 = pirt.get_cubic_spline_coefs(t-te, ST)
        cc_wrong2 = pirt.get_cubic_spline_coefs(t+te, ST)
        errors0.append(abs( cc_real[1]-cc_wrong1[1] ))
        errors1.append(abs( cc_real[1]-0.5*(cc_wrong1[1]+cc_wrong2[1]) ))
    EE0.append(max(errors0))
    EE1.append(max(errors1))

# Visualize
vv.figure(2); vv.clf()
a = vv.gca()
a.axis.showGrid = 1
#vv.plot(PP, EE, ms='.')
#vv.plot(NN, EE0, ms='.', lc='b', mc='b')
vv.plot(NN, EE1, ms='.', lc='r', mc='r')


## Quadratic fit 1D

import pirt
import numpy as np
import visvis as vv
from pirt import Pointset, fitting

# Input and result
pp = [2, 4, 3]
t_max, polynom = fitting.fit_lq1(pp)

# Sample polynom
polypp = Pointset(2)
for i in np.linspace(-1,1,100):
    polypp.append(i, polynom[0]*i**2 + polynom[1]*i + polynom[2])

# Visualize
vv.figure()
vv.plot([-1,0,1],pp)
vv.plot([t_max, t_max], [0,1], lc='r')
vv.plot(polypp, lc='r')


## Test quadratic fit 2D

import pirt
import numpy as np
import visvis as vv
from pirt import Aarray, fitting

# Input and result    
im = vv.imread('lena.png')[::,::,2].copy()
Y,X = np.where(im==im.max())
Y,X = Y[0], X[0]
im[Y-1,X] -= 20 # make more interesting :)
mat = im[Y-1:Y+2,X-1:X+2]

# Get result and scale mat
ymin, ymax, surface = fitting.fit_lq2(mat, True)
mat = Aarray(mat, sampling=(100,100), origin=(50,50))

# Visualize
vv.figure()
vv.imshow(mat)
m = vv.surf(surface)
a = vv.gca()
a.SetLimits()
m.colormap = vv.CM_HOT





## Illustrate of 3+1 order interpolation in 3D

import pirt
import numpy as np
import visvis as vv

# Make volume
vol = np.zeros((90,90,90))
d = 30
vol[d:2*d,d:2*d,:] = 1
vol[:,d:2*d,d:2*d] = 1
vol[d:2*d,:,d:2*d] = 1

# Filter
vol = pirt.gfilter(vol,4)

# Visualize
vv.figure(1); vv.clf()
t=vv.volshow(vol,None, 'iso')
t.isoThreshold = 0.5
