""" Script with cells containing small tests.

Copyright 2010 (C) Almar Klein, University of Twente.

"""

## Slicing a volume
# Loads a volume, so only works at my PC

import pirt
import stentDirect
import visvis as vv
from pirt import Point

# Load volume and get z position of slice 100
vol = stentDirect.loadVol(1)
z100 = (100 - vol.origin[0]) / vol.sampling[0]

# Get two slices representations
slice1 = pirt.SliceInVolume(Point(150,80,100))
slice2 = pirt.SliceInVolume(Point(152,83,106), previous=slice1)

# Show the slices they represent, plus the raw slice at z=100
fig = vv.figure(1); vv.clf()
vv.subplot(221); vv.imshow(vol[z100,:,:])
vv.subplot(223); vv.imshow(slice1.get_slice(vol, 128, 0.5))
vv.subplot(224); vv.imshow(slice2.get_slice(vol, 128, 0.5))


## Cubic coefficients

import pirt
import visvis as vv
import numpy as np

# Input
type = 'linear'
tension = 0.5
data = [1,4,4,2, 5, 3, 3, 3, 4, 5, 6]

# Calculate basis functions
tt = np.arange(0,1,0.005)
vv0 = np.zeros_like(tt)
vv1 = np.zeros_like(tt)
vv2 = np.zeros_like(tt)
vv3 = np.zeros_like(tt)
vvt = np.zeros_like(tt)
for i in range(len(tt)):
    cc = pirt.get_cubic_spline_coefs(tt[i], type)
    vv0[i] = cc[0]
    vv1[i] = cc[1]
    vv2[i] = cc[2]
    vv3[i] = cc[3]
    vvt[i] = sum(cc)

# Interpolate
samples = np.arange(0,len(data)-1,0.05, dtype=np.float32) 
values1 = pirt.interp(np.array(data, dtype=np.float32), samples, 3, 0 )
values2 = pirt.interp(np.array(data, dtype=np.float32), samples, 3,'linear')
values3 = pirt.interp(np.array(data, dtype=np.float32), samples, 3,'b')

# Visualize
fig = vv.figure(1); vv.clf()
fig.position = 57.00, 45.00,  948.00, 969.00

vv.subplot(211)
vv.plot(tt, vv0, lc='r')
vv.plot(tt, vv1, lc='g')
vv.plot(tt, vv2, lc='b')
vv.plot(tt, vv3, lc='m')
vv.plot(tt, vvt, lc='k', ls='--')

vv.subplot(212)
vv.plot(range(len(data)), data, ms='.', ls='')
vv.plot(samples, values1, lc='r')
vv.plot(samples, values2, lc='g')
vv.plot(samples, values3, lc='b', lw=3)
a = vv.gca()
a.legend = 'data', 'Cardinal', 'Linear', 'Basis'



## Edge effects for interpolating lines or images

import numpy as np
import visvis as vv
import pirt
from pirt import resize, imresize


# Create low resolution image
im1 = np.zeros((7,7), dtype=np.float32) + 0.2
im1[1:-1,1:-1] += 1.0

# Create high resolution image
im2 = np.zeros((70,70), dtype=np.float32)
im2[10:-10,10:-10] = 1.0
im2[35,:] = im2[0,0]

# Create higher/lower resolution image
order = 3
im3 = resize(im1, (70,70), order, prefilter=1)
im4 = resize(im2, (35,35), order, prefilter=1)


# Visualize
fig = vv.figure(2); vv.clf()
fig.position  = 140.00, 205.00,  760.00, 420.00
vv.subplot(231); vv.imshow(im1)
vv.subplot(232); vv.imshow(im3)
vv.subplot(233); vv.plot(im3[20,:])
#
vv.subplot(234); vv.imshow(im2)
vv.subplot(235); vv.imshow(im4)
vv.subplot(236); vv.plot(im4[5,:])



## Interpolation of 1D signal

import numpy as np
import visvis as vv
import pirt

# Create 1D signal
t0 = np.linspace(0,6,400).astype('float32')
a0 = np.sin(t0)+1

# Create decimated version
factor = 50
a1 = pirt.zoom(a0, 1.0/factor, 3, prefilter=0)
t1 = np.arange(a1.origin[0], a1.origin[0] + a1.sampling[0]* a1.shape[0], a1.sampling[0])

# Create increased resolution
factor = 2
a2 = pirt.zoom(a1, factor, 1, 0, extra=0)
a3 = pirt.zoom(a1, factor, 3, 'C', extra=0)

t2 = np.arange(a2.origin[0], a2.origin[0] + a2.sampling[0]* a2.shape[0], a2.sampling[0])
t3 = np.arange(a3.origin[0], a3.origin[0] + a3.sampling[0]* a3.shape[0], a3.sampling[0])

# Create increased resolution and beyond
t9 = np.linspace(-2, a1.shape[0]+1, 100).astype('float32')
a9 = pirt.interp(a1, (t9,), 3) 

# Visualize
fig = vv.figure(2); vv.clf()
fig.position  = 140.00, 205.00,  560.00, 560.00
vv.subplot(211)
vv.plot(a0, ls=':', lc='k', lw=2)
vv.plot(t1, a1, ms='x', mw=9, ls='', mc='k')
#
vv.plot(t2, a2, ms='.', ls='-', lc='r', mc='r')
vv.plot(t3, a3, ms='.', ls='-', lc='g', mc='g')
#
vv.gca().axis.showGrid = True
vv.title('Demonstrate resizing')

vv.subplot(212)
vv.plot(range(a1.shape[0]),a1, ms='x', mw=9, ls='', mc='k')
vv.plot(t9, a9, ms='', ls='-', lw=2, lc='g')
vv.gca().axis.showGrid = True
vv.title('Show that interpolation also does a bit of extrapolation.')


## Test speed of interpolation (deformation)

# Speeds, nn, lin, cubic
# First try:        0.31, 0.56, 1.41
# With decorators:  0.23, 0.43, 1.07
# Without mode and negative_indices: dito
import visvis as vv
import numpy as np
import pirt
import time

im = vv.imread('lena.png')[:,:,1].astype(np.float32) / 255
#     dx = np.ones_like(im) * 40    
#     dy = np.ones_like(im) * 40
grids = np.meshgrid(    np.arange(0,im.shape[1],0.2), 
                        np.arange(0,im.shape[0],0.2))
grids = [g.astype(np.float32) for g in grids]
dx, dy = tuple(grids)
t0 = time.time()
im2 = pirt.interp(im, (dx, dy), order=3, spline_type=-0.3)
print 'deforming took', time.time()-t0, 'seconds'    
vv.figure(2); vv.clf()
vv.imshow(im2)


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


## Investigate quadratic spline
# Fitting a quadratic spline from one side and from the other side
# (each with a support of 3) results in two different polynoms, which
# are best combined by linear interpolation. The result is a cubic cardinal
# spline!

import pirt
import numpy as np
import visvis as vv
from pirt import Pointset, Point

# Input
pp = [2, 4, 3, 1]

# Interpolate1
res1 = Pointset(2)
for t in np.arange(0,1,0.01):
    c_1 = 0.5*t**2 - 0.5*t
    c0 = -t**2 + 1
    c1 = 0.5*t**2 + 0.5*t
    res1.append(t, c_1*pp[0] + c0*pp[1] + c1*pp[2])

# Interpolate2
res2 = Pointset(2)
for t in np.arange(0,1,0.01):
    c0 = 0.5*t**2 - 1.5*t + 1
    c1 = -t**2 + 2*t
    c2 = 0.5*t**2 - 0.5*t
    res2.append(t, c0*pp[1] + c1*pp[2] + c2*pp[3])

# Linearly combine
res3 = Pointset(2)
for t in np.arange(0,1,0.01):
    # NOTE: if I write this out, I'll get a cubic spline!
    c_1 = (0.5*t**2 - 0.5*t) * (1-t)
    c0 = (-t**2 + 1) * (1-t) + (0.5*t**2 - 1.5*t + 1) * t
    c1 = (0.5*t**2 + 0.5*t) * (1-t) + (-t**2 + 2*t) * t
    c2 =  (0.5*t**2 - 0.5*t) * t
    res3.append(t, c_1*pp[0] + c0*pp[1] + c1*pp[2] + c2*pp[3])

# Combine by adding
res3 = 0.5*(res1 + res2)

# To compare, cardinal spline
resC = Pointset(2)
for t in np.arange(0,1,0.01):
    cc = pirt.get_cubic_spline_coefs(t, 0)
    resC.append(t, cc[0]*pp[0] + cc[1]*pp[1] + cc[2]*pp[2] + cc[3]*pp[3])

# Show
vv.figure(1); vv.clf()
vv.plot(res1, lc='r', lw=1)
vv.plot(res2, lc='b', lw=1)
vv.plot(res3, lc='g', lw=2)
vv.plot(resC, lc='m', lw=5, ls=':')
vv.plot([-1, 0, 1, 2],pp, lc='k', ls='--')
#
vv.legend('From the left', 'From the right', 'Combined', 'Catmull-rom')


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
