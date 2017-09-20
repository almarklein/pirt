"""
This example demonstrates quadratic fitting in 1D and 2D.

"""

import numpy as np
import visvis as vv

from pirt import PointSet, Aarray
from pirt import fitting

vv.figure(1)
vv.clf()


## 1D

# Input and result
pp = [2, 4, 3]
t_max, polynom = fitting.fit_lq1(pp)

# Sample polynom
polypp = PointSet(2)
for i in np.linspace(-1,1,100):
    polypp.append(i, polynom[0]*i**2 + polynom[1]*i + polynom[2])

# Visualize
vv.subplot(121)
vv.plot([-1,0,1],pp)
vv.plot([t_max, t_max], [0,1], lc='r')
vv.plot(polypp, lc='r')


## 2D


# Input and result    
im = vv.imread('astronaut.png')[::,::,2].copy()
Y,X = np.where(im==im.max())
Y,X = Y[0], X[0]
im[Y-1,X] -= 20 # make more interesting :)
mat = im[Y-1:Y+2, X-1:X+2]

# Get result and scale mat
ymin, ymax, surface = fitting.fit_lq2(mat, True)
mat = Aarray(mat, sampling=(100, 100), origin=(50, 50))

# Visualize
vv.subplot(122)
vv.imshow(mat)
m = vv.surf(surface)
a = vv.gca()
a.SetLimits()
m.colormap = vv.CM_MAGMA
