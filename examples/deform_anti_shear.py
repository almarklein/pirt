"""
This illustrates the shape of the (mathematical) function that limits
knot values to prefent folding and shearing (i.e. make the deformation
diffeomorphic).
"""

import visvis as vv
import numpy as np


c1 = np.linspace(-30, 30, 1000)

limit = 0.48 * 10
if True:
    I = c1<0
    c2 =     limit * (1 - np.exp(-c1/limit) )
    c2[I] = -limit * (1 - np.exp(c1[I]/limit) )
else:
    f = np.exp(-np.abs(c1)/limit)
    c2 = (f-1)* -np.sign(c1) * limit    

# Calculate derivative
n = int(len(c2)/2)
der1 = (c2[n+1] - c2[n]) / (c1[n+1] - c1[n])
der2 = (c2[n+2] - c2[n]) / (c1[n+2] - c1[n])
der3 = (c2[n+3] - c2[n]) / (c1[n+3] - c1[n])
print('Derivative', der1, der2, der3)

# Show
vv.figure(3); vv.clf()
a = vv.gca()
vv.plot(c1, c2)
a.axis.showGrid = 1
a.daspectAuto = 0
a.axis.xLabel = 'knot value'
a.axis.yLabel = 'new knot value'
