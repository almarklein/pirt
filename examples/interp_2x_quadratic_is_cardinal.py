"""
Investigate quadratic spline

Fitting a quadratic spline from one side and from the other side
(each with a support of 3) results in two different polynoms, which
are best combined by linear interpolation. The result is a cubic cardinal
spline! To me, this suggests that the Cardinal spline (with tension 0)
is the most "natural" cubic spline.
"""

import numpy as np
import visvis as vv
import pirt
from pirt import PointSet


# Input
pp = [2, 4, 3, 1]

# Interpolate1
res1 = PointSet(2)
for t in np.arange(0,1,0.01):
    c_1 = 0.5*t**2 - 0.5*t
    c0 = -t**2 + 1
    c1 = 0.5*t**2 + 0.5*t
    res1.append(t, c_1*pp[0] + c0*pp[1] + c1*pp[2])

# Interpolate2
res2 = PointSet(2)
for t in np.arange(0,1,0.01):
    c0 = 0.5*t**2 - 1.5*t + 1
    c1 = -t**2 + 2*t
    c2 = 0.5*t**2 - 0.5*t
    res2.append(t, c0*pp[1] + c1*pp[2] + c2*pp[3])

# Linearly combine
res3 = PointSet(2)
for t in np.arange(0,1,0.01):
    # NOTE: if I write this out, I'll get a cubic spline!
    c_1 = (0.5*t**2 - 0.5*t) * (1-t)
    c0 = (-t**2 + 1) * (1-t) + (0.5*t**2 - 1.5*t + 1) * t
    c1 = (0.5*t**2 + 0.5*t) * (1-t) + (-t**2 + 2*t) * t
    c2 =  (0.5*t**2 - 0.5*t) * t
    res3.append(t, c_1*pp[0] + c0*pp[1] + c1*pp[2] + c2*pp[3])

# Combine by adding
# res3 = 0.5*(res1 + res2)

# To compare, cardinal spline
resC = PointSet(2)
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

vv.use().Run()
