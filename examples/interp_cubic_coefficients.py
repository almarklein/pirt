"""
Visualize the cubic coefficients.
"""

import pirt
import visvis as vv
import numpy as np


# Input
type = 'Cardinal'
tension = 0
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

# Interpolate (0 means Cardinal with tension 0)
samples = np.arange(0,len(data)-1,0.05, dtype=np.float32) 
values1 = pirt.warp(np.array(data, dtype=np.float32), samples, 3, 'linear')
values2 = pirt.warp(np.array(data, dtype=np.float32), samples, 3, 'basis')
values3 = pirt.warp(np.array(data, dtype=np.float32), samples, 3, 0)  


# Visualize
fig = vv.figure(1); vv.clf()
fig.position = 57.00, 45.00,  948.00, 969.00

vv.subplot(211)
vv.title('The basis functions for the %s spline' % type)
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
a.legend = 'data', 'Linear', 'Basis', 'Cardinal'

vv.use().Run()
