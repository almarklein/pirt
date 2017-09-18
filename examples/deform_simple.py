"""
Deform an image of a grid using pixel forces.
"""

import numpy as np
import visvis as vv
from pirt import DeformationGridForward

# Make grid image
im1 = np.zeros((100,100), dtype=np.float32)
im1[::10,:] = 1.0
im1[:,::10] = 1.0

# Make deform image
fx = np.zeros_like(im1)
fy = np.zeros_like(im1)
fa = np.zeros_like(im1)
fx[30,30] = 10
fy[25,20] = 20
fa[30,30] = 5
fa[25,20] = 5

# Make grid and deform
grid = DeformationGridForward.from_field([fx, fy], 15, fa)
im2 = grid.apply_deformation(im1, 3)

fig = vv.figure(11); vv.clf()
fig.position = 73.00, 67.00,  560.00, 794.00
vv.subplot(211); vv.imshow(im1)
vv.subplot(212); vv.imshow(im2)
