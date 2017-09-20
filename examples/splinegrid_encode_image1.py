"""
Example that encodes (the pixels of) an image using a B-spline grid,
by randomly sampling a bunch of pixels and using these to define
a B-spine grid.
"""

import numpy as np
import visvis as vv
from pirt import PointSet, SplineGrid


# Read image
im = vv.imread('astronaut.png').astype(np.float32)
im_r = im[:,:,0]

# New sparse image and interpolated image
ims = np.zeros_like(im)
imi = np.zeros_like(im)

# Select points from the image
pp = PointSet(2)
R, G, B = [], [], []
for i in range(10000):
    y = np.random.randint(0, im.shape[0])
    x = np.random.randint(0, im.shape[1])
    pp.append(x,y)
    R.append(im[y,x,0])
    G.append(im[y,x,1])
    B.append(im[y,x,2])
    ims[y,x] = im[y,x]

# Make three grids
spacing = 10
grid1 = SplineGrid(im_r, spacing)
grid2 = SplineGrid(im_r, spacing)
grid3 = SplineGrid(im_r, spacing)

# Put data in
grid1._set_using_points(pp, R)
grid2._set_using_points(pp, G)
grid3._set_using_points(pp, B)

# Obtain interpolated image
imi[:,:,0] = grid1.get_field()
imi[:,:,1] = grid2.get_field()
imi[:,:,2] = grid3.get_field()

# Show
vv.figure(1); vv.clf()
vv.subplot(131); vv.imshow(im)
vv.subplot(132); vv.imshow(ims)
vv.subplot(133); vv.imshow(imi)
