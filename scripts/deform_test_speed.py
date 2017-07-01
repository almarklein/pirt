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
print('deforming took', time.time()-t0, 'seconds')
vv.figure(2); vv.clf()
vv.imshow(im2)

vv.use().Run()
