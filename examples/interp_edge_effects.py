"""
Visualize interpolation edge effects.
"""

import numpy as np
import visvis as vv
from pirt import resize


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

vv.use().Run()
