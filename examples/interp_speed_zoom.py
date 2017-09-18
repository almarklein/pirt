"""
Test speed of zoom() function in scipy.ndimage, scikit-image, and pirt.
"""

import time

import numpy as np
import scipy.ndimage
from skimage.transform import rescale
import pirt

print('Test speed of scipy.ndimage.zoom vs skimage.rescale vs pirt.zoom')

im = np.zeros((1000,10000), 'float32')
factor = 0.5

# Scipy
t0 = time.time()
im2 = scipy.ndimage.zoom(im, 0.5, order=3)
print('scipy', time.time() - t0)

# Skimage
t0 = time.time()
im3 = rescale(im, 0.5, order=3)
print('skimage', time.time() - t0)

# Pirt
#
# Jit warmup
pirt.zoom(im[:9,:9].copy(), 0.5, order=3)
#
t0 = time.time()
im3 = pirt.zoom(im, 0.5, order=3)
print('pirt', time.time() - t0)
