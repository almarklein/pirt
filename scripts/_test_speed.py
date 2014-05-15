
import scipy.ndimage
import pirt
import time
import numpy as np

im = np.zeros((1000,10000), 'float32')
factor = 0.5

t0 = time.time()
im2 = scipy.ndimage.zoom(im, 0.5,order=3)
print 'scipy', time.time() - t0

t0 = time.time()
im3 = pirt.zoom(im, 0.5,order=3)
print 'pirt', time.time() - t0
