""" Attempt to implement convolution, so that we do not need scipy.
I can't get the code as fast as scipys though (maybe they use threading?

""" 

import time

import scipy as sp
import scipy.ndimage

# Compile cython
from pyzo import pyximport
pyximport.install()
import convolution1D_


    
if __name__ == '__main__':
    import numpy as np
    
    import visvis as vv
    tmp = vv.imread('lena.png')[:,:,1]
    shape1 = tmp.shape
    shape2 = [s*2 for s in shape1]
    im = np.zeros(shape2, dtype='float32')
    im[:shape1[0],:shape1[1]] = tmp
    im[shape1[0]:,:shape1[1]] = tmp
    im[:shape1[0],shape1[1]:] = tmp
    im[shape1[0]:,shape1[1]:] = tmp
    
    k = np.array([1, 1, 1, 1, 0,-1, -1, -1, -1], dtype='float32')/3.0
    
    axis = 0
    
    t0 = time.time()
    for i in range(10):
        im2 = convolution1D_.convolve2_32(im, k, axis)
    t1 = time.time()
    for i in range(10):
        im3 = sp.ndimage.convolve1d(im, k, axis)
    t2 = time.time()
    for i in range(10):
        im2 = convolution1D_.convolve2_32(im, k, axis)
    t3 = time.time()
    
    print 'Mine:', t1-t0, 's'
    print 'Scipy:', t2-t1, 's'
    print 'Mine:', t3-t2, 's'
    
    vv.figure(1); vv.clf()
    a1 = vv.subplot(221); vv.imshow(im)
    a2 = vv.subplot(222); vv.imshow(im2)
    a3 = vv.subplot(223); vv.imshow(im3)
    a4 = vv.subplot(224); vv.imshow(im3-im2)
    
    
    