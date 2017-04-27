"""
Pirt's interp function is 4x faster than skimage warp(). 

The new Numba version is about as fast for cubic interpolation. For linear
its about twice as fast. For quasi linear its much faster, probably because
the old pirt version of quasi linear has a missing cdef or something.

Cuda can make things faster, seems about 2.5 x, but only for larger images,
and maybe the effect is negligable for linear/quasi linear.

"""

# todo: all backward funcs
# todo: all forward funcs
# todo: fast version of CM spline?
# todo: slice in volume
# todo: splinegrid cython
# todo: check other todos
# todo: perhaps we can speed other things up, e.g. via a quasi-exponent function

import sys
from time import perf_counter
from skimage.transform import warp as skimage_warp
import visvis as vv
import numpy as np
import numba
from numba import cuda
import imageio

try:
    import pirt.interp
except ImportError:
    pirt = None
    print('no pirt')


im1 = imageio.imread('imageio:chelsea.png')[:300, :300, 2].astype('float32') / 255
# im1 = np.row_stack([im1] * 8)
# im1 = np.column_stack([im1] * 8)
im2 = np.zeros_like(im1)
im3 = np.zeros_like(im1)
im4 = np.zeros_like(im1)

ny, nx = im1.shape
coords1 = np.zeros((2, ny, nx), np.float32)
coords2 = np.zeros((ny, nx), np.float32), np.zeros((ny, nx), np.float32)

@numba.jit
def apply_coords(coords, coords1, coords2):
    for y in range(coords1.shape[0]):
        for x in range(coords1.shape[1]):
            coords[0, y, x] = y + 10 * np.sin(x*0.01)
            coords[1, y, x] = x + 10 * np.sin(y*0.1)
            coords2[y, x] = y + 10 * np.sin(x*0.01)
            coords1[y, x] = x + 10 * np.sin(y*0.1)

apply_coords(coords1, coords2[0], coords2[1])

N = 100
order = 3

def timeit(title, func, *args, **kwargs):
    # Run once, allow warmup
    res = func(*args, **kwargs)
    # Prepare timer
    t0 = perf_counter()
    te = t0 + 0.5
    count = 0
    # Run
    while perf_counter() < te:
        func(*args, **kwargs)
        count += 1
    # Process
    tr = perf_counter() - t0
    if tr < 1:
        print(title, 'took %1.1f ms (%i loops)' % (1000 * tr / count, count))
    else:
        print(title, 'took %1.3f s (%i loops)' % (tr / count, count))
    return res

if not pirt:
    im2 = timeit('skimage warp', skimage_warp, im1, coords1, order=order)

if pirt and hasattr(pirt.interp, 'interp'):
    im3 = timeit('old pirt', pirt.interp.interp, im1, coords2, order=order)

if sys.version_info > (3, 5):
    im4 = timeit('new pirt warp', pirt.interp.warp, im1, coords2, order)


vv.figure(1); vv.clf()
vv.subplot(221); vv.imshow(im1)
vv.subplot(222); vv.imshow(im2)
vv.subplot(223); vv.imshow(im3)
vv.subplot(224); vv.imshow(im4)

vv.use().Run()
