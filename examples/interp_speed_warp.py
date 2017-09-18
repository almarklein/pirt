"""
Example that demonstrates that Pirt's warp() function is over 4x faster than skimage warp().
"""

from time import perf_counter

import imageio
import numpy as np
import visvis as vv

from pirt import warp as warp_pirt
from skimage.transform import warp as warp_skimage

# Read an image to test on
im1 = imageio.imread('imageio:chelsea.png')[:300, :300, 2].astype('float32') / 255
im2 = np.zeros_like(im1)
im3 = np.zeros_like(im1)

# Produce coordinates with a wave-like morph
ny, nx = im1.shape
coords = np.zeros((2, ny, nx), np.float32)
for y in range(coords.shape[1]):
    for x in range(coords.shape[2]):
        coords[0, y, x] = y + 10 * np.sin(x*0.01)
        coords[1, y, x] = x + 10 * np.sin(y*0.1)


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


# For nearest, linear, and cubic, do a warp and time it ...
for order in (0, 1, 3):
    print('Order %i:' % order)
    im2 = timeit('  skimage', warp_skimage, im1, coords, order=order)
    im3 = timeit('     pirt', warp_pirt, im1, coords, order=order)

    vv.figure(order+1); vv.clf()
    a1 = vv.subplot(221); vv.imshow(im1)
    a2 = vv.subplot(223); vv.imshow(im2)
    a3 = vv.subplot(224); vv.imshow(im3)
    a1.camera = a2.camera = a3.camera

vv.use().Run()
