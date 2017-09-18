"""
Demo app to deform an image by hand. Hold shift and click-n-drag.
"""

from pirt.apps.deform_by_hand import DeformByHand
import visvis as vv
import imageio

im = imageio.imread('imageio:astronaut.png')[:,:,2].astype('float32')
d = DeformByHand(im, grid_sampling=40)

vv.use().Run()
