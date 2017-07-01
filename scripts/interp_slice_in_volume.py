"""
Illustrate 2D slicing a 3D volume.
"""

import imageio
import visvis as vv
import pirt
from pirt import Point


# Load volume and get z position of slice 100
vol = imageio.volread('imageio:stent.npz')
z100 = 100

# Get three slices representations. The latter two relative to the first,
# at the same slice, but oriented differently
slice1 = pirt.SliceInVolume(Point(64,64,100))
slice2 = pirt.SliceInVolume(Point(66,65,106), previous=slice1)
slice3 = pirt.SliceInVolume(Point(68,67,106), previous=slice1)

# Show the slices they represent, plus the raw slice at z=100
fig = vv.figure(1); vv.clf()
vv.subplot(221); vv.imshow(vol[z100,:,:])
vv.subplot(222); vv.imshow(slice1.get_slice(vol, 128, 0.5))
vv.subplot(223); vv.imshow(slice2.get_slice(vol, 128, 0.5))
vv.subplot(224); vv.imshow(slice3.get_slice(vol, 128, 0.5))

vv.use().Run()
