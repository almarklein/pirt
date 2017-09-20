"""
Illustrate 2D slicing a 3D volume.

The point given to SliceInVolume can also be a 3-element tuple or a visvis.Point.
"""

import imageio
import visvis as vv
import pirt

from pirt import PointSet


# Load volume and get z position of slice 100
vol = imageio.volread('imageio:stent.npz')
z100 = 100

# Get three slices representations. The latter two relative to the first,
# at the same slice, but oriented differently
slice1 = pirt.SliceInVolume(PointSet((64,64,100)))
slice2 = pirt.SliceInVolume(PointSet((66,65,106)), previous=slice1)
slice3 = pirt.SliceInVolume(PointSet((68,67,106)), previous=slice1)

# Show the slices they represent, plus the raw slice at z=100
fig = vv.figure(1); vv.clf()
vv.subplot(221); vv.imshow(vol[z100,:,:])
vv.subplot(222); vv.imshow(slice1.get_slice(vol, 128, 0.5))
vv.subplot(223); vv.imshow(slice2.get_slice(vol, 128, 0.5))
vv.subplot(224); vv.imshow(slice3.get_slice(vol, 128, 0.5))

vv.use().Run()
