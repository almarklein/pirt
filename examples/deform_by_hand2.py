"""
Example that lets the user apply a deformation by hand, then use that 
deformation to map and reconstruct the image.

Note: the first applied deformation will take a while as all numba functions
are compilerd (a.k.a. JIT warmup).
"""

import visvis as vv
import imageio

from pirt.apps.deform_by_hand import DeformByHand


# Get image
im1 = imageio.imread('imageio:astronaut.png')[:,:,1].astype('float32') / 255

# Let user define a deform - close the figure to proceed
d = DeformByHand(im1, 30)
d.run()

# Deform the image
im2 = d.field.apply_deformation(im1)

# imageio.imwrite('~/astronaut_distorted.png', im2)


##

f = d.field
vv.figure(1); vv.clf()

im2 = f.as_forward().apply_deformation(im1)
im3 = f.as_forward_inverse().apply_deformation(im2)
im5 = f.as_backward().apply_deformation(im1)
im6 = f.as_backward_inverse().apply_deformation(im5)

a1 = vv.subplot(231); vv.imshow(im1); vv.title('Original')
a2 = vv.subplot(232); vv.imshow(im2); vv.title('Deformed forward')
a3 = vv.subplot(233); vv.imshow(im3); vv.title('Deformed forward and back')
a5 = vv.subplot(235); vv.imshow(im5); vv.title('Deformed backward')
a6 = vv.subplot(236); vv.imshow(im6); vv.title('Deformed backward and back')

for a in [a1, a2, a3, a5, a6]:
    a.axis.visible = False

vv.use().Run()
