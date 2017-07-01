""" Example how to deform an image. """
import visvis as vv
import pirt
from pirt.apps.deform_by_hand import DeformByHand


im1 = vv.imread('lena.png')[:,:,1].astype('float32') / 255

d = DeformByHand(im1, 30)
d.run()
im2 = d.field.apply_deformation(im1)

# vv.imwrite('C:/almar/data/images/lena_distorted04.png', im2)


##

f = d._field2
vv.figure(1); vv.clf()

im2 = f.as_forward().apply_deformation(im1)
im3 = f.as_forward_inverse().apply_deformation(im2)
im5 = f.as_backward().apply_deformation(im1)
im6 = f.as_backward_inverse().apply_deformation(im5)

a1 = vv.subplot(231); vv.imshow(im1)
a2 = vv.subplot(232); vv.imshow(im2)
a3 = vv.subplot(233); vv.imshow(im3)
a5 = vv.subplot(235); vv.imshow(im5)
a6 = vv.subplot(236); vv.imshow(im6)

for a in [a1, a2, a3, a5, a6]:
    a.axis.visible = False

vv.use().Run()
