""" Script _injective

This script shows the difference between using Choi's constraint by clipping
and by using a smooth constraint. I was hoping for a significant difference,
but the result seems only sighly better.

"""
import pirt
import visvis as vv
import numpy as np

# Create image
im = np.zeros((512,512), 'float32')

# Create pointsets
pp1 = pirt.PointSet(2)
pp2 = pirt.PointSet(2)
 
if False:
    pp1.append(200,300)
    pp2.append(300,300)
    pp1.append(300,300)
    pp2.append(200,300)
else:
    pp1.append(100,100)
    pp2.append(200,200)
    pp1.append(200,100)
    pp2.append(200,100)
    pp1.append(100,200)
    pp2.append(100,200)
    #
    pp1.append(300,100)
    pp2.append(300,100)
    pp1.append(400,100)
    pp2.append(300,200)
    pp1.append(400,200)
    pp2.append(400,200)
    #
    pp1.append(100,300)
    pp2.append(100,300)
    pp1.append(100,400)
    pp2.append(200,300)
    pp1.append(200,400)
    pp2.append(200,400)
    #
    pp1.append(300,400)
    pp2.append(300,400)
    pp1.append(400,400)
    pp2.append(300,300)
    pp1.append(400,300)
    pp2.append(400,300)

# Deform
def deform_and_show(injective):
    from_points_multiscale = pirt.DeformationFieldForward.from_points_multiscale
    deform = from_points_multiscale(im, 10, pp1, pp2, injective=injective, frozenedge=True)
    deform.show()

# Show
vv.figure(1);
vv.clf()
vv.subplot(131); deform_and_show(0); vv.title('Not injective')
vv.subplot(132); deform_and_show(-0.9); vv.title('Truncated injective')
vv.subplot(133); deform_and_show(+0.9); vv.title('Smooth injective')

