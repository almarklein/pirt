"""
Example with a black image containing a white bock, which is deformed
to move the block down, then right, and then back, using two deforms,
which are composed and inverted.
"""

import numpy as np
import visvis as vv

from pirt import Aarray, PointSet, DeformationFieldBackward

# Parameters, play with these!
INJECTIVE = True
FREEZE_EDGES = True


# Create test image with one block in the corner
im0 = np.zeros((100, 100), np.float32)
im0 = Aarray(im0, (0.66, 1.0))
im0[30:40, 40:50] = 1.0

# Draw a grid on it
grid_step = 10
im0[1::grid_step,:] = 0.3
im0[:,1::grid_step] = 0.3

# Define three locations, to move the block between
pp = PointSet(2)
pp.append(45, 35)
pp.append(45, 85)
pp.append(85, 85)
pp = pp * PointSet(reversed(im0.sampling))

# Get down-deformation and right-deformation
from_points_multiscale = DeformationFieldBackward.from_points_multiscale

deform1 = from_points_multiscale(im0, 4, pp[0:1], pp[1:2],
                                 injective=INJECTIVE, frozenedge=FREEZE_EDGES)

deform2 = from_points_multiscale(im0, 4, pp[1:2], pp[2:3],
                                 injective=INJECTIVE, frozenedge=FREEZE_EDGES)

# Combine the two, by composition, not by addition!
deform3 = deform1.compose(deform2)

# Move the block to the lower right ...

# Apply downward deform, then right deform
im1 = deform1.apply_deformation(im0)
im2 = deform2.apply_deformation(im1)

# And also in one go
im3 = deform3.apply_deformation(im0)

# Let's move it back ...

# The naive way
im0_1 = deform1.inverse().apply_deformation( deform2.inverse().apply_deformation(im2) )

# Create a deform and apply that
deform4 = deform2.inverse().compose(deform1.inverse())
im0_2 = deform4.apply_deformation(im2)

# Or inverse the single deform that we already had
im0_3 = deform3.inverse().apply_deformation(im2)

# Visualize
vv.figure(1)
vv.clf()

vv.subplot(321); vv.imshow(im0); vv.title('Original')
vv.subplot(322); vv.imshow(im1); vv.title('Moved down')
vv.subplot(323); vv.imshow(im2); vv.title('Moved down-image right')
vv.subplot(324); vv.imshow(im3); vv.title('Moved in one pass')
vv.subplot(325); vv.imshow(im0_1); vv.title('Moved back 1')
vv.subplot(326); vv.imshow(im0_2); vv.title('Moved back 2')
