"""
This illustrates that a multiscale approach is required to
create a B-spline grid from a discrete field.
"""

import numpy as np
import visvis as vv
from pirt import SplineGrid


# Read image
im = vv.imread('astronaut.png').astype(np.float32)

# Show
vv.figure(1); vv.clf()
vv.subplot(131); vv.imshow(im)

# Make three grids
spacing = 5
ims = []
imsd = []
for i in range(2):
    if i==0:
        fromField = SplineGrid.from_field_multiscale
    else:
        fromField = SplineGrid.from_field
    
    # Create grid
    grid1 = fromField(im[:,:,0], spacing)
    grid2 = fromField(im[:,:,1], spacing)
    grid3 = fromField(im[:,:,2], spacing)
    
    # Obtain interpolated image
    imi = np.zeros_like(im)
    imi[:,:,0] = grid1.get_field()
    imi[:,:,1] = grid2.get_field()
    imi[:,:,2] = grid3.get_field()
    
    ims.append(imi)
    imsd.append( abs(imi-im) )

diff = abs( ims[0]-ims[1] )

# Show
vv.figure(1); vv.clf()
#vv.subplot(221); vv.imshow(im)
for i in range(2):
    vv.subplot(2,2,i+1); vv.imshow(ims[i])
    vv.title('Multiscale' if i==0 else 'Single scale')
    vv.subplot(2,2,i+3); vv.imshow(imsd[i])    

for i in range(4):
    a = vv.subplot(2,2,i+1)
    t = a.FindObjects(vv.Texture2D)
    t[0].SetClim(0, 255)
