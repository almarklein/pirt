""" Script

This script file shows two small demo applications
for B-spline grids.

At the bottom of this file, several tests and other small
demos can be found.

"""

import numpy as np
import visvis as vv
from pirt import SplineGrid, DeformationGridForward
from pirt import Point, Pointset, Aarray
from pirt import SplineByHand, DeformByHand
    

if __name__ == '__main__':
    import visvis as vv
    
    # Show 1D spline demo
    sbh = SplineByHand()
    
    # Show 2D deformation demo
    im = vv.imread('lena.png')[::,::,1].astype(np.float32)
    im = Aarray(im[::2,:], (2,1))
    dbh = DeformByHand(im)

if 0:
    pass

## Example deform using pixel forces
    
    # Make grid image
    im = np.zeros((100,100), dtype=np.float32)
    im[::10,:] = 1.0
    im[:,::10] = 1.0
    
    # Make deform image
    fx = np.zeros_like(im)
    fy = np.zeros_like(im)
    fa = np.zeros_like(im)
    fx[30,30] = 10
    fy[25,20] = 20
    fa[30,30] = 5
    fa[25,20] = 5
    
    # Make grid and deform
    grid = DeformationGridForward(im, 15)
    grid._set_using_field([fy, fx], fa)
    im2 = grid.apply_deformation(im, 3)
    
    fig = vv.figure(11); vv.clf()
    fig.position = 73.00, 67.00,  560.00, 794.00
    vv.subplot(211); vv.imshow(im)
    vv.subplot(212); vv.imshow(im2)
    
    
## Example Representing an image with a Bspline grid
    
    # Read image
    im = vv.imread('lena.png').astype(np.float32)
    im_r = im[:,:,0]
    
    # New sparse image and interpolated image
    ims = np.zeros_like(im)
    imi = np.zeros_like(im)
    
    # Select points from the image
    pp = Pointset(2)
    R, G, B = [], [], []
    for i in range(10000):
        y = np.random.randint(0, im.shape[0])
        x = np.random.randint(0, im.shape[1])
        pp.append(x,y)
        R.append(im[y,x,0])
        G.append(im[y,x,1])
        B.append(im[y,x,2])
        ims[y,x] = im[y,x]
    
    # Make three grids
    spacing = 10
    grid1 = SplineGrid(im_r, spacing, spline_type='B')
    grid2 = SplineGrid(im_r, spacing, spline_type='B')
    grid3 = SplineGrid(im_r, spacing, spline_type='B')
    
    # Put data in
    grid1._set_using_points(pp, R)
    grid2._set_using_points(pp, G)
    grid3._set_using_points(pp, B)
    
    # Obtain interpolated image
    imi[:,:,0] = grid1.get_field()
    imi[:,:,1] = grid2.get_field()
    imi[:,:,2] = grid3.get_field()
    
    # Show
    vv.figure(1); vv.clf()
    vv.subplot(131); vv.imshow(im)
    vv.subplot(132); vv.imshow(ims)
    vv.subplot(133); vv.imshow(imi)


## Creating a B-spline grid from all the pixels in the image
# This illustrates that a multiscale approach is  required to
# create a B-spline grid from a discrete field.

    # Read image
    im = vv.imread('lena.png').astype(np.float32)
    
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
        grid1 = fromField(im[:,:,0], spacing, spline_type='B')
        grid2 = fromField(im[:,:,1], spacing, spline_type='B')
        grid3 = fromField(im[:,:,2], spacing, spline_type='B')
        
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
        vv.subplot(2,2,i+3); vv.imshow(imsd[i])    
    
    for i in range(4):
        a = vv.subplot(2,2,i+1)
        t = a.FindObjects(vv.Texture2D)
        t[0].SetClim(0, 255)

## Test refinement
    
    import numpy as np
    import visvis as vv
    import pirt
    
    # Read image
    im = vv.imread('lena.png').astype(np.float32)
    im_r = im[:,:,0]
    
    # Select points from the image
    pp = pirt.Pointset(2)
    R, G, B = [], [], []
    for i in range(1000):
        y = np.random.randint(0, im.shape[0])
        x = np.random.randint(0, im.shape[1])
        pp.append(x,y)
        R.append(im[y,x,0])
    
    # Create grid
    g1 = pirt.SplineGrid(im_r, 10, spline_type='B')
    g1._set_using_points(pp, R)
    
    # Refine that grid
    g2 = g1.refine()
    g3 = g2.refine()
    
    # Obtain fields, and errors
    f1 = g1.get_field()
    f2 = g2.get_field()
    f3 = g3.get_field()    
    # 
    e2 = f1-f2
    e3 = f1-f3
    
    # Illustrate
    vv.figure();
    showGrid = True
    a1 = vv.subplot(231); g1.show(a1, showGrid=showGrid)
    a2 = vv.subplot(232); g2.show(a2, showGrid=showGrid)
    a3 = vv.subplot(233); g3.show(a3, showGrid=showGrid)
    a4 = vv.subplot(234); vv.imshow(f1-f2)
    a5 = vv.subplot(235); vv.imshow(f2-f3)
    a6 = vv.subplot(236); vv.imshow(f1-f3)
    for a in [a1, a2, a3, a4, a5, a6]:
        a.axis.visible = False
    print 'Mean error f1-f2: ', np.abs(f1-f2).mean()
    print 'Mean error: f2-f3', np.abs(f2-f3).mean()
    print 'Mean error: f1-f3', np.abs(f1-f3).mean()


## Folding prevention / Anti-shearing
    
    import visvis as vv
    import numpy as np
    
    c1 = np.linspace(-30, 30, 1000)
    
    limit = 0.48 * 10
    if True:
        I = c1<0
        c2 =     limit * (1 - np.exp(-c1/limit) )
        c2[I] = -limit * (1 - np.exp(c1/limit) )
    else:
        f = np.exp(-np.abs(c1)/limit)
        c2 = (f-1)* -np.sign(c1) * limit    
    
    # Calculate derivative
    n = int(len(c2)/2)
    der1 = (c2[n+1] - c2[n]) / (c1[n+1] - c1[n])
    der2 = (c2[n+2] - c2[n]) / (c1[n+2] - c1[n])
    der3 = (c2[n+3] - c2[n]) / (c1[n+3] - c1[n])
    print 'Derivative', der1, der2, der3
    
    # Show
    vv.figure(3); vv.clf()
    a = vv.gca()
    vv.plot(c1, c2)
    a.axis.showGrid = 1
    a.daspectAuto = 0
    a.axis.xLabel = 'knot value'
    a.axis.yLabel = 'new knot value'

