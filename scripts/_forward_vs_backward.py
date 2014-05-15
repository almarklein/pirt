
import visvis as vv
import numpy as np
import pirt

# Create image
radius = 5.0
im = 0.0*np.random.normal(0.0, 0.1, (101,101)).astype('float32')
# Create square
circLoc = (50,50)
for y in range(im.shape[0]):
    for x in range(im.shape[1]):
        #if (y-circLoc[0])**2 + (x-circLoc[1])**2 < radius**2: # circles
        if abs(y-circLoc[0]) + abs(x-circLoc[1]) < radius*1.1: # diamonds
            im[y,x] += 1.0
# Create more squares
#circLocs = [(30,30), (30,70), (70,70), (70,30)]
circLocs = [(30,50), (50,70), (70,50), (50,30)]
radius = 2.0
for i in range(4):
    # Create circle
    circLoc = circLocs[i]
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            #if (y-circLoc[0])**2 + (x-circLoc[1])**2 < radius**2: # circles
            if abs(y-circLoc[0]) + abs(x-circLoc[1]) < radius*1.1: # diamonds
                im[y,x] += 1.0


def patch_image(im, patch, pos):
    """ patch_image(im, patch, pos)
    
    Modifies the given im in-place by putting the given patch at the 
    given (pixel) position).
    
    """
    
    # Get tail
    tail = int(np.ceil( patch.shape[0]/2 ))
    
    # Get upper right and lower left
    pos1, pos2 = [], []
    for d in range(im.ndim):
        pos1.append( pos[d] - tail )
        pos2.append( pos[d] + tail )
    
    # Get patch indices
    pos3, pos4 = [], []
    for d in range(im.ndim):
        pos3.append( 0 )
        pos4.append( tail*2 )
    
    # Correct indices
    for d in range(im.ndim):
        if pos1[d] < 0:
            pos3[d] = -pos1[d]
            pos1[d] = 0
        if pos2[d] >= im.shape[d]:
            pos4[d] = im.shape[d] - pos2[d] - 2
            pos2[d] = im.shape[d] - 1
    
    # Build slice objects
    slices_im = []
    slices_patch = []
    for d in range(im.ndim):
        slices_im.append( slice(pos1[d],pos2[d]+1) )
        slices_patch.append( slice(pos3[d],pos4[d]+1) )
    
    # Put patch in 
    im[tuple(slices_im)] += patch[tuple(slices_patch)]


def arrows(points, vectors, head=(0.2, 1.0), **kwargs):
    
    
    if 'ls' in kwargs:
        raise ValueError('Cannot set line style for arrows.')
    
    
    ppd = vv.Pointset(pp.ndim)
    for i in range(len(points)):
        
        p1 = points[i]      # source point 
        v1 = vectors[i]     # THE vector
        v1_norm = v1.norm()
        p2 = p1 + v1        # destination point        
        if v1_norm:
            pn = v1.normal() * v1_norm * abs(head[0])  # normal vector
        else:
            pn = vv.Point(0,0)
        ph1 = p1 + v1 * head[1]
        ph2 = ph1 - v1 * head[0]
        
        # Add stick
        ppd.append(p1); ppd.append(p2)
        # Add arrowhead
        ppd.append(ph1); ppd.append(ph2+pn);
        ppd.append(ph1); ppd.append(ph2-pn);
    
    return vv.plot(ppd, ls='+', **kwargs)
    
    

# Create two deformations in forward mode
sigma = 11
amplitude = 40
deforms = []
for i in range(2):
    deform = []
    for d in range(2):
        a = np.zeros_like(im)
        if i==0:
            if d==0:            
                amplitude_ = amplitude * 1
            else:
                amplitude_ = amplitude * -0.1
        else:
            if d==0:            
                amplitude_ = amplitude * -0.1
            else:
                amplitude_ = amplitude * 1
        k = pirt.gaussfun.gaussiankernel2(sigma,0,0, N=-6*sigma)
        k *= amplitude_ * (1.0/k.max())
        patch_image(a, k, (51, 51))
        deform.append(a)
    deforms.append(deform)
deforms_forward = [pirt.DeformationFieldForward(deform) for deform in deforms]

# Create two deformations in backwards mode
deforms_backward = [deform.as_backward() for deform in deforms_forward]

# Deform the image with the average of the two deforms
deform_forward = deforms_forward[0].add(deforms_forward[1]).scale(0.5)
im_forward = deform_forward.as_backward().apply_deformation(im)
#
deform_backward = deforms_backward[0].add(deforms_backward[1]).scale(0.5)
im_backward = deform_backward.apply_deformation(im)

# Create averages for first deform
im_forward1 =  deforms_forward[1].as_backward().apply_deformation(im)
im_forward2 =  deforms_forward[1].scale(0.5).as_backward().apply_deformation(im)
#
im_backward1 =  deforms_backward[1].apply_deformation(im)
im_backward2 =  deforms_backward[1].scale(0.5).apply_deformation(im)


# Calculate vectors
vecstep = 20
pp = vv.Pointset(2)
vvf, vvb = vv.Pointset(2), vv.Pointset(2)
vvf1, vvb1 = vv.Pointset(2), vv.Pointset(2)
vvf2, vvb2 = vv.Pointset(2), vv.Pointset(2)
for y in np.arange(0, im.shape[0], vecstep):
    for x in np.arange(0, im.shape[1], vecstep):
        # Center point
        p1 = vv.Point(x,y)
        pp.append(p1)
        # Vector for forward
        deform = deform_forward
        vvf.append( vv.Point(deform[1][y,x], deform[0][y,x]) )
        deform = deforms_forward[0]
        vvf1.append( vv.Point(deform[1][y,x], deform[0][y,x]) )
        deform = deforms_forward[1]
        vvf2.append( vv.Point(deform[1][y,x], deform[0][y,x]) )
        # Vector for backward
        deform = deform_backward
        vvb.append( vv.Point(deform[1][y,x], deform[0][y,x]) )
        deform = deforms_backward[0]
        vvb1.append( vv.Point(deform[1][y,x], deform[0][y,x]) )
        deform = deforms_backward[1]
        vvb2.append( vv.Point(deform[1][y,x], deform[0][y,x]) )



# Show

if True:
    f = vv.figure(1); vv.clf()
    f.position = 120,100, 1150, 290
    clim = (0.0,1.5)
    cmap = (0,0,0), (0.5, 1.0, 1.0)
    veccolor = 'w'
    #
    a1 = vv.subplot(141); vv.imshow(im, clim=clim, cm=cmap)
    arrows(pp, vvf1, (0.2, 0.9), lc=veccolor, lw=2, axesAdjust=False)
    arrows(pp, vvf2, (0.2, 0.9), lc=veccolor, lw=2, axesAdjust=False)
    #
    a2 = vv.subplot(142); vv.imshow(im, clim=clim, cm=cmap)
    arrows(pp, vvb1, (-0.2, 0.7), lc=veccolor, lw=2, axesAdjust=False)
    arrows(pp, vvb2, (-0.2, 0.7), lc=veccolor, lw=2, axesAdjust=False)
    #
    a3 = vv.subplot(143); vv.imshow(im_forward, clim=clim, cm=cmap)    
    arrows(pp, vvf, (0.2, 0.9), lc=veccolor, lw=2, axesAdjust=False)
    #
    a4 = vv.subplot(144); vv.imshow(im_backward, clim=clim, cm=cmap)
    arrows(pp, -1*vvb, (0.2, 0.3), lc=veccolor, lw=2, axesAdjust=False)
    #
    #
    for a in [a1, a2, a3, a4]:
        a.axis.visible = False
    
if False:
    
    f = vv.figure(2); vv.clf()
    f.position = 150,550, 1100, 365
    clim = (0.0,1.5)
    cmap = (0,0,0), (0.5, 1.0, 1.0)
    veccolor = 'w'
    #
    a1 = vv.subplot(141); vv.imshow(im, clim=clim, cm=cmap)
    #
    a2 = vv.subplot(142); vv.imshow(im_backward1, clim=clim, cm=cmap)
    arrows(pp, vvf1, (0.2, 0.5), lc=veccolor, lw=2, axesAdjust=False)
    #
    a3 = vv.subplot(143); vv.imshow(im_forward2, clim=clim, cm=cmap)
    arrows(pp, vvf2, (0.2, 0.9), lc=veccolor, lw=2, axesAdjust=False)
    #
    a4 = vv.subplot(144); vv.imshow(im_backward2, clim=clim, cm=cmap)
    arrows(pp, -1*vvb2, (0.2, 0.3), lc=veccolor, lw=2, axesAdjust=False)
    #
    for a in [a1, a2, a3, a4]:
        a.axis.visible = False

if 1:
    vv.figure(2); vv.clf()
    #im_, deforms = im_forward, deforms_forward
    im_, deforms = im_backward, deforms_backward
    clim = -amplitude, amplitude
    vv.subplot(321); vv.imshow(im)
    vv.subplot(322); vv.imshow(im_)
    vv.subplot(323); vv.imshow(deforms[0][0], clim=clim)
    vv.subplot(324); vv.imshow(deforms[0][1], clim=clim)
    vv.subplot(325); vv.imshow(deforms[1][0], clim=clim)
    vv.subplot(326); vv.imshow(deforms[1][1], clim=clim)

if False:
    vv.screenshot('c:/almar/projects/forwardVSbackward1.jpg', a1, sf=2)
    vv.screenshot('c:/almar/projects/forwardVSbackward2.jpg', a2, sf=2)
    vv.screenshot('c:/almar/projects/forwardVSbackward3.jpg', a3, sf=2)
    vv.screenshot('c:/almar/projects/forwardVSbackward4.jpg', a4, sf=2)