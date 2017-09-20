"""
Example illustrating groupwise registration and interpolating between deformations
to make smooth animations.

Applicable to different registration algorithms.
"""

## Init

from functools import reduce

import pirt
import visvis as vv
import numpy as np

# Create images
ims = []
circLocs = [(40,40), (40,60), (60,60), (60,40)]
radius = 5.0
for i in range(4):
    # Create empy image
    im = np.random.normal(0.0, 0.1, (101,101)).astype('float32')
    # Create circle
    circLoc = circLocs[i]
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            #if (y-circLoc[0])**2 + (x-circLoc[1])**2 < radius**2: # circles
            if abs(y-circLoc[0]) + abs(x-circLoc[1]) < radius*1.1: # diamonds
                im[y,x] += 1.0
    # Add
    ims.append(im)

INDEXMAP = {0:1, 1:2, 2:4, 3:3} # map index to subplot location

# Show images
fig = vv.figure(1); vv.clf()
fig.position = 200,200, 500,500
for i in range(4):
    j = INDEXMAP[i] # map index to subplot location
    a = vv.subplot(2,2,j)
    vv.imshow(ims[i])
    a.axis.visible = False
# vv.screenshot('c:/almar/projects/fourBlocks_initial.jpg', vv.gcf(), sf=2, bg='w')

## Register groupwise

# Init figure for registration
fig = vv.figure(2); vv.clf()
fig.position = 200,100, 900, 500

# Apply registration
reg = pirt.GravityRegistration(*ims)
# reg = pirt.DiffeomorphicDemonsRegistration(*ims)
# reg = pirt.ElastixGroupwiseRegistration(*ims)

if isinstance(reg, pirt.GravityRegistration):
    reg.params.mapping = 'backward'
    reg.params.frozenedge = True
    reg.params.final_grid_sampling = 20
    reg.params.speed_factor = 2.0
elif isinstance(reg, pirt.DiffeomorphicDemonsRegistration):
    reg.params.mapping = 'backward'
    reg.params.scale_levels = 5
    reg.params.final_grid_sampling = 20
    reg.params.scale_sampling = 20
    reg.params.final_scale = 1
    reg.params.speed_factor = 1
    reg.params.noise_factor = 0.01
elif isinstance(reg, pirt.ElastixGroupwiseRegistration):
    reg.params.NumberOfResolutions = 5
    reg.params.MaximumNumberOfIterations = 150
    reg.params.FinalGridSpacingInPhysicalUnits = 20
    reg.params.NumberOfSpatialSamples = 2**11
    
#
reg.register(1, fig)


## Get combined at first
fig = vv.figure(3); vv.clf()
fig.position = 200,200, 500,500
im = ims[0].copy()
showGrid = True
for i in range(1,4):
    deform = reg.get_final_deform(i, 0)
    imd = deform.apply_deformation(ims[i])
    im += imd
    if showGrid:
        grid = pirt.reg.reg_base.create_grid_image(im, None, 10, 10)
        imd = deform.apply_deformation(grid)
    j = INDEXMAP[i] # map index to subplot location
    a = vv.subplot(2,2,j); vv.imshow(imd)
    a.axis.visible = False
a = vv.subplot(221); vv.imshow(im)
a.axis.visible = False
# vv.screenshot('c:/almar/projects/fourBlocks_mapto1.jpg', vv.gcf(), sf=2, bg='w')


## Get 12 phases

# Prepare
ims2 = []
nsamples = 10
N = len(ims)
spline_type = -0.25

# Get deformations from reference frame to individuals
# They need to be in forward mapping, so we can combine them properly
inverseDeforms = []
for i in range(N):
    d = reg.get_deform(i).as_forward_inverse() # for free if we used bw mapping
#     d = reg.get_deform(i).as_backward_inverse() # for free if we used bw mapping
    inverseDeforms.append(d)

# Create new images
for i in range(N): # for all images
    
    # Get deformation that map from reference frame to 4 nearest points
    # (Cheap operation, no copying or calculations yet)
    fourDeforms = []
    kk = []
    for k in range(i-1,i+3): 
        if k<0: k += N
        if k>=N: k -= N 
        fourDeforms.append( inverseDeforms[k] )
        kk.append(k)
    
    for j in range(nsamples): # sample points for each space between two images
        
        # Combine to get deformation from reference frame to point of interest
        t = float(j) / nsamples
        cc = pirt.get_cubic_spline_coefs(t, spline_type)
        theDeform = pirt.DeformationIdentity()
        for k in range(4): 
            theDeform += fourDeforms[k].scale(cc[k])
        
        # Map all images there
        # deform2 = theDeform.as_forward() # forward introduces smoothing!
        deform2 = theDeform.as_backward()
        combined_im = None
        for k in range(N):
            # deform1 = reg.get_deform(k).as_forward()
            deform1 = reg.get_deform(k).as_backward()
            im = deform1.compose(deform2).apply_deformation(ims[k])
            if combined_im is None:
                combined_im = im
            else:
                combined_im += im
        combined_im *= 1.0/N
        
        # Add
        ims2.append(combined_im)


# Determine center of mass of all images
def centerOfMass(im):
    im = im.copy()
    th = im.mean() + im.std()*1.5
    im[im<th] = 0
    mx, my = pirt.meshgrid(im)
    w = 1.0/im.sum()
    return (im*mx).sum()*w, (im*my).sum()*w
com = pirt.Pointset(2)
for i in range(len(ims2)):
    com.append(centerOfMass(ims2[i]))
com.append(com[0])

# Combine a couple
im_shadows = np.zeros((101,101,3), 'float32')
Nshadows = 8
offset = 0
color_i = 0
for i in range(Nshadows):
    j = int(i * len(ims2)/Nshadows)
    im_shadows[:,:,color_i] += ims2[j+offset]
    color_i += 1
    if color_i > 1:
        color_i = 0
    
    
# Show images
vv.figure(4); vv.clf()
vv.movieShow(ims2, 0.5)
vv.plot(com, ls='-', ms='', lw=2, mw=7, lc='k', mc='c')

# Show sequence still
fig = vv.figure(5); vv.clf(); 
fig.position = 200,200, 500,500
vv.imshow(im_shadows)
vv.plot(com, ls='-', ms='', lw=2, mw=7, lc='k', mc='c')
a = vv.gca(); a.axis.visible = False
# vv.screenshot('c:/almar/projects/fourBlocks_maptoN.jpg', vv.gca(), sf=2, bg='w')
# vv.movieWrite('c:/almar/projects/fourBlocks_maptoN.swf', ims2)


## Average full deforms to create atlas image
# Demonstrates that the paper of Seghers et al. makes a mistake

ims_atlas_forward = []
ims_atlas_backward = []

def sum(L):
    return reduce(lambda x,y:x+y, L)

# Map all images to reference frame by averging all deforms to the other ims
for i in range(N):
    
    # Get deforms of this image to all the others
    fullDeforms_forward = []
    fullDeforms_backward = []
    for j in range(N):
        fullDeforms_forward.append( reg.get_final_deform(i,j,'forward') )
        fullDeforms_backward.append( reg.get_final_deform(i,j,'backward') )
    
    # Take average deform
    avgDeform_forward = sum(fullDeforms_forward).scale(1.0/N).as_backward()
    avgDeform_backward = sum(fullDeforms_backward).scale(1.0/N).as_backward()
    ims_atlas_forward.append( avgDeform_forward.apply_deformation(ims[i]) )
    ims_atlas_backward.append( avgDeform_backward.apply_deformation(ims[i]) )

# Take average
im_atlas_forward = sum(ims_atlas_forward) * (1.0/N)
im_atlas_backward = sum(ims_atlas_backward) * (1.0/N)

# Show
fig = vv.figure(6); vv.clf()
fig.position = 200,200, 1000,500
a=vv.subplot(121); a.axis.visible=False; vv.imshow(im_atlas_forward)
a=vv.subplot(122); a.axis.visible=False; vv.imshow(im_atlas_backward)
# vv.screenshot('c:/almar/projects/fourBlocks_maptoMean_.jpg', vv.gca(), sf=2, bg='w')

vv.use().Run()
