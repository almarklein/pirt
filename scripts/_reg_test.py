import sys, os
import pirt
import visvis as vv
import numpy as np

# Get image
if 1:
    im1 = vv.imread('lena.png')[:,:,1].astype('float32')
    im1 = pirt.diffuse(im1, 1)[::2,::2] / 255.0 
    #im1 = im1[::4,::4]
    #im1 = vv.Aarray(im1)#[::2,::]
else:
    # Define home directory
    homeDir = '/home/almar/'
    if sys.platform.startswith('win'):
        homeDir = 'c:/almar/'
    
    # Get image    
    fname = os.path.join(homeDir, 'data/misc/reg2D_simdata.ssdf')
    s = pirt.ssdf.load(fname)
    im1 = pirt.diffuse(s.im1*4, 0.5)

# Deform image
scale = im1.shape[0] * 0.4
rd = pirt.randomDeformations.create_random_deformation(im1,40, 40, mapping='backward',seed=1001)
im2 = rd.apply_deformation(im1)

# Add noise
im1 += np.random.normal(0, 0.1, im1.shape)
im2 += np.random.normal(0, 0.1, im1.shape)

# Get figure
vv.closeAll()
fig = vv.figure(1); vv.clf()
fig.position = 200,100, 900, 500

# Init registration
reg = pirt.GravityRegistration(im1, im2)

if isinstance(reg, pirt.DiffeomorphicDemonsRegistration):
    reg.params.speed_factor = 2
    reg.params.noise_factor = 0.1
    reg.params.mapping = 'backward'
    reg.params.scale_sampling = 16
    reg.params.final_grid_sampling = 35
if isinstance(reg, pirt.GravityRegistration):
    reg.params.scale_levels = 12
    reg.params.scale_sampling = 15
    reg.params.mapping = 'backward'
    reg.params.deform_wise = 'groupwise'
    reg.params.deform_limit = 0.25
    reg.params.final_scale = 1
    reg.params.final_grid_sampling = 20
    reg.params.grid_sampling_factor = 0.5 # !! important especially for Laplace !!
    reg.params.frozenedge = True
    reg.params.mass_transforms = 2
    reg.params.speed_factor = 1.0

# Register
reg.register(1, fig)
vv.figure(2); vv.clf()
reg.show_result('diff', vv.figure(2))


deform = reg.get_final_deform(0,1,'backward')
refErr = np.abs(im1 - im2)
err = np.abs(deform.apply_deformation(im1) - im2)

parts = []
for def1, def2 in zip(rd, deform):
    parts.append(def1-def2)
D = (parts[0]**2 + parts[1]**2)**0.5            
#
print 'error', err.mean()/refErr.mean(), D.mean()
vv.imshow(D)
