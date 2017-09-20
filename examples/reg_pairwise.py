"""
Example demonstrating registering an image to another.

Applicable using different registration algorithms, e.g. Gravity or Demons.
This can also be used to play with the registration parameters.
"""

import pirt
import visvis as vv
import numpy as np

# Get image
im1 = vv.imread('astronaut.png')[:,:,1].astype('float32')
im1 = pirt.diffuse(im1, 1)[::2,::2] / 255.0 

# Deform image
scale = im1.shape[0] * 0.4
rd = pirt.create_random_deformation(im1, 40, 40, mapping='backward', seed=1001)
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
# reg = pirt.DiffeomorphicDemonsRegistration(im1, im2)
# reg = pirt.ElastixRegistration(im1, im2)

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

# Register, pass the figure so that the algorithm can show progress
reg.register(1, fig)

# Visualize end results
vv.figure(2); vv.clf()
reg.show_result('diff', vv.figure(2))

# Get tge found deform and the error compared to the known deform
deform = reg.get_final_deform(0, 1,'backward')
refErr = np.abs(im1 - im2)
err = np.abs(deform.apply_deformation(im1) - im2)

# Analyse errors in registration
parts = []
for def1, def2 in zip(rd, deform):
    parts.append(def1-def2)
D = (parts[0]**2 + parts[1]**2)**0.5            
#
print('error', err.mean()/refErr.mean(), D.mean())
vv.imshow(D)

vv.use().Run()
