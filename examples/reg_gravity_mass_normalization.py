"""
Illustrate the kind of mass images that the Gravity registration
algororithm makes to base the image forces on. Normalization of these
images is an important part of making Gravity robust.
"""

import visvis as vv
import numpy as np
import scipy.ndimage
import imageio
import pirt


# Init ims
ims = []

# Add astronaut
ims.append(imageio.imread('imageio:astronaut.png')[:,:,1].astype('float32'))

# Add slice from stent image
ims.append(imageio.volread('imageio:stent.npz')[:,90,:].astype('float32'))

# Add a photo
ims.append(imageio.imread('imageio:wikkie.png')[:,:,1].astype('float32'))


def normalize(mass):
    mass *= (1/mass.std()) # i.e. mass = mass / (2*std)
    mass += (-0.5-mass.mean()) # i.e. move mean to 1.0
    return mass


def soft_limit(data, limit):
    if limit == 1:
        data[:] = 1.0 - np.exp(-data)
    else:
        f = np.exp(-data/limit)
        data[:] = -limit * (f-1)


def getMass(im, truncate=True):
    
    # Smooth
    im = pirt.diffuse(im, 1)
    
    # Gradient magnitude
    massParts = []
    for d in range(im.ndim):
        k = np.array([0.5, 0, -0.5], dtype='float64')
        tmp = scipy.ndimage.convolve1d(im, k, d, mode='constant')
        massParts.append(tmp**2)    
    # Sum and take square root
    mass = np.add(*massParts)**0.5
    
    # Normalize
    mass = normalize(mass)
    
    # Truncate
    if truncate:
        mass[mass<0] = 0.0
    soft_limit(mass, 1) # todo: half limit
    #mass = mass**0.5
    
    return mass


# Show
vv.figure(10); vv.clf()
for i in range(3):
    a = vv.subplot(3,3,i+1)
    vv.imshow(ims[i])
    a.axis.visible = False

for i in range(3):
    a = vv.subplot(3,3,i+4)
    vv.imshow(getMass(ims[i]))
    a.axis.visible = False

for i in range(3):
    a = vv.subplot(3,3,i+7)
    vv.hist(getMass(ims[i], False), 64)    
    a.SetLimits()
    