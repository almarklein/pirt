import sys, os, time
import visvis as vv
import numpy as np
import pirt
from visvis import ssdf
import scipy.ndimage

# Define home directory
homeDir = '/home/almar/'
if sys.platform.startswith('win'):
    homeDir = 'c:/almar/'

# Init ims
ims = []

# Add lena
im1 = vv.imread(homeDir+'data/images/lena_distorted00.png')
im1 = im1[::2,::2].astype(np.float32)
ims.append(im1)

# Add MRI
s = ssdf.load(homeDir+'projects/brainwebExample.bsdf')
ims.append(s.im1)

# Add sparse
s = ssdf.load(os.path.join(homeDir, 'projects/py/pirt/data/reg2D_simdata.ssdf'))
im1 = pirt.diffuse(s.im1,0.5)
im1 = im1 + np.random.normal(0.0, 0.1, im1.shape)
ims.append(im1)


def normalize(mass):
    mass *= (1/mass.std()) # i.e. mass = mass / (2*std)
    mass += (-0.5-mass.mean()) # i.e. move mean to 1.0
    return mass

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
    reg._soft_limit(mass, 1) # todo: half limit
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
    