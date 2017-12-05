"""
Minor tests for basic functionality of the registration algorithms.
"""

import numpy as np
# import visvis as vv

import pirt


def get_two_images():

    # Create test image
    im0 = np.zeros((100, 100), np.float32)
    im0 = pirt.Aarray(im0, (0.6, 2.0))
    im0[30:40, 40:50] = 1.0
    
    # Create test image
    im1 = np.zeros((100, 100), np.float32)
    im1 = pirt.Aarray(im1, (0.6, 2.0))
    im1[40:50, 44:54] = 1.0
    
    return im0, im1


def diff(im0, im1):
    return np.abs(im0 - im1).sum()


def test_gravity1():
    im0, im1 = get_two_images()
    
    reg = pirt.GravityRegistration(im0, im1)
    reg.params.deform_wise = 'pairwise'
    reg.register()
    
    im0_1 = reg.get_final_deform(0, 1).apply_deformation(im0)
    im1_0 = reg.get_final_deform(1, 0).apply_deformation(im1)
    
    # vv.figure(1); vv.clf()
    # vv.subplot(221); vv.imshow(im0)
    # vv.subplot(222); vv.imshow(im1)
    # vv.subplot(223); vv.imshow(im1_0 - im0)
    # vv.subplot(224); vv.imshow(im0_1 - im1)
    
    print('gravity1', diff(im0_1, im1))
    assert diff(im0, im1) > 100
    assert diff(im0_1, im1) < 15
    assert diff(im1_0, im0) < 15


def test_gravity2():
    im0, im1 = get_two_images()
    
    reg = pirt.GravityRegistration(im0, im1)
    reg.params.deform_wise = 'groupwise'
    reg.register()
    
    im0_1 = reg.get_final_deform(0, 1).apply_deformation(im0)
    im1_0 = reg.get_final_deform(1, 0).apply_deformation(im1)
    
    print('gravity2', diff(im0_1, im1))
    assert diff(im0, im1) > 100
    assert diff(im0_1, im1) < 15
    assert diff(im1_0, im0) < 15


def test_demons1():
    im0, im1 = get_two_images()
    
    reg = pirt.OriginalDemonsRegistration(im0, im1)
    reg.params.deform_wise = 'pairwise'
    reg.register()
    
    im0_1 = reg.get_final_deform(0, 1).apply_deformation(im0)
    im1_0 = reg.get_final_deform(1, 0).apply_deformation(im1)
   
    print('demons1', diff(im0_1, im1))
    assert diff(im0, im1) > 100
    assert diff(im0_1, im1) < 15
    assert diff(im1_0, im0) < 15


def test_demons2():
    im0, im1 = get_two_images()
    
    reg = pirt.DiffeomorphicDemonsRegistration(im0, im1)
    reg.params.deform_wise = 'groupwise'
    reg.register()
    
    im0_1 = reg.get_final_deform(0, 1).apply_deformation(im0)
    im1_0 = reg.get_final_deform(1, 0).apply_deformation(im1)
    
    print('demons2', diff(im0_1, im1))
    assert diff(im0, im1) > 100
    assert diff(im0_1, im1) < 15
    assert diff(im1_0, im0) < 15


if __name__ == '__main__':
    
    test_gravity1()
    test_gravity2()
    test_demons1()
    test_demons2()
