""" Test script for DeformationField and DeformationGrid class
"""

import numpy as np
# import visvis as vv

from pirt import (PointSet, FD, DeformationGridForward, DeformationFieldForward,
                  DeformationGridBackward, DeformationFieldBackward)

import pirt


def cog(im):
    """ Calculate center of gravity (x, y), in world coordinates.
    """
    sampling = tuple(reversed(im.sampling))
    grids = pirt.meshgrid(im)
    total_weight = im.sum()
    return PointSet([sampling[i] * (grids[i] * im).sum() / total_weight for i in range(len(grids))])


def get_data_small_deform():
    """ Create an image with a block if white pixels and two deforms, one to
    move it down, and another to move it right, in the form of a field and
    a pointset.
    """
    
    # Create test image
    im0 = np.zeros((100, 100), np.float32)
    im0 = pirt.Aarray(im0, (0.5, 3.0))
    im0[20:30, 40:50] = 1.0
    c0 = cog(im0)
    
    # Create test deformation fields - 50 px down and 40 px right
    dfield1 = np.zeros((100, 100), np.float32), np.zeros((100, 100), np.float32)
    weight1 = np.zeros((100, 100), np.float32)
    dfield1[0][25+10, 45] = -10 * im0.sampling[0]  # because backward, correct for sampling
    weight1[25+10, 45] = 1
    #
    dfield2 = np.zeros((100, 100), np.float32), np.zeros((100, 100), np.float32)
    weight2 = np.zeros((100, 100), np.float32)
    dfield2[1][35, 45+10] = -10 * im0.sampling[1]
    weight2[35, 45+10] = 1
    
    # Create test deformation pointsets - 50 px down and 40 px right
    pp1, pp2, pp3 = PointSet(2), PointSet(2), PointSet(2)
    pp1.append(45, 25) # begin pos
    pp2.append(45, 35) # intermediate
    pp3.append(55, 35) # end pos
    for pp in (pp1, pp2, pp3):
        pp[:] *= PointSet(reversed(im0.sampling))
    
    return im0, c0, dfield1, weight1, dfield2, weight2, pp1, pp2, pp3


def get_data_big_deform():
    """ Create an image with a block if white pixels and two deforms, one to
    move it down, and another to move it right, in the form of a field and
    a pointset.
    """
    
    # Create test image
    im0 = np.zeros((100, 100), np.float32)
    im0 = pirt.Aarray(im0, (0.25, 2.0))
    im0[30:40, 40:50] = 1.0
    c0 = cog(im0)
    
    # Create test deformation fields - 50 px down and 40 px right
    dfield1 = np.zeros((100, 100), np.float32), np.zeros((100, 100), np.float32)
    weight1 = np.zeros((100, 100), np.float32)
    dfield1[0][35+50, 45] = -50 * im0.sampling[0]  # because backward, correct for sampling
    weight1[35+50, 45] = 1
    #
    dfield2 = np.zeros((100, 100), np.float32), np.zeros((100, 100), np.float32)
    weight2 = np.zeros((100, 100), np.float32)
    dfield2[1][85, 45+40] = -40 * im0.sampling[1]
    weight2[85, 45+40] = 1
    
    # Create test deformation pointsets - 50 px down and 40 px right
    pp1, pp2, pp3 = PointSet(2), PointSet(2), PointSet(2)
    pp1.append(45, 35) # begin pos
    pp2.append(45, 85) # intermediate
    pp3.append(85, 85) # end pos
    for pp in (pp1, pp2, pp3):
        pp[:] *= PointSet(reversed(im0.sampling))
    
    return im0, c0, dfield1, weight1, dfield2, weight2, pp1, pp2, pp3


def test_deformation_grid():
    """ Test deformation grid, there is only so much that we can do ...
    """
    
    im0, c0, dfield1, weight1, dfield2, weight2, pp1, pp2, pp3 = get_data_small_deform()
    gridsampling = 6
    
    # Create identity deform
    d1 = DeformationGridBackward(im0, gridsampling)
    assert d1.ndim == 2
    assert d1.field_sampling == im0.sampling
    for grid in d1.grids:
        assert grid.field_sampling == im0.sampling
        assert grid.grid_sampling == gridsampling
    #
    im1 = d1.apply_deformation(im0)
    assert np.all(im1 == im0)
    
    # Deform from field
    
    # Create a deform - wrong weight, so is unit
    d2 = DeformationGridBackward.from_field(dfield1, gridsampling, weight2, fd=im0, injective=False, frozenedge=False)
    im2 = d2.apply_deformation(im0)
    assert np.all(im2 == im0)
    for grid in d2.grids:
        assert np.all(grid._knots == 0)
    
    # Create a deform - now get it right
    d2 = DeformationGridBackward.from_field(dfield1, gridsampling, weight1, fd=im0, injective=False, frozenedge=False)
    assert d2.field_sampling == im0.sampling
    for grid in d2.grids:
        assert grid.field_sampling == im0.sampling
        assert grid.grid_sampling == gridsampling
    im2 = d2.apply_deformation(im0)
    c2 = cog(im2)
    # Assert that we shifted down, if only by a bit
    assert abs(c2[0,0] - c0[0,0]) < 1 and c2[0,1] > c0[0,1] + 1
    
    # Deform from points, single-step
    
    # Create a deform - now get it right
    d3 = DeformationGridBackward.from_points(im0, gridsampling, pp1, pp2, injective=False, frozenedge=False)
    assert d3.field_sampling == im0.sampling
    im3 = d3.apply_deformation(im0)
    c3 = cog(im3)
    assert np.all(im2 == im3)
    assert c2.distance(c3)[0] == 0

    # vv.figure(1); vv.clf(); vv.subplot(221); vv.imshow(im0); vv.subplot(222); vv.imshow(im2); vv.subplot(223); vv.imshow(im3);
    
    # vv.figure(2); vv.clf(); vv.subplot(211); d2.show(); vv.subplot(212); d3.show()
    
    print('deformation_grid ok')


def test_deformation_grid_multiscale():
    """ Multiscale, so that we can get the actual deformation, but keep in mind
    that DeformationField.from_xxx() are to be preferred because they care
    about injectivity and composition.
    """
    
    im0, c0, dfield1, weight1, dfield2, weight2, pp1, pp2, pp3 = get_data_big_deform()
    
    # Shift down
    
    # Shift down using deformation as field
    d4 = DeformationGridBackward.from_field_multiscale(dfield1, 4, weight1, fd=im0)
    assert d4.field_sampling == im0.sampling
    for grid in d4.grids:
        assert grid.field_sampling == im0.sampling
        assert grid.grid_sampling == 4
    
    # Shift down using deformation as points
    d5 = DeformationGridBackward.from_points_multiscale(im0, 4, pp1, pp2)
    assert d5.field_sampling == im0.sampling
    for grid in d5.grids:
        assert grid.field_sampling == im0.sampling
        assert grid.grid_sampling == 4
    
    # Test that d4 and d5 are equal, so further tests can be combined
    for d in range(d4.ndim):
        assert np.all(d4.get_field(d) == d5.get_field(d))
    
    # Assert that the deform shifts down
    im4 = d4.apply_deformation(im0)
    c4 = cog(im4)
    assert c4.distance(c0 + PointSet((0, 50 * im0.sampling[0]))) < 1
    
    # Shift right
    
    # Create deforms
    d6 = DeformationGridBackward.from_field_multiscale(dfield2, 4, weight2, fd=im0)
    d7 = DeformationGridBackward.from_points_multiscale(im0, 4, pp2, pp3)
    
    # Test that d6 and d7 are equal, so further tests can be combined
    for d in range(d6.ndim):
        assert np.all(d6.get_field(d) == d7.get_field(d))
    
    # Shift the original image to the right, works because whole image is shifted
    im6 = d6.apply_deformation(im0)
    c6 = cog(im6)
    assert c6.distance(c0 + PointSet((40 * im0.sampling[1], 0))) < 1
    
    # Shift down-shifted image to right
    im6 = d6.apply_deformation(im4)
    c6 = cog(im6)
    assert c6.distance(c0 + PointSet((40 * im0.sampling[1], 50 * im0.sampling[0]))) < 1
    
    # Combine deforms in wrong way, but result is still pretty good because deform is near-uniform
    d7 = d4 + d6
    im7 = d7.apply_deformation(im0)
    assert not np.allclose(im7, im6, atol=1)
    
    # Compising is much better!
    d8 = d4.compose(d6)
    im8 = d8.apply_deformation(im0)
    assert np.allclose(im8, im6, atol=0.01)
    
    # vv.figure(1); vv.clf(); vv.subplot(221); vv.imshow(im0); vv.subplot(222); vv.imshow(im6); vv.subplot(223); vv.imshow(im7); vv.subplot(224); vv.imshow(im8)
    
    print('deformation_grid_multiscale ok')


def test_deformation_field():
    """ Test deformation field, now we can really do a deform ...
    we can force injectivity, but we dont freeze edges, because it would
    deform the blocks a lot.
    """
    
    im0, c0, dfield1, weight1, dfield2, weight2, pp1, pp2, pp3 = get_data_big_deform()
    
    # Create identity deform
    d1 = DeformationFieldBackward(FD(im0))
    assert d1.ndim == 2
    assert d1.field_sampling == im0.sampling
    for f in d1._fields:
        assert f.fsampling == im0.sampling
    #
    im1 = d1.apply_deformation(im0)
    assert np.all(im1 == im0)
    assert d1.is_identity
    
    # Get down deformation from field
    d2 = DeformationFieldBackward.from_field_multiscale(dfield1, 4, weight1, fd=im0, injective=True, frozenedge=True)
    assert d2.field_sampling == im0.sampling
    for f in d2._fields:
        assert f.sampling == im0.sampling
    
    # Get down deformation from points
    d3 = DeformationFieldBackward.from_points_multiscale(im0, 4, pp1, pp2, injective=True, frozenedge=True)
    assert d1.field_sampling == im0.sampling
    for f in d3._fields:
        assert f.sampling == im0.sampling
    
    # Get right deformation from field
    d4 = DeformationFieldBackward.from_field_multiscale(dfield2, 4, weight2, fd=im0, injective=True, frozenedge=True)
    assert d4.field_sampling == im0.sampling
    for f in d4._fields:
        assert f.sampling == im0.sampling
    
    # Get right deformation from points
    d5 = DeformationFieldBackward.from_points_multiscale(im0, 4, pp2, pp3, injective=True, frozenedge=True)
    assert d5.field_sampling == im0.sampling
    for f in d5._fields:
        assert f.sampling == im0.sampling
    
    # Test that d2 and d3 are equal, so further tests can be combined
    # With frozen edges, this does not hold, since the multiscale approach
    # is done using grids vs fields, so you get slighly different results.
    # for d in range(d2.ndim):
    #     assert np.all(d2.get_field(d) == d3.get_field(d))
    
    # Apply downward deform
    im2 = d2.apply_deformation(im0)
    im3 = d3.apply_deformation(im0)
    c2 = cog(im2)
    c3 = cog(im3)
    assert c2[0, 1] > c0[0, 1] + 10 * im0.sampling[0]  # frozen edges hold us back
    assert c3[0, 1] > c0[0, 1] + 10 * im0.sampling[0]  # frozen edges hold us back
    
    # Compose right deform in two ways
    im4 = d4.apply_deformation(im2)
    im5 = d5.apply_deformation(im3)
    im6 = d2.compose(d4).apply_deformation(im0)
    im7 = d3.compose(d5).apply_deformation(im0)
    c4, c5, c6, c7 = cog(im4), cog(im5), cog(im6), cog(im7)
    assert c4.distance(c6) < 0.1
    assert c5.distance(c7) < 0.1
    assert c4.distance(c5) < 5
    
    # Reverse ... im0 -> d2 -> d4 -> im6
    
    im0r1 = d2.inverse().apply_deformation( d4.inverse().apply_deformation(im6) )
    im0r2 = d4.inverse().compose(d2.inverse()).apply_deformation(im6)
    im0r3 = d2.compose(d4).inverse().apply_deformation(im6)
    
    assert cog(im0r1).distance(c0) < 0.05
    assert cog(im0r2).distance(c0) < 0.05
    assert cog(im0r3).distance(c0) < 0.11  # mmm, was expecting this one to perform best
    
    # Goof around with forward mapping to achieve the same
    # In this, case the inversion is costless, but we pay the price when we aply
    # the deformation, which is slower and less accurate
    im0r4 = d2.compose(d4).as_forward_inverse().apply_deformation(im6)
    assert cog(im0r4).distance(c0) > 0.1
    assert cog(im0r4).distance(c0) < 0.5
    
    # Test composition op
    im0r5 = (d4 * d2).inverse().apply_deformation(im6)
    assert cog(im0r5).distance(c0) < 0.11
    
    # vv.figure(1); vv.clf(); vv.subplot(221); vv.imshow(im0); vv.subplot(222); vv.imshow(im2); vv.subplot(223); vv.imshow(im4); vv.subplot(224); vv.imshow(im6)
    
    # vv.figure(2); vv.clf(); vv.subplot(211); d2.show(); vv.subplot(212); d3.show()
    
    print('deformation_field ok')



if __name__ == '__main__':
    
    test_deformation_grid()
    test_deformation_grid_multiscale()
    test_deformation_field()
