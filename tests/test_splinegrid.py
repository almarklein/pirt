import numpy as np
# import visvis as vv

from pirt import Aarray, PointSet, SplineGrid, FD, GridContainer
from pirt.splinegrid import calculate_multiscale_sampling
from pirt.testing import raises, run_tests_if_main


class Struct:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_field_description():
    
    a = Struct(field_shape=(20, 20), field_sampling=(2,1), field_origin=(9,9))
    fd = FD(a)
    assert fd.shape == (20, 20)
    assert fd.sampling == (2, 1)
    assert fd.origin == (9, 9)
    assert fd.ndim == 2
    assert fd.defined_sampling
    assert fd.defined_origin
    
    a = Struct(field_shape=(20, 20, 20))
    fd = FD(a)
    assert fd.shape == (20, 20, 20)
    assert fd.sampling == (1, 1, 1)
    assert fd.origin == (0, 0, 0)
    assert fd.ndim == 3
    assert not fd.defined_sampling
    assert not fd.defined_origin
    
    with raises(TypeError):
        FD('meh')
    with raises(TypeError):
        FD((20, 20), 'meh')
    with raises(TypeError):
        FD((20, 20), (2, 1), 'meh')


def test_spline_grid_1D():
    
    sg = SplineGrid(FD((90, )), 4)  # field and sampling
    
    # Test basic params
    assert sg.ndim == 1
    assert sg.field_shape == (90, )
    assert sg.field_sampling == (1, )
    
    assert sg.grid_sampling == 4
    assert sg.grid_sampling_in_pixels == (4, )
    
    assert sg.knots.shape == (26, )  # ceil(90/4) == 23. Add 3 (2 left, 1 right)
    
    # Get field, should be empty
    field = sg.get_field()
    assert field.shape == sg.field_shape
    assert np.all(field == 0)
    
    # From a field
    im = np.array([4, 4, 4, 4, 8, 4, 4, 4, 4, 7, 4, 4, 4, 4], np.float32)
    sg = SplineGrid.from_field(im, 2)
    field1 = sg.get_field()
    assert field1.max() > 6
    assert field1.max() < 7
    assert field1.min() > 4
    assert field1.min() < 4.2
    
    # From a weighted field
    im = np.array([4, 4, 4, 4, 8, 4, 4, 4, 4, 7, 4, 4, 4, 4], np.float32)
    ww = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], np.float32)
    sg = SplineGrid.from_field(im, 2, ww)
    field2 = sg.get_field()
    assert field2.max() > 7.5
    assert field2.max() < 8.5
    assert field2.min() > 0
    assert field2.min() < 0.5
    
    # From a weighted field, multiscale
    im = np.array([4, 4, 4, 4, 8, 4, 4, 4, 4, 7, 4, 4, 4, 4], np.float32)
    ww = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], np.float32)
    sg = SplineGrid.from_field_multiscale(im, 2, ww)
    field3 = sg.get_field()
    assert field3.max() > 7.5
    assert field3.max() < 8.5
    assert field3.min() > 6
    assert field3.min() < 7
    
    # From points
    pp = PointSet(1)
    pp.append(4)
    pp.append(9)
    sg = SplineGrid.from_points(FD((14, )), 2, pp, [8, 7])
    field5 = sg.get_field()
    assert all(field5 == field2)
    
    # From points, multiscale
    pp = PointSet(1)
    pp.append(4)
    pp.append(9)
    sg = SplineGrid.from_points_multiscale(FD((14, )), 2, pp, [8, 7])
    field6 = sg.get_field()
    assert all(field6 == field3)
    
    # Get field in points
    values = sg.get_field_in_points(pp)
    assert list(values) == [field6[int(p[0])] for p in pp]
    
    # Test copy
    sg2 = sg.copy()
    assert sg2.field_shape == sg.field_shape
    assert sg2.field_sampling == sg.field_sampling
    assert sg2.grid_sampling == sg.grid_sampling
    assert sg2.grid_sampling_in_pixels == sg.grid_sampling_in_pixels
    assert sg2.knots is not sg.knots
    
    # Test refine
    sg2 = sg.refine()
    assert sg2.field_shape == sg.field_shape
    assert sg2.field_sampling == sg.field_sampling
    assert sg2.grid_sampling == sg.grid_sampling / 2
    assert sg2.grid_sampling_in_pixels == tuple([i/2 for i in sg.grid_sampling_in_pixels])
    
    # Test resize_field
    sg2 = sg.resize_field(FD((50, )))
    assert sg2.get_field().shape == (50, )
    
    # Test addition
    sg2 = sg.add(sg.add(sg))
    field = sg2.get_field()
    assert np.allclose(field, field6 * 3)
    
    print('test_spline_grid_1D ok')
    
    
def test_spline_grid_2D():
    
    sg = SplineGrid(FD((90, 90)), 4)  # field and sampling
    
    # Test basic params
    assert sg.ndim == 2
    assert sg.field_shape == (90, 90)
    assert sg.field_sampling == (1, 1)
    
    assert sg.grid_sampling == 4
    assert sg.grid_sampling_in_pixels == (4, 4)
    
    assert sg.knots.shape == (26, 26)  # ceil(90/4) == 23. Add 3 (2 left, 1 right)
    
    # Get field, should be empty
    field = sg.get_field()
    assert field.shape == sg.field_shape
    assert np.all(field == 0)
    
    # From a field
    im = 4 * np.ones((20, 20), np.float32)
    im[4, 5] = 8
    im[8, 9] = 7
    sg = SplineGrid.from_field(im, 2)
    field1 = sg.get_field()
    assert field1.ndim == 2
    assert field1.max() > 5.5
    assert field1.max() < 6.5
    assert field1.min() > 4
    assert field1.min() < 4.5
    
    # From a weighted field
    im = 4 * np.ones((20, 20), np.float32)
    im[4, 5] = 8
    im[8, 9] = 7
    ww = np.zeros((20, 20), np.float32)
    ww[4, 5] = 1
    ww[8, 9] = 1
    sg = SplineGrid.from_field(im, 2, ww)
    field2 = sg.get_field()
    assert field2.ndim == 2
    assert field2.max() > 7.5
    assert field2.max() < 8.5
    assert field2.min() >= 0
    assert field2.min() < 0.5
    
    # From a weighted field, multiscale
    im = 4 * np.ones((20, 20), np.float32)
    im[4, 5] = 8
    im[8, 9] = 7
    ww = np.zeros((20, 20), np.float32)
    ww[4, 5] = 1
    ww[8, 9] = 1
    sg = SplineGrid.from_field_multiscale(im, 2, ww)
    field3 = sg.get_field()
    assert field3.ndim == 2
    assert field3.max() > 7.5
    assert field3.max() < 8.5
    assert field3.min() > 5
    assert field3.min() < 7
    
    # From points
    pp = PointSet(2)
    pp.append(5, 4)
    pp.append(9, 8)
    sg2 = SplineGrid.from_points(FD((20, 20)), 2, pp, [8, 7])
    field5 = sg2.get_field()
    assert np.all(field5 == field2)
    
    # From points, multiscale
    pp = PointSet(2)
    pp.append(5, 4)
    pp.append(9, 8)
    sg = SplineGrid.from_points_multiscale(FD((20, 20)), 2, pp, [8, 7])
    field6 = sg.get_field()
    assert np.all(field6 == field3)
    
    # Get field in points
    values = sg.get_field_in_points(pp)
    assert list(values) == [field6[int(p[0,1]), int(p[0,0])] for p in pp]
    
    # Get field in points beyond field
    pp = PointSet(2)
    pp.append(100, 103)
    pp.append(-100, -108)
    values = sg.get_field_in_points(pp)
    assert list(values) == [field6[int(p[0,1]), int(p[0,0])] for p in pp]
    
    # Test copy
    sg2 = sg.copy()
    assert sg2.field_shape == sg.field_shape
    assert sg2.field_sampling == sg.field_sampling
    assert sg2.grid_sampling == sg.grid_sampling
    assert sg2.grid_sampling_in_pixels == sg.grid_sampling_in_pixels
    assert sg2.knots is not sg.knots
    
    # Test refine
    sg2 = sg.refine()
    assert sg2.field_shape == sg.field_shape
    assert sg2.field_sampling == sg.field_sampling
    assert sg2.grid_sampling == sg.grid_sampling / 2
    assert sg2.grid_sampling_in_pixels == tuple([i/2 for i in sg.grid_sampling_in_pixels])
    
    # Test resize_field
    sg2 = sg.resize_field(FD((50, 50)))
    assert sg2.get_field().shape == (50, 50)
    
    # Test addition
    sg2 = sg.add(sg.add(sg))
    field = sg2.get_field()
    assert np.allclose(field, field6 * 3)
    
    print('test_spline_grid_2D ok')


def test_spline_grid_3D():
    
    sg = SplineGrid(FD((90, 90, 90)), 4)  # field and sampling
    
    # Test basic params
    assert sg.ndim == 3
    assert sg.field_shape == (90, 90, 90)
    assert sg.field_sampling == (1, 1, 1)
    
    assert sg.grid_sampling == 4
    assert sg.grid_sampling_in_pixels == (4, 4, 4)
    
    assert sg.knots.shape == (26, 26, 26)  # ceil(90/4) == 23. Add 3 (2 left, 1 right)
    
    # Get field, should be empty
    field = sg.get_field()
    assert field.shape == sg.field_shape
    assert np.all(field == 0)
    
    # From a field
    im = 4 * np.ones((20, 20, 20), np.float32)
    im[4, 5, 5] = 8
    im[8, 9, 9] = 7
    sg = SplineGrid.from_field(im, 2)
    field1 = sg.get_field()
    assert field1.ndim == 3
    assert field1.max() > 5.5
    assert field1.max() < 6.5
    assert field1.min() > 4
    assert field1.min() < 4.7
    
    # From a weighted field
    im = 4 * np.ones((20, 20, 20), np.float32)
    im[4, 5, 5] = 8
    im[8, 9, 9] = 7
    ww = np.zeros((20, 20, 20), np.float32)
    ww[4, 5, 5] = 1
    ww[8, 9, 9] = 1
    sg = SplineGrid.from_field(im, 2, ww)
    field2 = sg.get_field()
    assert field2.ndim == 3
    assert field2.max() > 7.5
    assert field2.max() < 8.5
    assert field2.min() >= 0
    assert field2.min() < 0.5
    
    # From a weighted field, multiscale
    im = 4 * np.ones((20, 20, 20), np.float32)
    im[4, 5, 5] = 8
    im[8, 9, 9] = 7
    ww = np.zeros((20, 20, 20), np.float32)
    ww[4, 5, 5] = 1
    ww[8, 9, 9] = 1
    sg = SplineGrid.from_field_multiscale(im, 2, ww)
    field3 = sg.get_field()
    assert field3.ndim == 3
    assert field3.max() > 7.5
    assert field3.max() < 8.5
    assert field3.min() > 4.5
    assert field3.min() < 7
    
    # From points
    pp = PointSet(3)
    pp.append(5, 5, 4)
    pp.append(9, 9, 8)
    sg2 = SplineGrid.from_points(FD((20, 20, 20)), 2, pp, [8, 7])
    field5 = sg2.get_field()
    assert np.all(field5 == field2)
    
    # From points, multiscale
    pp = PointSet(3)
    pp.append(5, 5, 4)
    pp.append(9, 9, 8)
    sg = SplineGrid.from_points_multiscale(FD((20, 20, 20)), 2, pp, [8, 7])
    field6 = sg.get_field()
    assert np.all(field6 == field3)
    
    # Get field in points
    values = sg.get_field_in_points(pp)
    assert list(values) == [field6[int(p[0,2]), int(p[0,1]), int(p[0,0])] for p in pp]
    
    # Test copy
    sg2 = sg.copy()
    assert sg2.field_shape == sg.field_shape
    assert sg2.field_sampling == sg.field_sampling
    assert sg2.grid_sampling == sg.grid_sampling
    assert sg2.grid_sampling_in_pixels == sg.grid_sampling_in_pixels
    assert sg2.knots is not sg.knots
    
    # Test refine
    sg2 = sg.refine()
    assert sg2.field_shape == sg.field_shape
    assert sg2.field_sampling == sg.field_sampling
    assert sg2.grid_sampling == sg.grid_sampling / 2
    assert sg2.grid_sampling_in_pixels == tuple([i/2 for i in sg.grid_sampling_in_pixels])
    
    # Test resize_field
    sg2 = sg.resize_field(FD((50, 50, 50)))
    assert sg2.get_field().shape == (50, 50, 50)
    
    # Test addition
    sg2 = sg.add(sg.add(sg))
    field = sg2.get_field()
    assert np.allclose(field, field6 * 3)
    
    print('test_spline_grid_3D ok')


def test_spline_grid_1D_anisotropic():
    
    sg = SplineGrid(FD((90, ), (0.5, )), 4)  # field and sampling
    
    # Test basic params
    assert sg.ndim == 1
    assert sg.field_shape == (90, )
    assert sg.field_sampling == (0.5, )
    
    assert sg.grid_sampling == 4
    assert sg.grid_sampling_in_pixels == (8, )
    
    assert sg.knots.shape == (15, )
    
    # Get field, should be empty
    field = sg.get_field()
    assert field.shape == sg.field_shape
    assert np.all(field == 0)
    
    # From a field
    im = np.array([4, 4, 4, 4, 8, 4, 4, 4, 4, 7, 4, 4, 4, 4], np.float32)
    im = Aarray(im, (0.5, ))
    sg = SplineGrid.from_field(im, 2)
    field1 = sg.get_field()
    assert field1.max() > 5.5
    assert field1.max() < 7
    assert field1.min() > 4
    assert field1.min() < 4.2
    
    # From a weighted field
    im = np.array([4, 4, 4, 4, 8, 4, 4, 4, 4, 7, 4, 4, 4, 4], np.float32)
    ww = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], np.float32)
    im = Aarray(im, (0.5, ))
    sg = SplineGrid.from_field(im, 2, ww)
    field2 = sg.get_field()
    assert field2.max() > 7.5
    assert field2.max() < 9.5
    assert field2.min() > 0
    assert field2.min() < 3.5
    
    # From a weighted field, multiscale
    im = np.array([4, 4, 4, 4, 8, 4, 4, 4, 4, 7, 4, 4, 4, 4], np.float32)
    ww = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], np.float32)
    im = Aarray(im, (0.5, ))
    sg = SplineGrid.from_field_multiscale(im, 2, ww)
    field3 = sg.get_field()
    assert field3.max() > 7.5
    assert field3.max() < 8.5
    assert field3.min() > 6
    assert field3.min() < 7
    
    # From points
    pp = PointSet(1)
    pp.append(4)
    pp.append(9)
    pp2 = pp * PointSet((0.5, ))
    sg = SplineGrid.from_points(FD((14, ), (0.5, )), 2, pp2, [8, 7])
    field5 = sg.get_field()
    assert all(field5 == field2)
    
    # From points, multiscale
    pp = PointSet(1)
    pp.append(4)
    pp.append(9)
    pp2 = pp * PointSet((0.5, ))
    sg = SplineGrid.from_points_multiscale(FD((14, ), (0.5, )), 2, pp2, [8, 7])
    field6 = sg.get_field()
    assert all(field6 == field3)
    
    # Get field in points, note pp2, which is in world coords
    values = sg.get_field_in_points(pp2)
    assert list(values) == [field6[int(p[0])] for p in pp]
    
    # Get field in samples
    values2 = sg.get_field_in_samples((pp[:, 0], ))
    assert list(values) == list(values2)
    
    # Test copy
    sg2 = sg.copy()
    assert sg2.field_shape == sg.field_shape
    assert sg2.field_sampling == sg.field_sampling
    assert sg2.grid_sampling == sg.grid_sampling
    assert sg2.grid_sampling_in_pixels == sg.grid_sampling_in_pixels
    assert sg2.knots is not sg.knots
    
    # Test refine
    sg2 = sg.refine()
    assert sg2.field_shape == sg.field_shape
    assert sg2.field_sampling == sg.field_sampling
    assert sg2.grid_sampling == sg.grid_sampling / 2
    assert sg2.grid_sampling_in_pixels == tuple([i/2 for i in sg.grid_sampling_in_pixels])
    
    # Test resize_field
    sg2 = sg.resize_field(FD((50, )))
    assert sg2.get_field().shape == (50, )
    
    # Test addition
    sg2 = sg.add(sg.add(sg))
    field = sg2.get_field()
    assert np.allclose(field, field6 * 3)
    
    print('test_spline_grid_1D_anisotropic ok')


def test_spline_grid_2D_anisotropic():
    
    sg = SplineGrid(FD((90, 90), (0.5, 2.2), (11, 12)), 4)  # field and sampling
    
    # Test basic params
    assert sg.ndim == 2
    assert sg.field_shape == (90, 90)
    assert sg.field_sampling == (0.5, 2.2)
    
    assert sg.grid_sampling == 4
    assert sg.grid_sampling_in_pixels == (4/0.5, 4/2.2)
    
    assert sg.knots.shape == (15, 52)
    
    # Get field, should be empty
    field = sg.get_field()
    assert field.sampling == sg.field_sampling
    assert field.origin == (0, 0)  # Origin is ignored!
    assert field.shape == sg.field_shape
    assert np.all(field == 0)
    
    # From a field
    im = 4 * np.ones((20, 20), np.float32)
    im[4, 5] = 8
    im[8, 9] = 7
    im = Aarray(im, (0.5, 2.2))
    sg = SplineGrid.from_field(im, 2)
    field1 = sg.get_field()
    assert field1.ndim == 2
    assert field1.max() > 5.5
    assert field1.max() < 6.5
    assert field1.min() > 4
    assert field1.min() < 4.5
    
    # From a weighted field
    im = 4 * np.ones((20, 20), np.float32)
    im[4, 5] = 8
    im[8, 9] = 7
    im = Aarray(im, (0.5, 2.0))
    ww = np.zeros((20, 20), np.float32)
    ww[4, 5] = 1
    ww[8, 9] = 1
    sg = SplineGrid.from_field(im, 2, ww)
    field2 = sg.get_field()
    assert field2.ndim == 2
    assert field2.max() > 7.5
    assert field2.max() < 8.5
    assert field2.min() >= 0
    assert field2.min() < 0.5
    
    # From a weighted field, multiscale
    im = 4 * np.ones((20, 20), np.float32)
    im[4, 5] = 8
    im[8, 9] = 7
    im = Aarray(im, (0.5, 2.0))
    ww = np.zeros((20, 20), np.float32)
    ww[4, 5] = 1
    ww[8, 9] = 1
    sg = SplineGrid.from_field_multiscale(im, 2, ww)
    field3 = sg.get_field()
    assert field3.ndim == 2
    assert field3.max() > 7.5
    assert field3.max() < 8.5
    assert field3.min() > 5
    assert field3.min() < 7
    
    # From points
    pp = PointSet(2)
    pp.append(5, 4)
    pp.append(9, 8)
    pp2 = pp * PointSet((2.0, 0.5))
    sg2 = SplineGrid.from_points(FD((20, 20), (0.5, 2.0)), 2, pp2, [8, 7])
    field5 = sg2.get_field()
    assert np.all(field5 == field2)
    
    # From points, multiscale
    pp = PointSet(2)
    pp.append(5, 4)
    pp.append(9, 8)
    pp2 = pp * PointSet((2.0, 0.5))
    sg = SplineGrid.from_points_multiscale(FD((20, 20), (0.5, 2.0)), 2, pp2, [8, 7])
    field6 = sg.get_field()
    assert np.all(field6 == field3)
    
    # Get field in points
    values = sg.get_field_in_points(pp2)
    assert list(values) == [field6[int(p[0,1]), int(p[0,0])] for p in pp]
    
    # Get field in samples, x-y-z order
    values2 = sg.get_field_in_samples((pp[:, 1], pp[:, 0] ))
    assert list(values) != list(values2)
    values2 = sg.get_field_in_samples((pp[:, 0], pp[:, 1] ))
    assert list(values) == list(values2)
    
    # Test copy
    sg2 = sg.copy()
    assert sg2.field_shape == sg.field_shape
    assert sg2.field_sampling == sg.field_sampling
    assert sg2.grid_sampling == sg.grid_sampling
    assert sg2.grid_sampling_in_pixels == sg.grid_sampling_in_pixels
    assert sg2.knots is not sg.knots
    
    # Test refine
    sg2 = sg.refine()
    assert sg2.field_shape == sg.field_shape
    assert sg2.field_sampling == sg.field_sampling
    assert sg2.grid_sampling == sg.grid_sampling / 2
    assert sg2.grid_sampling_in_pixels == tuple([i/2 for i in sg.grid_sampling_in_pixels])
    
    # Test resize_field
    sg2 = sg.resize_field(FD((50, 50)))
    assert sg2.get_field().shape == (50, 50)
    
    # Test addition
    sg2 = sg.add(sg.add(sg))
    field = sg2.get_field()
    assert np.allclose(field, field6 * 3)
    
    print('test_spline_grid_2D_anisotropic ok')


def test_spline_grid_3D_anisotropic():
    
    sg = SplineGrid(FD((90, 90, 90), (2.0, 3.0, 0.5)), 4)  # field and sampling
    
    # Test basic params
    assert sg.ndim == 3
    assert sg.field_shape == (90, 90, 90)
    assert sg.field_sampling == (2.0, 3.0, 0.5)
    
    assert sg.grid_sampling == 4
    assert sg.grid_sampling_in_pixels == (4/2.0, 4/3.0, 4/0.5)
    
    assert sg.knots.shape == (48, 70, 15)  # ceil(90/4) == 23. Add 3 (2 left, 1 right)
    
    # Get field, should be empty
    field = sg.get_field()
    assert field.shape == sg.field_shape
    assert np.all(field == 0)
    
    # From a field
    im = 4 * np.ones((20, 20, 20), np.float32)
    im[4, 5, 5] = 8
    im[8, 9, 9] = 7
    im = Aarray(im, (2.0, 3.0, 0.5))
    sg = SplineGrid.from_field(im, 2)
    field1 = sg.get_field()
    assert field1.ndim == 3
    assert field1.max() > 5.5
    assert field1.max() < 7.5
    assert field1.min() > 4
    assert field1.min() < 4.7
    
    # From a weighted field
    im = 4 * np.ones((20, 20, 20), np.float32)
    im[4, 5, 5] = 8
    im[8, 9, 9] = 7
    ww = np.zeros((20, 20, 20), np.float32)
    ww[4, 5, 5] = 1
    ww[8, 9, 9] = 1
    im = Aarray(im, (2.0, 3.0, 0.5))
    sg = SplineGrid.from_field(im, 2, ww)
    field2 = sg.get_field()
    assert field2.ndim == 3
    assert field2.max() > 7.5
    assert field2.max() < 8.5
    assert field2.min() >= 0
    assert field2.min() < 0.5
    
    # From a weighted field, multiscale
    im = 4 * np.ones((20, 20, 20), np.float32)
    im[4, 5, 5] = 8
    im[8, 9, 9] = 7
    ww = np.zeros((20, 20, 20), np.float32)
    ww[4, 5, 5] = 1
    ww[8, 9, 9] = 1
    im = Aarray(im, (2.0, 3.0, 0.5))
    sg = SplineGrid.from_field_multiscale(im, 2, ww)
    field3 = sg.get_field()
    assert field3.ndim == 3
    assert field3.max() > 7.5
    assert field3.max() < 8.5
    assert field3.min() > 4.5
    assert field3.min() < 7
    
    # From points
    pp = PointSet(3)
    pp.append(5, 5, 4)
    pp.append(9, 9, 8)
    pp2 = pp * PointSet((0.5, 3.0, 2.0))
    sg2 = SplineGrid.from_points(FD((20, 20, 20), (2.0, 3.0, 0.5)), 2, pp2, [8, 7])
    field5 = sg2.get_field()
    assert np.all(field5 == field2)
    
    # From points, multiscale
    pp = PointSet(3)
    pp.append(5, 5, 4)
    pp.append(9, 9, 8)
    pp2 = pp * PointSet((0.5, 3.0, 2.0))
    sg = SplineGrid.from_points_multiscale(FD((20, 20, 20), (2.0, 3.0, 0.5)), 2, pp2, [8, 7])
    field6 = sg.get_field()
    assert np.all(field6 == field3)
    
    # Get field in points
    values = sg.get_field_in_points(pp2)
    assert list(values) == [field6[int(p[0, 2]), int(p[0,1]), int(p[0,0])] for p in pp]
    
    # Get field in samples, x-y-z order
    values2 = sg.get_field_in_samples((pp[:, 2], pp[:, 1], pp[:, 0] ))
    assert list(values) != list(values2)
    values2 = sg.get_field_in_samples((pp[:, 0], pp[:, 1], pp[:, 2]))
    assert list(values) == list(values2)
    
    # Test copy
    sg2 = sg.copy()
    assert sg2.field_shape == sg.field_shape
    assert sg2.field_sampling == sg.field_sampling
    assert sg2.grid_sampling == sg.grid_sampling
    assert sg2.grid_sampling_in_pixels == sg.grid_sampling_in_pixels
    assert sg2.knots is not sg.knots
    
    # Test refine
    sg2 = sg.refine()
    assert sg2.field_shape == sg.field_shape
    assert sg2.field_sampling == sg.field_sampling
    assert sg2.grid_sampling == sg.grid_sampling / 2
    assert sg2.grid_sampling_in_pixels == tuple([i/2 for i in sg.grid_sampling_in_pixels])
    
    # Test resize_field
    sg2 = sg.resize_field(FD((50, 50, 50)))
    assert sg2.get_field().shape == (50, 50, 50)
    
    # Test addition
    sg2 = sg.add(sg.add(sg))
    field = sg2.get_field()
    assert np.allclose(field, field6 * 3)
    
    print('test_spline_grid_3D_anisotropic ok')


def test_grid_container():
    
    class ColorGridContainer(GridContainer):
        
        def __init__(self, shape, grid_sampling):
            super().__init__(shape, grid_sampling)
            for i in range(3):
                self._grids.append(SplineGrid(shape, grid_sampling))
    
    gc = ColorGridContainer((100, 100), 3)
    assert gc.field_shape == (100, 100)
    
    assert len(gc) == 3
    assert len(gc.grids) == 3
    for grid in gc:
        assert isinstance(grid, SplineGrid)
    assert list(gc) == [gc[0], gc[1], gc[2]]
    with raises(IndexError):
        gc[3]
    with raises(IndexError):
        gc[-1]
    with raises(IndexError):
        gc[:]
    
    # resize
    gc2 = gc.resize_field((60, 60))
    assert gc2.field_shape == (60, 60)
    
    # add
    gc3 = gc.add(gc)
    assert gc3.field_shape == (100, 100)
    assert gc3 is not gc  # not in-place
    for i in range(3):
        gc3[i] is not gc[i]
    
    # refine
    gc4 = gc.refine()
    assert gc4.field_shape == (100, 100)
    assert gc4.grid_sampling == gc.grid_sampling / 2
    for i in range(3):
        gc4[i].grid_sampling == gc4.grid_sampling
    
    # copy
    gc5 = gc.add(gc)
    assert gc5.field_shape == (100, 100)
    assert gc5 is not gc  # not in-place
    for i in range(3):
        gc5[i] is not gc[i]


def test_calculate_multiscale_sampling():
    
    # From grid and one value
    
    fd = FD((100, 200), (2, 2))
    mi, ma = calculate_multiscale_sampling(SplineGrid(fd, 1), 10)
    assert mi == 10
    assert ma > 200 * 2 and ma / 2 < 200 * 2
    assert ma > 10
    while ma > mi:
        ma /= 2
    assert mi == ma
    
    fd = FD((2, 9), (0.2, 0.2))
    mi, ma = calculate_multiscale_sampling(SplineGrid(fd, 1), 10)
    assert mi == 10
    assert ma == 10
    while ma > mi:
        ma /= 2
    assert mi == ma
    
    fd = FD((2, 9), (3, 3))
    mi, ma = calculate_multiscale_sampling(SplineGrid(fd, 1), 10)
    assert mi == 10
    assert ma > 9 * 3 and ma / 2 < 9 * 3
    assert ma > 10
    while ma > mi:
        ma /= 2
    assert mi == ma
    
    # From two values, selects such that ma is power-of-two of mi
    
    mi, ma = calculate_multiscale_sampling(None, (10, 10))
    assert (mi, ma) == (10, 10)
    while ma > mi:
        ma /= 2
    assert mi == ma
    
    mi, ma = calculate_multiscale_sampling(None, (3, 17))
    assert (mi, ma) == (3, 12)
    while ma > mi:
        ma /= 2
    assert mi == ma
    
    mi, ma = calculate_multiscale_sampling(None, (3, 18))
    assert (mi, ma) == (3, 24)
    while ma > mi:
        ma /= 2
    assert mi == ma

    
if __name__ == '__main__':
    # test_field_description()
    # test_spline_grid_1D()
    # test_spline_grid_2D()
    # test_spline_grid_3D()
    # test_spline_grid_3D_anisotropic()
    # test_spline_grid_3D_anisotropic()
    # test_grid_container()
    # test_calculate_multiscale_sampling()
    
    run_tests_if_main()
