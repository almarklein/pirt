import numpy as np
import pirt
from pirt.testing import raises, run_tests_if_main


def test_cubic():
    
    assert pirt.get_cubic_spline_coefs(0.46, 'nearest') == (0, 1, 0, 0)
    assert pirt.get_cubic_spline_coefs(0.54, 'nearest') == (0, 0, 1, 0)
    assert pirt.get_cubic_spline_coefs(0.44, 'linear') == (0, 1 - 0.44, 0.44, 0)
    
    cc1 = pirt.get_cubic_spline_coefs(0.44, 'catmull–rom')
    cc2 = pirt.get_cubic_spline_coefs(0.44, 'cardinal')
    cc3 = pirt.get_cubic_spline_coefs(0.44, 0.0)
    assert cc1 == cc2 == cc3
    
    # Wrong spline type
    with raises(ValueError):
        pirt.get_cubic_spline_coefs(0.44, 'unknown_spline_type')
        
    # Wrong cardinal spline tension
    with raises(ValueError):
        pirt.get_cubic_spline_coefs(0.44, -1.01)
    with raises(ValueError):
        pirt.get_cubic_spline_coefs(0.44, +1.01)
    
    # Our of range t is ok
    pirt.get_cubic_spline_coefs(-0.2, 0.0)
    pirt.get_cubic_spline_coefs(1.2, 0.0)
    
    # Iterate all existing splines
    for spline_type in ('nearest', 'linear', 'quadratic', 'lanczos',
                        'cardinal', 'basic', 'hermite', 'lagrange'):
        cc = pirt.get_cubic_spline_coefs(0.44, spline_type)
        assert len(cc) == 4
        assert 0.97 < sum(cc) < 1.03
    
    # We also do look up tables, for histroric reasons
    n = 1000
    lut = pirt.interp._cubic.get_lut(0.0, n)
    assert lut.shape == (4 * (n + 2), )
    

def test_meshgrid():
    
    res = pirt.meshgrid(2, 3)
    assert len(res) == 2
    assert res[0].dtype == 'float32'
    assert res[0].shape == (3, 2)
    assert res[1].shape == (3, 2)
    
    assert res[0].ravel().tolist() == [0, 1] * 3
    assert res[1].ravel().tolist() == [0, 0, 1, 1, 2, 2]
    
    res = pirt.meshgrid(4, 5)
    assert len(res) == 2
    assert res[0].dtype == 'float32'
    assert res[0].shape == (5, 4)
    assert res[1].shape == (5, 4)
    
    res = pirt.meshgrid(4, 5, 6)
    assert len(res) == 3
    assert res[0].dtype == 'float32'
    assert res[0].shape == (6, 5, 4)
    assert res[1].shape == (6, 5, 4)
    assert res[2].shape == (6, 5, 4)
    
    assert res[0].ravel().tolist() == [0, 1, 2, 3] * 5 * 6
    assert res[1].ravel().tolist() == ([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4) * 6
    assert res[2].ravel().tolist() == [0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20 + [5] * 20
    
    # Auto-conversions
    
    res2 = pirt.meshgrid([4, 5, 6])
    assert np.all(res[0] == res2[0])
    
    with raises(ValueError):
        pirt.meshgrid('meh')
    with raises(ValueError):
        pirt.meshgrid(3, 'meh')
    with raises(ValueError):
        pirt.meshgrid([3, 'meh'])
    
    print('meshgrid ok')


def test_make_samples_absolute():
    
    # 1D
    samples1 = np.array([0, 1, 0]), 
    samples2 = pirt.interp.make_samples_absolute(samples1)
    assert len(samples2) == 1
    assert list(samples2[0]) == [0, 2, 2]
    
    # 1D anisotropic
    samples1 = pirt.Aarray(np.array([0, 1, 0], np.float32), (2, )), 
    samples2 = pirt.interp.make_samples_absolute(samples1)
    assert len(samples2) == 1
    assert list(samples2[0]) == [0, 1.5, 2]
    
    # 1D anisotropic - note that origin is ignored
    samples1 = pirt.Aarray(np.array([0, 1, 0], np.float32), (0.5, ), (7, )), 
    samples2 = pirt.interp.make_samples_absolute(samples1)
    assert len(samples2) == 1
    assert list(samples2[0]) == [0, 3, 2]
    
    # 2D - wrong
    samples1 = np.array([0, 1, 0]), np.array([0, 0, 1])
    with raises(ValueError):
        pirt.interp.make_samples_absolute(samples1)
    
    # 2D
    samples1 = np.array([[0, 1, 0], [0, 1, 0]]), np.array([[0, 0, 0], [1, 1, 1]])
    samples2 = pirt.interp.make_samples_absolute(samples1)
    assert len(samples2) == 2
    assert list(samples2[0].flat) == [0, 2, 2,  0, 2, 2]
    assert list(samples2[1].flat) == [0, 0, 0,  2, 2, 2]
    
    # 3D
    samples1 = (np.array([[[0, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]]]),
                np.array([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]]]),
                np.array([[[1, 1, 1], [1, 1, 1]], [[0, 0, 0], [0, 0, 0]]]))
    samples2 = pirt.interp.make_samples_absolute(samples1)
    assert len(samples2) == 3
    assert list(samples2[0].flat) == [0, 2, 2,  0, 2, 2,  0, 2, 2,  0, 2, 2]
    assert list(samples2[1].flat) == [0, 0, 0,  2, 2, 2,  0, 0, 0,  2, 2, 2]
    assert list(samples2[2].flat) == [1, 1, 1,  1, 1, 1,  1, 1, 1,  1, 1, 1]


if __name__ == '__main__':
    run_tests_if_main()
