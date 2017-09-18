import time

import numpy as np
import pirt

from pirt.testing import raises, run_tests_if_main


## Warp

def test_warp1d():
    
    data = np.array([10, 21, 31, 40, 50, 60, 70]).astype('float32')
    
    # Sampling on data positions
    for order in (0, 1, 2, 3):
        result = pirt.warp(data, np.array([1, 4, 3, 2, 5]), order)
        assert result.tolist() == [21, 50, 40, 31, 60]
    
    # Sampling in between
    samplesx = np.array([3.4])
    val0 = pirt.warp(data, samplesx, 0)[0]
    val1 = pirt.warp(data, samplesx, 1)[0]
    val2 = pirt.warp(data, samplesx, 2)[0]
    val3 = pirt.warp(data, samplesx, 3)[0]
    
    assert val0 == 40
    assert val1 == 44
    assert val2 != val1 and val3 != val1 and val3 != val2
    assert 43 < val2 < 44
    assert 43 < val3 < 44
    
    # Sampling the sides
    # This tests against a regression where int() was used instead of floor
    result = pirt.warp(data, np.array([-0.6, -0.4, -0.1, 0.1, 0.4]), 0)
    assert result.tolist() == [0] + [10] * 4
    for order in (1, 2, 3):
        result = pirt.warp(data, np.array([-0.6, -0.4, -0.1, 0.1, 0.4]), order)
        assert result[0] == 0
        assert 5 <= result[1] <= 10
        assert 8.5 <= result[2] <= 11.5
        assert 10 <= result[3] <= 15
    
    print('warp 1d ok')


def test_warp2d():
    
    data = np.array([[10, 21, 31], [40, 50, 60], [70, 80, 90]]).astype('float32')
    
    # Sampling on data positions 1D
    for order in (0, 1, 2, 3):
        result = pirt.warp(data, (np.array([1, 2, 1]), np.array([0, 1, 0])), order)
        assert result.tolist() == [21, 60, 21]
    
    # Sampling on data positions 2D
    for order in (0, 1, 2, 3):
        result = pirt.warp(data, (np.array([[1, 2], [1, 2]]), np.array([[0, 1], [2, 2]])), order)
        assert result.tolist() == [[21, 60], [80, 90]]
    
    # Sampling in between
    samplesx = np.array([1.5])
    samplesy = np.array([0.5])
    val0 = pirt.warp(data, (samplesx, samplesy), 0).tolist()[0]
    val1 = pirt.warp(data, (samplesx, samplesy), 1).tolist()[0]
    val2 = pirt.warp(data, (samplesx, samplesy), 2).tolist()[0]
    val3 = pirt.warp(data, (samplesx, samplesy), 3).tolist()[0]
    assert val0 == 60
    assert val1 == (21 + 31 + 50 + 60) / 4
    assert val2 != val1 and val3 != val1  # and val3 != val2 - edge-logic cause them to be the same
    assert 35 < val2 < 40
    assert 35 < val3 < 40
   
    print('warp 2d ok')


def test_warp3d():
    
    data = sum(pirt.meshgrid([1, 2, 3], [10, 20, 30], [100, 200, 300]))
    
    # Sampling on data positions 1D
    for order in (0, 1, 2, 3):
        result = pirt.warp(data, (np.array([1, 2, 1]),
                                  np.array([0, 1, 0]),
                                  np.array([2, 1, 0])), order)
        assert result.tolist() == [312, 223, 112]
    
    # Sampling on data positions 2D
    for order in (0, 1, 2, 3):
        result = pirt.warp(data, (np.array([[1, 2], [1, 2]]),
                                  np.array([[0, 1], [2, 2]]),
                                  np.array([[0, 1], [2, 2]])), order)
        assert result.tolist() == [[112, 223], [332, 333]]
    
    # Sampling in between
    data[0, 0, 0] += 10
    data[1, 0, 0] += 10
    samplesx = np.array([1.5])
    samplesy = np.array([0.5])
    samplesz = np.array([0.5])
    val0 = pirt.warp(data, (samplesx, samplesy, samplesz), 0).tolist()[0]
    val1 = pirt.warp(data, (samplesx, samplesy, samplesz), 1).tolist()[0]
    val2 = pirt.warp(data, (samplesx, samplesy, samplesz), 2).tolist()[0]
    val3 = pirt.warp(data, (samplesx, samplesy, samplesz), 3).tolist()[0]
    assert val0 == 223
    assert val1 == (112 + 113 + 122 + 123 + 212 + 213 + 222 + 223) / 8
    assert val2 != val1 and val3 != val1  # and val3 != val2
    assert 155 < val2 < 160
    assert 155 < val3 < 160
    
    print('warp 3d ok')


## Project

def test_project1d():
    
    data = np.array([0, 9, 0, 0, 0]).astype('float32')
    
    # No change
    result = pirt.project(data, np.array([0, 1, 2, 3, 4]))
    assert result.tolist() == [0, 9, 0, 0, 0]
    assert result.shape == (5, )
    
    # Project out of bounds
    result = pirt.project(data, np.array([99, 99, 99, 3, 4]))
    assert result.tolist() == [0, 0, 0, 0, 0]
    
    # Project to other spot
    result = pirt.project(data, np.array([0, 3, 2, 3, 4]))
    assert result[1] < 9
    assert result[3] >= 4.5  # loads of splatting
    
    print('project 1d ok')


def test_project2d():
    
    data = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]]).astype('float32')
    
    # No change
    result = pirt.project(data, (np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
                                 np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])))
    assert result.tolist() == data.tolist()
    assert result.shape == (3, 3)
    
    # Project away
    result = pirt.project(data, (np.array([[0, 1, 2], [0, 1, 2], [0, 1, -99]]),
                                 np.array([[0, 0, 0], [1, 1, 1], [2, 2, -99]])))
    assert result.max() == 80
    
    # Project to other spot
    result = pirt.project(data, (np.array([[0, 1, 2], [0, 1, 2], [0, 1, 0]]),
                                 np.array([[0, 0, 0], [1, 1, 1], [2, 2, 0]])))
    assert result[0, 0] >= 50
    
    print('project 2d ok')


def test_project3d():
    
    samplesx, samplesy, samplesz = pirt.meshgrid([0, 1, 2], [0, 1, 2], [0, 1, 2])
    data = sum(pirt.meshgrid([1, 2, 3], [10, 20, 30], [100, 200, 300]))
    
    # no change
    result = pirt.project(data, (samplesx, samplesy, samplesz))
    assert result.tolist() == data.tolist()
    assert result.shape == (3, 3, 3)
    
    # Project away
    samplesx[-1, -1, -1] = -99
    samplesy[-1, -1, -1] = -99
    samplesz[-1, -1, -1] = -99
    result = pirt.project(data, (samplesx, samplesy, samplesz))
    assert result.max() < 333
    
    # Project to other spot
    samplesx[-1, -1, -1] = 0
    samplesy[-1, -1, -1] = 0
    samplesz[-1, -1, -1] = 0
    result = pirt.project(data, (samplesx, samplesy, samplesz))
    assert result[0, 0, 0] >= 200
    
    print('project 3d ok')


## Misc


def test_awarp():
    
    data = np.array([[10, 21, 31], [40, 50, 60], [70, 80, 90]]).astype('float32')
    samples = np.array([1, 2, 1]) * 2, np.array([0, 1, 0]) / 2
    order = 1
    
    # Cannot use data like this
    with raises(ValueError):
        pirt.awarp(data, samples, order)
    
    # Data must have sampling and origin
    data = pirt.Aarray(data, (0.5, 2.0))
    
    # Check that using normal warp fails
    result = pirt.warp(data, samples, order)
    assert result.tolist() != [21, 60, 21]
    
    result = pirt.awarp(data, samples, order)
    assert result.tolist() == [21, 60, 21]
    

def test_aproject():
    
    data = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]]).astype('float32')
    samples = (np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]) * 2,
               np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]) / 2)
    
    # Cannot use data like this
    with raises(ValueError):
        pirt.aproject(data, samples)
    
    # Data must have sampling and origin
    data = pirt.Aarray(data, (0.5, 2.0))
    
    # Check that using normal project fails
    result = pirt.project(data, samples)
    assert result.tolist() != data.tolist()

    result = pirt.aproject(data, samples)
    assert result.tolist() == data.tolist()


def test_warp_fails_and_conversions():
    
    # Prepare data
    data = np.array([[10, 21, 31], [40, 50, 60], [70, 80, 90]]).astype('float32')
    samples = np.array([1, 2, 1]), np.array([0, 1, 0])
    order = 1
    
    # Default
    result = pirt.warp(data, samples, order)
    assert result.tolist() == [21, 60, 21]
    
    # data argument
    
    # Wrong type
    with raises(ValueError):
        pirt.warp('not_array', samples, order)
    
    # Wrong shape
    with raises(ValueError):
        pirt.warp(data.reshape(-1, 1, 1, 1), samples, order)
    
    # samples argument
    
    # Samples as list -> tuple
    result = pirt.warp(data, list(samples), order)
    assert result.tolist() == [21, 60, 21]
    
    # Samples as one nd-array (skimage api)
    result = pirt.warp(data, np.stack(reversed(samples)), order)
    assert result.tolist() == [21, 60, 21]
    
    # Wrong type
    with raises(ValueError):
        pirt.warp(data, 'wrong', order)
    
    # Inside samples
    samples2 = (np.array([1, 2, 1]), )
    with raises(ValueError):
        pirt.warp(data, samples2, order)
    samples2 = np.array([1, 2, 1]), 'meh'
    with raises(ValueError):
        pirt.warp(data, samples2, order)
    samples2 = np.array([1, 2, 1]), np.array([0, 1, 0, 2])
    with raises(ValueError):
        pirt.warp(data, samples2, order)
    
    # order argument
    
    # Wrong type
    with raises(ValueError):
        pirt.warp(data, samples, [0])
    
    # Wrong text
    with raises(ValueError):
        pirt.warp(data, samples, 'unknown order')
    
    # Wrong int
    with raises(ValueError):
        pirt.warp(data, samples, 4)


def test_project_fails_and_conversions():
    
    data = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]]).astype('float32')
    samples = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    
    # default
    result = pirt.project(data, samples)
    assert result.tolist() == data.tolist()
    
    # data argument
    
    # Wrong type
    with raises(ValueError):
        pirt.project('not_array', samples)
    
    # Wrong shape
    with raises(ValueError):
        pirt.project(data.reshape(-1, 1, 1, 1), samples)
    
    # samples argument
    
    # Samples as list -> tuple
    result = pirt.project(data, list(samples))
    assert result.tolist() == data.tolist()
    
    # Samples as one nd-array (skimage api)
    result = pirt.project(data, np.stack(reversed(samples)))
    assert result.tolist() == data.tolist()
    
    # Wrong type
    with raises(ValueError):
        pirt.project(data, 'wrong')
    
    # inside samples - project is more restrictive than warp
    samples = (np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), )
    with raises(ValueError):
        pirt.project(data, samples)
    samples = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), 'meh'
    with raises(ValueError):
        pirt.project(data, samples)
    samples = np.array([[0, 1, 2], [0, 1, 2]]), np.array([[0, 0, 0], [1, 1, 1]])
    with raises(ValueError):
        pirt.project(data, samples)


## Timings

def test_times_warp():
    # This is to test that all funcs are connectly numba-decorated.
    
    data1 = np.arange(27).astype('float32')
    data2 = data1.reshape((3, 9))
    data3 = data1.reshape((3, 3, 3))
    
    samplesx = samplesy = samplesz = np.ones(10000, ) * 1.1
    
    # Warm up
    pirt.warp(data1, (samplesx, ))
    pirt.warp(data2, (samplesx, samplesy))
    pirt.warp(data3, (samplesx, samplesy, samplesz))
    
    # 1D
    t0 = time.time()
    pirt.warp(data1, (samplesx, ))
    te = time.time() - t0
    print(te)
    assert te < 0.01
    
    # 2D
    t0 = time.time()
    pirt.warp(data2, (samplesx, samplesy))
    te = time.time() - t0
    print(te)
    assert te < 0.01
    
    # 3D
    t0 = time.time()
    pirt.warp(data3, (samplesx, samplesy, samplesz))
    te = time.time() - t0
    print(te)
    assert te < 0.01
    
    print('warp timings ok')


def test_times_project():
    # This is to test that all funcs are connectly numba-decorated.
    
    data1 = np.arange(9000).astype('float32')
    data2 = data1.reshape((1000, 9))
    data3 = data1.reshape((1000, 3, 3))
    
    samplesx = samplesy = samplesz = np.ones(1000*9, ) * 1.1
    
    # Warm up
    pirt.project(data1, (samplesx, ))
    pirt.project(data2, (samplesx.reshape(1000, 9), samplesy.reshape(1000, 9)))
    pirt.project(data3, (samplesx.reshape(1000, 3, 3), samplesy.reshape(1000, 3, 3), samplesz.reshape(1000, 3, 3)))
    
    # 1D
    t0 = time.time()
    pirt.project(data1, (samplesx, ))
    te = time.time() - t0
    print(te)
    assert te < 0.01
    
    # 2D
    t0 = time.time()
    pirt.project(data2, (samplesx.reshape(1000, 9), samplesy.reshape(1000, 9)))
    te = time.time() - t0
    print(te)
    assert te < 0.01
    
    # 3D
    t0 = time.time()
    pirt.project(data3, (samplesx.reshape(1000, 3, 3), samplesy.reshape(1000, 3, 3), samplesz.reshape(1000, 3, 3)))
    te = time.time() - t0
    print(te)
    assert te < 0.01
    
    print('project timings ok')


if __name__ == '__main__':
    test_awarp()
    test_aproject()
    test_warp_fails_and_conversions()
    test_project_fails_and_conversions()
    # run_tests_if_main()
