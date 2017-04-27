import time

import numpy as np
import pirt


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


def test_times():
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
    
    print('timings ok')


if __name__ == '__main__':
    test_warp1d()
    test_warp2d()
    test_warp3d()
    test_times()
