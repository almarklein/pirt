import pirt


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
    
    print('meshgrid ok')


if __name__ == '__main__':
    test_meshgrid()
