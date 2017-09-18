# This touches the code of the high level interp functions, but does
# not really test the outcome; we've pretty much got that covered in
# the other tests (or that's what we assume).
 
import numpy as np
import pirt
from pirt.testing import raises, run_tests_if_main


def test_deform():
   
    im = np.zeros((64, 64), np.float32)
    im = pirt.Aarray(im, (1, 2))
    
    deltas = np.zeros((64, 64), np.float32), np.zeros((64, 64), np.float32)
    
    result = pirt.deform_backward(im, deltas)
    assert result.shape == im.shape
    
    result = pirt.deform_forward(im, deltas)
    assert result.shape == im.shape
    
    with raises(ValueError):
        pirt.deform_backward(im, deltas[1:])
    with raises(ValueError):
        pirt.deform_forward(im, deltas[1:])
    
    
def test_resize():
    
    im = np.zeros((64, 64), np.float32)
    im = pirt.Aarray(im, (1, 2))
    
    im2 = pirt.imresize(im, (50, 50))
    assert im2.shape == (50, 50)
    
    im2 = pirt.resize(im, (50, 50))  # extra=False
    assert im2.shape == (50, 50)
    
    # Raises 
    with raises(ValueError):
        pirt.resize(im, 'meh')
    with raises(ValueError):
        pirt.resize(im, (3, 3, 3))


def test_zoom():
    
    im = np.zeros((64, 64), np.float32)
    im = pirt.Aarray(im, (1, 2))
    
    im3 = pirt.imzoom(im, 0.5)
    assert im3.shape == (32, 32)

    im3 = pirt.imzoom(im, np.array(0.25))
    assert im3.shape == (16, 16)
    
    # Raises 
    with raises(ValueError):
        pirt.zoom(im, 'meh')
    with raises(ValueError):
        pirt.zoom(im, (3, 3, 3))


run_tests_if_main()
