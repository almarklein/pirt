""" Approximate the exponent.
"""
import numpy as np
import visvis as vv

def near_exp(v):
    v = 1.0 + v / 256.0;
    v *= v; v *= v; v *= v; v *= v
    v *= v; v *= v; v *= v; v *= v
    return v


t = np.linspace(0, 0, 100)
x = [near_exp(i) for i in t]
y = [np.exp(i) for i in t]

vv.figure(1); vv.clf()
vv.plot(t, x, lc='r')
vv.plot(t, y, lc='g')
