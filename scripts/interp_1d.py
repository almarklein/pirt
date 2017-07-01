"""
Demonstrate resizing of 1D signal, and also the extrapolation effect.
"""

import numpy as np
import visvis as vv
import pirt

# Create 1D signal
t0 = np.linspace(0,6,400).astype('float32')
a0 = np.sin(t0)+1

# Create decimated version
factor = 50
a1 = pirt.interp.zoom(a0, 1.0/factor, 3, prefilter=0)
t1 = np.arange(a1.origin[0], a1.origin[0] + a1.sampling[0]* a1.shape[0], a1.sampling[0])

# Create increased resolution
factor = 2
a2 = pirt.interp.zoom(a1, factor, 1, 0, extra=0)
a3 = pirt.interp.zoom(a1, factor, 3, 'C', extra=0)

t2 = np.arange(a2.origin[0], a2.origin[0] + a2.sampling[0]* a2.shape[0], a2.sampling[0])
t3 = np.arange(a3.origin[0], a3.origin[0] + a3.sampling[0]* a3.shape[0], a3.sampling[0])

# Create increased resolution and beyond
t9 = np.linspace(-2, a1.shape[0]+1, 100).astype('float32')
a9 = pirt.interp.warp(a1, (t9,), 3) 

# Visualize
fig = vv.figure(2); vv.clf()
fig.position  = 140.00, 205.00,  560.00, 560.00
vv.subplot(211)
vv.plot(a0, ls=':', lc='k', lw=2)
vv.plot(t1, a1, ms='x', mw=9, ls='', mc='k')
#
vv.plot(t2, a2, ms='.', ls='-', lc='r', mc='r')
vv.plot(t3, a3, ms='.', ls='-', lc='g', mc='g')
#
vv.gca().axis.showGrid = True
vv.title('Demonstrate resizing')

vv.subplot(212)
vv.plot(range(a1.shape[0]),a1, ms='x', mw=9, ls='', mc='k')
vv.plot(t9, a9, ms='', ls='-', lw=2, lc='g')
vv.gca().axis.showGrid = True
vv.title('Show that interpolation also does a bit of extrapolation.')

vv.use().Run()
