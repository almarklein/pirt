"""
Demo app that lets you double-click to add points, which can be dragged.
The knots of the underlying grid that tries to "follow" the points are shown,
and its sampling and multi-scale approach can be controlled using the
up/down and left/right keys.
"""

from pirt.apps.spline_by_hand import SplineByHand
import visvis as vv

app = SplineByHand()

vv.use().Run()
