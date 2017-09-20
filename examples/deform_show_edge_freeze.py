"""
Example/test illustrating how the knots of the B-spline grid are modified
to freeze the edges of the underlying field.
"""

import pirt
import visvis as vv
import numpy as np

def _freeze_edges(self):
        
        def get_t_factor(grid, d):
            field_edge = (grid.field_shape[d]-1) * grid.field_sampling[d] 
            grid_edge = (grid.grid_shape[d]-4) * grid.grid_sampling
            return 1.0 - (field_edge - grid_edge) / grid.grid_sampling
        
        for d in [0]:
            grid = self
            
            # Check if grid is large enough
            if grid._knots.shape[d] < 6:
                grid._knots[:] = 0
                continue
            
            if d==0:
                # top/left
                grid._knots[1] = 0
                grid._knots[0] = -grid._knots[2]
                
                # Same approach as other side (t=0)
                # k1, k2, k3 = knots[0], knots[1], knots3
                grid._knots[0] = 0
                grid._knots[1] = - 0.25*grid._knots[2]
                
                # Get t factor and coefficients
                t = get_t_factor(grid, d)
                c1, c2, c3, c4 = pirt.get_cubic_spline_coefs(t, 'B')
                print('t', t, c1, c2, c3, c4)
                
                # bottom/right
                grid._knots[-3] = (1-t)*grid._knots[-3]
                grid._knots[-1] = 0
                k3, k4 = grid._knots[-3], grid._knots[-4]
                grid._knots[-2] = -(k3*c3 + k4*c4)/c2


grid1 = pirt.SplineGrid((33,),5)
grid1.knots[:] = 1
grid2 = grid1.copy()

_freeze_edges(grid2)

field1 = grid1.get_field() 
field2 = grid2.get_field()

# Get grid points
ppg1 = pirt.PointSet(2)
ppg2 = pirt.PointSet(2)
for gx in range(grid1.grid_shape[0]):
    ppg1.append( (gx-1)* grid1.grid_sampling, grid1.knots[gx] )
    ppg2.append( (gx-1)* grid2.grid_sampling, grid2.knots[gx] )


# Draw
f = vv.figure(1); vv.clf()
a = vv.gca()
vv.plot(np.arange(0, field2.size), field2, ls='-', lc='k', lw=3)
#vv.plot(ppg1, ls='', ms='.', mc='c', mw=10)
vv.plot(ppg2, ls='', ms='.', mc='g', mw=14)
a.axis.showGrid = 1
a.SetLimits(margin=0.1)
# a.legend = 'field', 'original knots', 'changed knots'
f.relativeFontSize = 1.4
f.position.h = 300
f.position.w = 600
