import numpy as np
import visvis as vv

from pirt import SplineGrid
from pirt import interp
from visvis import Point, Pointset


class SplineByHand:
    """ SplineByHand()
    
    Demo application to influence a 1D spline grid using control points.
    """
    
    def __init__(self):
        
        # Setup visualization
        self._fig = fig = vv.figure()
        self._a1 = a1 = vv.subplot(111)
        
        # Init pointset with control points
        self._pp = Pointset(2)
        
        # Init lines to show control points
        self._line1 = vv.plot(self._pp, ls='', ms='.', mc='r', mw=11, axes=a1)
        self._line1.hitTest = True
        
        # Init grid properties
        self._fieldSize = 100
        self._sampling = 10
        self._levels = 5
        
        # Member to indicate the point being dragged (as an index)
        self._active = None
        
        # Bind to events
        a1.eventDoubleClick.Bind(self.on_doubleclick)
        self._line1.eventDoubleClick.Bind(self.on_doubleclick_line)
        self._line1.eventMouseDown.Bind(self.on_down_line)
        self._line1.eventMouseUp.Bind(self.on_up_line)
        a1.eventMotion.Bind(self.on_motion)
        fig.eventKeyDown.Bind(self.on_key_down)
        
        print('Use left/right to control #levels and up/down to control grid sampling.')
        
        # Init
        self.apply()
        a1.daspectAuto = False
        a1.SetLimits()
        a1.axis.showGrid = True
    
    
    def on_key_down(self, event):
        
        # Update level
        if event.key == vv.KEY_UP:
            self._sampling += 1
        elif event.key == vv.KEY_DOWN:
            self._sampling -= 1
        elif event.key == vv.KEY_RIGHT:
            self._levels += 1
        elif event.key == vv.KEY_LEFT:
            self._levels -= 1
        else:
            return
        
        # Correct
        if self._sampling < 1:
            self._sampling = 1
        
        # Apply and print
        print('Using B-spline grid with %i sampling and %i levels.' % (
                self._sampling, self._levels))
        self.apply()
    
    
    def on_doubleclick(self, event): # On axes
        
        # Get new point
        p = Point(event.x2d, event.y2d)
        
        # Add to pointset
        self._pp.append(p)
        self._line1.SetPoints(self._pp)
        
        # Apply
        self.apply()
    
    
    def on_doubleclick_line(self, event): # On axes
        
        # Get closest point
        dists = Point(event.x2d, event.y2d).distance(self._pp)
        I, = np.where( dists == dists.min() )
        if not len(I):
            return False
        
        # Remove from pointset
        self._pp.Pop(I[0])
        self._line1.SetPoints(self._pp)
        
        # Apply
        self._a1.Draw()
        self.apply()
    
    
    def on_down_line(self, event): # On line instance
        
        if event.button != 1:
            return False
        
        # Get closest point
        dists = Point(event.x2d, event.y2d).distance(self._pp)
        I, = np.where( dists == dists.min() )
        if not len(I):
            return False
        
        # Store
        self._active = I[0]
        
        # Prevent dragging by indicating the event needs no further handling
        return True
    
    
    def on_motion(self, event):
        
        if self._active is None:
            return False
        
        # Update line
        self._pp[self._active] = Point(event.x2d, event.y2d)
        self._line1.SetPoints(self._pp)
        # Draw
        self._a1.Draw()
    
    
    def on_up_line(self, event):
        
        if self._active is None:
            return False
        
        # Update point
        self.on_motion(event)
        
        # Deactivate
        self._active = None
        
        # Apply
        self.apply()
    
    
    def apply(self, event=None):
        
        # Get axes
        a1 = self._a1
        
        # Get sampling
        grid_sampling = self._sampling, self._sampling *2**self._levels
        
        # Create grid
        tmp = self._pp[:,0]
        tmp.shape = (tmp.size,1)
        pp = Pointset(tmp)
        grid1 = SplineGrid.from_points_multiscale((self._fieldSize,), grid_sampling,
                        pp.data, self._pp[:,1])
        
        # Get copy
        grid2 = grid1.copy()
        
        # Freeze edges
        self.freeze_edges(grid1)
        
#         # Create second grid
#         grid2 = SplineGrid.from_field(grid.get_field(), grid.grid_sampling)
#         grid3 = SplineGrid.from_field_multiscale(grid.get_field(), grid.grid_sampling)
        
        # Get grid points
        ppg1 = Pointset(2)
        ppg2 = Pointset(2)
        for gx in range(grid1.grid_shape[0]):
            ppg1.append( (gx-1)* grid1.grid_sampling, grid1.knots[gx] )
            ppg2.append( (gx-1)* grid2.grid_sampling, grid2.knots[gx] )
        
        # Get field
        field = grid1.get_field()
        #field2 = grid2.get_field()
        #field3 = grid3.get_field()
        
        # Delete objects in scene
        for ob in a1.wobjects:
            if ob is not self._line1 and not isinstance(ob, vv.axises.BaseAxis):
                ob.Destroy()
        
        # Draw
        vv.plot(ppg1, ls='', ms='x', mc='b', mw=9, axes=a1, axesAdjust=False)
        vv.plot(ppg2, ls='', ms='x', mc='c', mw=9, axes=a1, axesAdjust=False)
        vv.plot(np.arange(0, field.size), field, ls='-', lc='g', lw=3, axes=a1, axesAdjust=False)
        #vv.plot(field2, ls=':', lc=(0,0.5,0), lw=6, axes=a1, axesAdjust=False)
        #vv.plot(field3, ls=':', lc=(0,0.75,0), lw=6, axes=a1, axesAdjust=False)
    
    
    def freeze_edges(self, grid):
        
        # Store grid for debugging
        
        self._grid = grid
        
        # Freeze left edge
        grid.knots[1] = 0
        grid.knots[0] = -grid.knots[2]
        
#         c0, c1, c2, c3 = pirt.get_cubic_spline_coefs(0, 'B')
#         k1, k2, k3 = grid.knots[0], grid.knots[1], grid.knots[2]
#         grid.knots[0] = 0
#         grid.knots[1]=0
#         grid.knots[2]= )
        
        # Calculate t factor
        field_edge = (grid.field_shape[0]-1) * grid.field_sampling[0] 
        grid_edge = (grid.grid_shape[0]-4) * grid.grid_sampling
        t = (field_edge - grid_edge) / grid.grid_sampling
        
        # Get coefficients
        c0, c1, c2, c3 = interp.get_cubic_spline_coefs(t, 'B')
        
        # Freeze right edge
#         grid.knots[-3] = t*grid.knots[-3] # + (1-t)*0
#         k0, k1 = grid.knots[-4], grid.knots[-3]
#         grid.knots[-2] = t**2 * (k1-k0) + t*(2*k0-k1) - k0
#         k0, k1, k2, k3 = grid.knots[-4], grid.knots[-3], grid.knots[-2], grid.knots[-1]
#         grid.knots[-1] = -(k0*c0 + k1*c1 + k2*c2)/c3
        
        grid.knots[-3] = t*grid.knots[-3] # + (1-t)*0
        grid.knots[-1] = 0
        k0, k1 = grid.knots[-4], grid.knots[-3]
        grid.knots[-2] = -(k0*c0 + k1*c1)/c2


if __name__ == '__main__':
    v = SplineByHand()
    vv.use().Run()
