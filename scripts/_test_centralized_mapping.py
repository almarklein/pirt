import time
import numpy as np
import pirt
from pirt import SplineGrid, FieldDescription
from pirt import ( DeformationGridForward, DeformationFieldForward,
                    DeformationGridBackward, DeformationFieldBackward)
from pirt import Point, Pointset, Aarray

try:
    import visvis as vv
except ImportError:
    vv = None


class SplineByHand:
    """ SplineByHand()
    
    Demo application to influence a 1D spline grid using control points.
    
    """
    
    def __init__(self):
        
        # Check visvis
        if vv is None:
            raise RuntimeError('Require visvis.')
        
        # Setup visualization
        self._fig = fig = vv.figure()
        self._a1 = a1 = vv.subplot(111)
        
        # Init pointset with control points
        self._pp = Pointset(2)
        
        # Init lines to show control points
        self._line1 = vv.plot(self._pp, ls='', ms='.', mc='r', mw=11, axes=a1)
        self._line1.hitTest = True
        
        # Init grid properties
        self._spline_type = 'B'
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
        #
        elif event.text.upper() == 'B':
            self._spline_type = 'B'
        elif event.text.upper() == 'C':
            self._spline_type = 'C'
        elif event.text.upper() == 'L':
            self._spline_type = 'linear'
        #
        else:
            return
        
        # Correct
        if self._sampling < 1:
            self._sampling = 1
        
        # Apply and print
        print 'Using %s grid with %i sampling and %i levels.' % (
                self._spline_type, self._sampling, self._levels)
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
                        pp, self._pp[:,1], spline_type=self._spline_type )
        
        # Get copy
        grid2 = grid1.copy()
        
        # Freeze edges
        self.freeze_edges(grid1)
        
#         # Create second grid
#         grid2 = SplineGrid.from_field(grid.get_field(), grid.grid_sampling, 
#             spline_type=self._spline_type)
#         grid3 = SplineGrid.from_field_multiscale(grid.get_field(), grid.grid_sampling,
#             spline_type=self._spline_type)
        
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
        c0, c1, c2, c3 = pirt.get_cubic_spline_coefs(t, 'B')
        
        # Freeze right edge
#         grid.knots[-3] = t*grid.knots[-3] # + (1-t)*0
#         k0, k1 = grid.knots[-4], grid.knots[-3]
#         grid.knots[-2] = t**2 * (k1-k0) + t*(2*k0-k1) - k0
#         k0, k1, k2, k3 = grid.knots[-4], grid.knots[-3], grid.knots[-2], grid.knots[-1]
#         grid.knots[-1] = -(k0*c0 + k1*c1 + k2*c2)/c3
        
        grid.knots[-3] = t*grid.knots[-3] # + (1-t)*0
        grid.knots[-1] = 0
        k0, k1, k2 = grid.knots[-4], grid.knots[-3], grid.knots[-2]
        grid.knots[-2] = -(k0*c0 + k1*c1)/c2


class DeformByHand:
    """ DeformByHand(im, grid_sampling=40)
    
    Demo application to deform a 2D image by hand using a spline grid.
    
    Use the grid property to obtain the deformation grid.
    Use the run() method to wait for the user to close the figure.
    
    
    """
    
    def __init__(self, im, grid_sampling=40):
        
        # Check visvis
        if vv is None:
            raise RuntimeError('Require visvis.')
        
        # Store image
        self._im = im
        
        # Setup visualization
        self._fig = fig = vv.figure()
        self._a1 = a1 = vv.subplot(231);
        self._a2 = a2 = vv.subplot(232);
        self._a3 = a3 = vv.subplot(233);
        self._a4 = a4 = vv.subplot(234);
        self._a5 = a5 = vv.subplot(235);
        self._a6 = a6 = vv.subplot(236);
        
        # Text objects
        self._text1 = vv.Label(fig)
        self._text1.position = 5, 2
        self._text2 = vv.Label(fig)
        self._text2.position = 5, 20
        
        # Move axes 
        a1.parent.position = 0.0, 0.1, 0.33, 0.45
        a2.parent.position = 0.33, 0.1, 0.33, 0.45
        a3.parent.position = 0.66, 0.1, 0.33, 0.45
        a4.parent.position = 0.0, 0.55, 0.33, 0.45
        a5.parent.position = 0.33, 0.55, 0.33, 0.45
        a6.parent.position = 0.66, 0.55, 0.33, 0.45
        
        # Correct axes, share camera
        cam = vv.cameras.TwoDCamera()
        for a in [a1, a2, a3, a4, a5, a6]:
            a.axis.visible = False
            a.camera = cam
        
        # Show images
        im0 = im*0
        self._t1 = vv.imshow(im, axes=a1)
        self._t2 = vv.imshow(im, axes=a2)
        self._t3 = vv.imshow(im, axes=a3)
        self._t4 = vv.imshow(im0, axes=a4)
        self._t5 = vv.imshow(im0, axes=a5)
        self._t6 = vv.imshow(im0, axes=a6)
        
        # Init pointsets
        self._pp1 = Pointset(2)
        self._pp2 = Pointset(2)
        self._active = None
        self._lines = []
        
        # Init lines to show all deformations
        tmp = vv.Pointset(2)        
        self._line1 = vv.plot(tmp, ls='', ms='.', mc='c', axes=a2)
        self._line2 = vv.plot(tmp, ls='+', lc='c', lw='2', axes=a2)
        
        # Init grid properties
        self._spline_type = 'Basic'
        self._sampling = grid_sampling
        self._levels = 5
        self._multiscale = True
        self._injective = 0.5
        self._frozenedge = 1
        self._forward = True
        
        # Init grid
        self.DeformationField = DeformationFieldForward
        self._field1 = self.DeformationField(FieldDescription(self._im))
        self._field2 = self.DeformationField(FieldDescription(self._im))
        
        # Bind to events
        a2.eventMouseDown.Bind(self.on_down)
        a2.eventMouseUp.Bind(self.on_up)
        a2.eventMotion.Bind(self.on_motion)
        fig.eventKeyDown.Bind(self.on_key_down)
        #a1.eventDoubleClick.Bind(self.OnDone)
        
        # Apply
        self.apply()
    
    
    def on_key_down(self, event):
        
        spline_types ={'Basic':'Cardinal', 'Cardinal':'Linear', 'Linear':'Basic'}
        
        # Update level
        if event.key == vv.KEY_UP:
            self._sampling += 2
        elif event.key == vv.KEY_DOWN:
            self._sampling -= 2
        elif event.key == vv.KEY_RIGHT:
            self._levels += 1
        elif event.key == vv.KEY_LEFT:
            self._levels -= 1
        #
        elif event.text.upper() == 'S':
            self._spline_type = spline_types[self._spline_type] # Next!
        elif event.text.upper() == 'M':
            self._multiscale = not self._multiscale
        elif event.text.upper() == 'I':
            self._injective += 0.4
            if self._injective > 0.8:
                self._injective = -0.8
        elif event.text.upper() == 'E':
            self._frozenedge = not self._frozenedge
        elif event.text.upper() == 'F':
            self._forward = not self._forward
            # reset global field
            if self._forward:
                self.DeformationField = DeformationFieldForward
            else:
                self.DeformationField = DeformationFieldBackward
            self._field1 = self.DeformationField(FieldDescription(self._im))
        #
        elif event.key == vv.KEY_ESCAPE:
            self._pp1.clear()
            self._pp2.clear()
            self.apply()
        elif event.text == ' ':
            self.apply_deform()
        #
        else:
            return
        
        # Correct
        if self._sampling < 1:
            self._sampling = 1
        
        self.apply()
    
    
    def on_down(self, event):
        
        if event.button != 1:
            return False        
        if not vv.KEY_SHIFT in event.modifiers:
            return False
        
        # Store location
        self._active = Point(event.x2d, event.y2d)
        
        # Clear any line object
        for l in self._lines:
            l.Destroy()
        
        # Create line objects
        tmp = Pointset(2)
        tmp.append(self._active)
        tmp.append(self._active)
        l1 = vv.plot(tmp, lc='g', lw='3', axes=self._a2, axesAdjust=0)
        l2 = vv.plot(tmp[:1], ls='', ms='.', mc='g', axes=self._a2, axesAdjust=0)
        self._lines = [l1, l2]
        
        # Draw
        self._a2.Draw()
        
        # Prevent dragging by indicating the event needs no further handling
        return True
    
    
    def on_motion(self, event):
        if self._active and self._lines:
            # Update line
            tmp = Pointset(2)
            tmp.append(self._active)
            tmp.append(event.x2d, event.y2d)
            l1 = self._lines[0]
            l1.SetPoints(tmp)
            # Draw
            self._a2.Draw()
    
    
    def on_up(self, event):
        
        if self._active is None:
            return False
        
        # Get points
        p1 = self._active
        p2 = Point(event.x2d, event.y2d)
        
        # Add!
        self._pp1.append(p1)
        self._pp2.append(p2)
        
        # We're done with this one
        self._active = None
        
        # Clear any line object
        for l in self._lines:
            l.Destroy()
        
        # Apply
        self.apply()
    
    
    def apply_deform(self):
        # Apply current point-wise deformation
        
        # Compose deformations
        self._field1 = self._field1.compose(self._field2)
        
        # Clear points
        self._pp1.clear()
        self._pp2.clear()
        
        # Update
        self.apply()
    
    
    def apply(self):
        
        # Get sampling
        grid_sampling = self._sampling, self._sampling*2**self._levels
        
        # Init field and deform
        if not self._pp1:
            # Unit deform
            deform = self.DeformationField(FieldDescription(self._im))
        elif self._multiscale:
            deform = self.DeformationField.from_points_multiscale(self._im, grid_sampling, 
                        self._pp1, self._pp2,
                        injective=self._injective, frozenedge=self._frozenedge,
                        spline_type=self._spline_type)
        else:
            DeformationGrid = DeformationGridForward
            if not self._forward:
                DeformationGrid = DeformationGridBackward
            grid = DeformationGrid.from_points(self._im, self._sampling, 
                        self._pp1, self._pp2, 
                        injective=self._injective, frozenedge=self._frozenedge,
                        spline_type=self._spline_type)
            deform = grid.as_deformation_field()
        
        # Store grid
        field0 = self.DeformationField(FieldDescription(self._im))
        self._field2 = deform
        field3 = self._field1.compose(self._field2)
        
        # Deform
        im2 = self._field1.apply_deformation(self._im)
        #
        im3 = pirt.reg.reg_base.create_grid_image(self._im)*250 #self._im1
        t0 = time.time()
        im3_1 = field3.apply_deformation(im3)
        t1 = time.time()
        im3_2 = apply_deformation(field3, im3)
        t2 = time.time()
        im3 = np.zeros(im3_1.shape+(3,),'float32')
        im3[:,:,0] = im3_1
        im3[:,:,1] = im3_2
        
        print '%1.5f s and %1.5f s' % (t1-t0, t2-t1)
        
        # Update imagesf
        self._t2.SetData(im2)
        self._t3.SetData(im3)
        
        # Update grids
        for a, field in zip(    [self._a4, self._a5, self._a6],
                                [field0, self._field1, field3] ):
            a.Clear()
            field.show(a, False)
            a.axis.visible = False
        
        # Update lines
        tmp = Pointset(2)
        for i in range(len(self._pp1)):            
            tmp.append(self._pp1[i])
            tmp.append(self._pp2[i])
        self._line1.SetPoints(self._pp1)
        self._line2.SetPoints(tmp)
        
        # Draw
        self._a2.Draw()
        
        # Show text
        text1 = '%s spline (S) with sampling %i (U/D) and %i levels (L/R).' % (
            self._spline_type, self._sampling, self._levels)
        
        text2 = 'Multiscale %i (M), injective %1.1f (I), frozen edge %i (E), forward %i (F).' % (
            self._multiscale, self._injective, self._frozenedge, self._forward)
        
        # Display and apply
        self._text1.text = text1
        self._text2.text = text2
    
    
    @property
    def field(self):
        return self._field1
    
    
    def run(self):
        
        # Setup detecting closing of figure
        self._closed = False
        def callback(event):
            self._closed = True
        self._fig.eventClose.Bind(callback)
        
        while not self._closed:
            time.sleep(0.02)
            vv.processEvents()


def apply_deformation(self, data, interpolation=3):
    
    # Null deformation
    if self.is_identity:
        return data
    
    # Make sure it is a deformation field
    deform = self.as_deformation_field()
    
    # Need upsampling?
    deform = deform.resize_field(data)
    
    # Reverse (from z-y-x to x-y-z)
    samples = [s for s in reversed(deform)]
    
    # Deform!
    mi = int(not self.forward_mapping)
    result = deform_data(data, samples, interpolation, mapping_index=mi)
    
    # Make Aarray and return
    result = Aarray(result, deform.field_sampling)
    return result



def deform_data(data, deltas, order=1, mapping_index=1, spline_type=0.0):
    
    from pirt.interpolation_ import make_samples_absolute, interp, project
    from pirt import is_Aarray, Aarray
    
    # Check
    if len(deltas) != data.ndim:
        tmp = "Samples must contain as many arrays as data has dimensions."
        raise ValueError(tmp)
    
    if mapping_index==0:
        samples = make_samples_absolute([d*1.0 for d in deltas])
        #deltas = [pirt.project(-delta, samples) for delta in deltas]
        #deltas = [pirt.interpolation_.project2_32(-delta, *samples) for delta in deltas]
        deltas = pirt.interpolation_.project22_32(
                        -deltas[0], -deltas[1], samples[0], samples[1])
        #samples = [delta for delta in reversed(deltas)]
        #deltas = [pirt.deform_forward(-delta, samples) for delta in deltas]
    
    # Interpolation in data
    samples = make_samples_absolute([delta for delta in deltas])
    result = interp(data, samples, order, spline_type)
    #samples = [delta for delta in reversed(deltas)]
    #result = pirt.deform_backward(data, samples, order, spline_type)
    
    # Make Aarray
    if is_Aarray(data):
        result = Aarray(result, data.sampling, data.origin)
    
    # Done
    return result


def deform_data5(data, deltas, order=1, mapping_index=1, spline_type=0.0):
    
    from pirt.interpolation_ import make_samples_absolute, interp
    from pirt import is_Aarray, Aarray
    
    # Check
    if len(deltas) != data.ndim:
        tmp = "Samples must contain as many arrays as data has dimensions."
        raise ValueError(tmp)
    
    # Make delta small
    N = 17
    if mapping_index==1:
        N=0
    elif 0 and mapping_index==0:
        N=0
    else:
        factor = mapping_index - 1.0
    
    # Make deltas in pixels
    sam = deltas[0].sampling
    sam1 = [1.0 for i in sam]
    deltas = [Aarray(-deltas[d]*(1.0/sam[d]), sam1) for d in range(len(sam))]
    deltas2 = deltas
    
    # Recurse
    loc0 = make_samples_absolute([d*0.0 for d in deltas])    
    loc1 = make_samples_absolute(deltas)
    locd = None
    for i in range(N):
        # Get sample locations
        if locd is not None:
            loc1 = [l1+0.5*l2 for l1,l2 in zip(loc1,locd)]
        # Sample new deltas: deltas2 = deltas(loc2)
        deltas2 = [interp(delta, loc1, 1) for delta in deltas]
        # loc_est = loc2 + deltas(loc2)
        loc_est = [l1-l2 for l1,l2 in zip(loc1,deltas2)]
        # Get error in found vector
        locd = [l2-l1 for l1,l2 in zip(loc_est,loc0)]
    
    samples = make_samples_absolute([delta for delta in deltas2])
    
    # Final interpolation
    result = interp(data, samples,1, order, spline_type)
    
    # Make Aarray
    if is_Aarray(data):
        result = Aarray(result, data.sampling, data.origin)
    
    # Done
    return result

def deform_data6(data, deltas, order=1, mapping_index=1, spline_type=0.0):
    
    from pirt.interpolation_ import make_samples_absolute, interp
    from pirt import is_Aarray, Aarray
    
    # Check
    if len(deltas) != data.ndim:
        tmp = "Samples must contain as many arrays as data has dimensions."
        raise ValueError(tmp)
    
    # Make delta small
    N = 17
    if mapping_index==1:
        N=0
    elif 0 and mapping_index==0:
        N=0
    else:
        factor = mapping_index - 1.0
    
    # Make deltas in pixels
    sam = deltas[0].sampling
    sam1 = [1.0 for i in sam]
    deltas = [Aarray(-deltas[d]*(1.0/sam[d]), sam1) for d in range(len(sam))]
    deltas2 = deltas
    
    # loc0 is original position (identity)
    # loc1 is sample position to get vector
    # loc2 is loc1 + that vector
    # loc3 is loc2 + vector sampled at loc2
    
    # Recurse
    loc0 = make_samples_absolute([d*0.0 for d in deltas])    
    loc1 = loc0
#    loc1 = make_samples_absolute(deltas)
    locd = None
    for i in range(N):
        # Sample new deltas: deltas2 = deltas(loc2)
        if locd is not None:
            loc1 = [l1+0.5*l2 for l1,l2 in zip(loc1,locd)]
        loc2 = [loc+interp(delta, loc1, 1) for loc,delta in zip(loc1, deltas)]
        # Sample new deltas: deltas2 = deltas(loc2)
        deltas2 = [interp(delta, loc2, 1) for delta in deltas]
        # loc_est = loc2 + deltas(loc2)
        loc3 = [l1-l2 for l1,l2 in zip(loc2,deltas2)]
        # Get error in found vector
        locd = [l2-l1 for l1,l2 in zip(loc3,loc0)]
    
    samples = make_samples_absolute([delta for delta in deltas2])
    
    # Final interpolation
    result = interp(data, samples,1)#, order, spline_type)
    
    # Make Aarray
    if is_Aarray(data):
        result = Aarray(result, data.sampling, data.origin)
    
    # Done
    return result
    
if __name__ == '__main__':
#     v = SplineByHand()

    d = None
    d = DeformByHand(vv.imread('lena.png')[:,:,2].astype('float32'))
    
    WHAT = 'twist'
    
    if not d:
        pass    
    elif WHAT == 'twist':
        d._pp1.append(108.27,       416.57 )
        d._pp1.append(330.78,       385.58 )
        d._pp1.append(220.08,       393.32 )
        d._pp1.append(406.06,       266.02 )
        d._pp1.append(389.45,       186.31 )
        d._pp1.append(347.38,        122.1 )
        d._pp1.append(401.63,       332.44 )
        d._pp1.append(290.92,       395.54 )
        #
        d._pp2.append(249.96,       449.78 )
        d._pp2.append(414.91,       333.55 )
        d._pp2.append(295.35,       363.44 )
        d._pp2.append(409.38,       207.34 )
        d._pp2.append(351.81,       88.892 )
        d._pp2.append(272.11,       68.966 )
        d._pp2.append(404.95,       280.41 )
        d._pp2.append(316.39,        388.9 )

    elif WHAT == 'outward':
        # outward
        d._pp1.append(265,265)
        d._pp2.append(225,265)
        d._pp1.append(328,267)
        d._pp2.append(368,267)
    elif WHAT == 'inward':
        # inward
        d._pp1.append(265,265)
        d._pp2.append(328,267)
        d._pp1.append(328,267)
        d._pp2.append(265,265)
    elif WHAT == 'enlarge':
        d._pp1.append(200,200)
        d._pp2.append(100,100)
        d._pp1.append(300,200)
        d._pp2.append(400,100)
        d._pp1.append(200,300)
        d._pp2.append(100,400)
        d._pp1.append(300,300)
        d._pp2.append(400,400)
    elif WHAT=='pinch':
        d._pp1.append(100,100)
        d._pp2.append(200,200)
        d._pp1.append(200,100)
        d._pp2.append(200,100)
        d._pp1.append(100,200)
        d._pp2.append(100,200)
        #
        d._pp1.append(300,100)
        d._pp2.append(300,100)
        d._pp1.append(400,100)
        d._pp2.append(300,200)
        d._pp1.append(400,200)
        d._pp2.append(400,200)
        #
        d._pp1.append(100,300)
        d._pp2.append(100,300)
        d._pp1.append(100,400)
        d._pp2.append(200,300)
        d._pp1.append(200,400)
        d._pp2.append(200,400)
        #
        d._pp1.append(300,400)
        d._pp2.append(300,400)
        d._pp1.append(400,400)
        d._pp2.append(300,300)
        d._pp1.append(400,300)
        d._pp2.append(400,300)
    else:
        # Identity transform
        d._pp1.append(100,100)
        d._pp2.append(100,100)
    
    if d:
        d.apply()


if 0:
##
    a = d._a3
    t = a.wobjects[1]
    im = t._texture1._dataRef
    vv.imwrite('c:/almar/projects/lena_eyes111.png', im[::2,::2]/255)
    