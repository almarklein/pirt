import time

import visvis as vv

from pirt import FieldDescription
from pirt import (DeformationGridForward, DeformationFieldForward,
                  DeformationGridBackward, DeformationFieldBackward)
from visvis import Point, Pointset


class DeformByHand:
    """ DeformByHand(im, grid_sampling=40)
    
    Demo application to deform a 2D image by hand using a spline grid.
    
    Use the grid property to obtain the deformation grid.
    Use the run() method to wait for the user to close the figure.
    
    """
    
    def __init__(self, im, grid_sampling=40):
        
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
                        self._pp1.data, self._pp2.data,
                        injective=self._injective, frozenedge=self._frozenedge)
        else:
            DeformationGrid = DeformationGridForward
            if not self._forward:
                DeformationGrid = DeformationGridBackward
            grid = DeformationGrid.from_points(self._im, self._sampling, 
                        self._pp1.data, self._pp2.data, 
                        injective=self._injective, frozenedge=self._frozenedge)
            deform = grid.as_deformation_field()
        
        # Store grid
        field0 = self.DeformationField(FieldDescription(self._im))
        self._field2 = deform
        field3 = self._field1.compose(self._field2)
        
        # Deform
        im2 = self._field1.apply_deformation(self._im)
        im3 = field3.apply_deformation(self._im)
        
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
        text1 = 'B-spline (S) with sampling %i (U/D) and %i levels (L/R).' % (
            self._sampling, self._levels)
        
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
        
        self.apply_deform()


if __name__ == '__main__':
    
    d = DeformByHand(vv.imread('astronaut.png')[:,:,2].astype('float32'))
    
    WHAT = 'twist'
    
    if WHAT == 'twist':
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


    if False:
        ## 
        a = d._a3
        t = a.wobjects[1]
        im = t._texture1._dataRef
        vv.imwrite('~/warped.jpg', im[::1,::1]/255)
