""" Demonstrates the scale space pyramid with Gaussian diffusion.
"""

import numpy as np

from pirt import ScaleSpacePyramid
from pirt import Aarray


class Demo2D:
    def __init__(self, im):
        
        # Make Aarray
        im = Aarray(im)
        
        # Create pyramid
        self._p = ScaleSpacePyramid(im)
        maxLevel, maxSigma = self._p.calculate()
        
        # Init visualization
        vv.figure(1); vv.clf()
        self._axes = axes = vv.gca()
        axes.position.Correct(dy=40, dh=-40)
        
        # Make slider
        self._slider = vv.Slider(axes)
        self._slider.position = 0,-40, 1, 20
        self._slider.fullRange = 0, maxLevel
        self._slider.value = 0
        
        # Show image
        self._t = vv.imshow(self._p.get_level(0))
        
        # Bind to handler
        self._slider.eventSliding.Bind(self.on_sliding)
        self._slider.eventSliderChanged.Bind(self.on_sliding)
    
    
    def on_sliding(self, event):
        
        # Get level
        level = self._slider.value
        
        # Get image
        im = self._p.get_level(level)
        
        # Replace
        self._t.SetData(im)
    
    
class Demo2D3:
    def __init__(self, im, min_scale=None, scale_offset=0):
        
        # Make Aarray
        if True:# not isinstance(im, Aarray):
            im = Aarray(im)
        
        # Create pyramids
        self._p1 = ScaleSpacePyramid(im, min_scale, scale_offset, level_factor=1.5)
        self._p2 = ScaleSpacePyramid(im, min_scale, scale_offset, level_factor=2)
        self._p3 = ScaleSpacePyramid(im, min_scale, scale_offset, level_factor=3)
        #maxLevel, maxSigma = self._p1.calculate()
        #self._p2.calculate()
        #self._p3.calculate()
        
        # Init visualization
        fig = vv.figure(1); vv.clf()
        self._axes1 = axes1 = vv.subplot(131); vv.title('level factor 1.5')
        self._axes2 = axes2 = vv.subplot(132); vv.title('level factor 2.0')
        self._axes3 = axes3 = vv.subplot(133); vv.title('level factor 3.0')
        axes1.position.Correct(dy=40, dh=-40)
        axes2.position.Correct(dy=40, dh=-40)
        axes3.position.Correct(dy=40, dh=-40)
        
        # Share camera
        cam = vv.cameras.TwoDCamera()
        for ax in [axes1, axes2, axes3]:
            ax.camera = cam
        
        # Make slider
        self._slider = vv.Slider(fig)
        self._slider.position = 0.1, 5, 0.8, 20
        self._slider.fullRange = 0, 25
        self._slider.value = 1
        
        # Show image
        self._t1 = vv.imshow(self._p1.get_level(0), axes=axes1)
        self._t2 = vv.imshow(self._p2.get_level(0), axes=axes2)
        self._t3 = vv.imshow(self._p3.get_level(0), axes=axes3)
        
        # Bind to handler
        self._slider.eventSliding.Bind(self.on_sliding)
        self._slider.eventSliderChanged.Bind(self.on_sliding)
    
    
    def on_sliding(self, event):
        
        # Get level
        sigma = self._slider.value
        
        # Get images
        im1 = self._p1.get_scale(sigma)
        im2 = self._p2.get_scale(sigma)
        im3 = self._p3.get_scale(sigma)
        
        # Replace textures
        self._t1.SetData(im1)
        self._t2.SetData(im2)
        self._t3.SetData(im3)


if __name__ == '__main__':
    import visvis as vv
    
    # Read image
    im = vv.imread('astronaut.png')[:,:,1].astype(np.float32)
    im = Aarray(im)[::2,:]
    
    d = Demo2D3(im, 1.5)
    vv.use().Run()

    if 0:

        ## Diffusionkernel vs Gaussiankernel
        import pirt
        import visvis as vv
        
        sigma = 300
        k1, t1 = pirt.diffusionkernel(sigma, returnt=True)
        k2, t2 = pirt.gaussiankernel(sigma, returnt=True)
        
        vv.figure()
        vv.plot(t1, k1, lc='r', ls=':')
        vv.plot(t2, k2, lc='b')
