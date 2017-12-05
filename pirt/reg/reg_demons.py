""" Demons registration algorithm
"""

import numpy as np
import scipy.ndimage

from .. import Aarray, diffuse2
from .reg_base import GDGRegistration, BaseRegistration, create_grid_image


class BaseDemonsRegistration(object):
    """ BaseDemonsRegistration
    
    Abstract class that implements the base functionality of the
    Demons algorithm.
    
    """
    
    def _get_derivative(self, im, d, o=1, edgeMode='constant'):
        """ _get_derivative(im, d, o=1)
        
        Calculate the derivative (of order o) of the given image
        in the given dimension.
        
        """
        
        # Set edgeMode to constant, because when the image is deformed, 
        # its (deformed) edge will give rise to high filter response any way.
        
        # We can apply differentiation using compact support kernels
        # because we use a scale space pyramid based on discrete 
        # diffusion kernels as proposed by Tony Lindeberg. The
        # resulting differentiation is theoretically valid.
        # (Of course, for low scales, the results suffer from discretisation
        # errors, which is why the scale should be at least 0.5/1.0.)
        
        if o == 0:
            return im # No differentiation
        elif o == 1:
            k = np.array([0.5, 0, -0.5], dtype='float64')
        elif o == 2:
            k = np.array([1, -2, 1], dtype='float64')
        else:
            raise ValueError('Order of differentiation must be {0,1,2}.')
            
        # For o in [1,2]
        tmp = scipy.ndimage.convolve1d(im, k, d, mode=edgeMode)
        return Aarray(tmp, im.sampling)
    
    
    def _get_image_and_gradient(self, image_id, iterInfo):
        """ _get_image_and_gradient(image_id, iterInfo)
        
        Get the image and the gradient for the given image id.
        Returns a tuple (mass, (gradz, grady, gradx))
        
        """
        scale = iterInfo[2]
        
        # Use buffered?
        buffered = self._get_buffered_data(image_id, iterInfo)
        if buffered is not None:
            im = buffered
        
        else:
            # Get deformed image
            im = self.get_deformed_image(image_id, scale)
            
            # Buffer
            self._set_buffered_data(image_id, iterInfo, im)
        
        
        # Calculate gradient of image.
        gradient = []
        for d in range(im.ndim):
            tmp = self._get_derivative(im, d, 1, 'nearest')
            gradient.append( tmp )
        
        # Done
        return im, tuple(gradient)
    
    
    def _visualize(self, gridStep=10):
        """ _visualize(self)
        
        Visualize the registration process.
        
        """
        
        if self.visualizer.fig is None:
            return
        
        import visvis as vv
        
        firstpass = not self.visualizer.fig.children
        
        if True:
            # Show
            
            # Apply deformation to image
            im1 = self.get_deformed_image(0, 0)
            im2 = self.get_deformed_image(1, 0)
            
            # Get grid images
            grid1 = create_grid_image(im1.shape, im1.sampling, gridStep)
            grid2 = grid1.copy()
            deform1, deform2 = self._deforms.get(0, None), self._deforms.get(1, None)
            if deform1 and deform2:
                grid1 = deform1.apply_deformation(grid1)# + 0.1*self._deforms[0][0])
                grid2 = deform2.apply_deformation(grid2)# + 0.1*self._deforms[1][0])
            
            # Get images
            ims = [ None, im1, grid1, 
                    None, im2, grid2]
            
            # Update textures
            for i in range(len(ims)):
                if ims[i] is not None:
                    self.visualizer.imshow((2,3,i+1), ims[i])
        
        if firstpass:
            # Init
            
            # Show originals
            self.visualizer.imshow(231, self._ims[0])
            self.visualizer.imshow(234, self._ims[1])
            
            # Show titles
            title_map = [   'im 1', 'deformed 1', 'grid 1', 
                            'im 2', 'deformed 2', 'grid 2']
            for i in range(len(title_map)):
                a = vv.subplot(2,3,i+1)
                vv.title(title_map[i])
                a.axis.visible = False
        
        # Draw now
        self.visualizer.fig.DrawNow()
    
    
    def _deform_from_image_pair(self, i, j, iterInfo):
        """ _deform_from_image_pair(i, j, iterInfo)
        
        Calculate the deform for image i to image j.
        
        """
        
        # Extract iter info
        level, iter, scale = iterInfo
        
        # Try using buffered data
        # we can make good use of the fact that our delta deforms are symetric
        buffered = self._get_buffered_data((i,j), iterInfo)
        if buffered is not None:
            return buffered
        buffered = self._get_buffered_data((j,i), iterInfo)
        if buffered is not None:
            for grid in buffered:
                grid._knots = - grid._knots
            return buffered
        
        # Get images and their gradients
        self.timer.start('getting images')
        im1, grad1 = self._get_image_and_gradient(i, iterInfo)
        im2, grad2 = self._get_image_and_gradient(j, iterInfo)
        self.timer.stop('getting images')
        
#         # Prevent too high scales
#         if min(im1.shape) < 16:
#             return None
        
        
        self.timer.start('calculating vectors')
        
        # Calculate norms
        norm1 = im1 * 0.0 # copy
        norm2 = im2 * 0.0
        for d in range(im1.ndim):
            norm1 += grad1[d]**2
            norm2 += grad2[d]**2
        
        # Calculate denumerators
        imd = im1-im2
        imd2 = imd**2
        alpha = float(self.params.noise_factor)
        denum1 = norm1 + alpha**2 * imd2
        denum2 = norm2 + alpha**2 * imd2
        del norm1, norm2, imd2
        
        # Make sure the division doesnt cause us problems
        denum1[denum1==0] = np.inf
        denum2[denum2==0] = np.inf
        
        # Find deformation; use the gradient of both images. 
        speed_factor = float(self.params.speed_factor)
        if not self.forward_mapping:
            speed_factor *= -1
        dd_ = []
        for d in range(imd.ndim):
            tmp = imd * ( grad1[d]/denum1 + grad2[d]/denum2 ) * speed_factor
            dd_.append(tmp)
        
        self.timer.stop('calculating vectors')
        
        if isinstance(self, GDGRegistration):
            
            # Regularize using a B-spline grid
            deformForce = self.DeformationField(*dd_)
            deform = self._regularize_diffeomorphic(scale, deformForce)
        
        else:
            
            # Regularize deformation field using diffusion
            self.timer.start('regularizing')
            for d in range(imd.ndim):
                # Get sigma
                final_scale = float(self.params.final_scale)
                final_smoothing = float(self.params.final_smoothing)
                sigma_reg = scale * final_smoothing / final_scale
                # Diffuse
                dd_[d] = diffuse2(dd_[d], sigma_reg)
                dd_[d] = dd_[d].astype('float32')
            self.timer.stop('regularizing')
            
            # Make deform
            deform = self.DeformationField(*dd_)
        
        # Show
        gridSize = 10
        if hasattr(self, '_get_grid_sampling'):
            gridSize = self._get_grid_sampling(scale)
        if i==0 and j==1:
            self._visualize(gridSize)
        elif self.params.deform_wise.lower() == 'pairwise2' and i==1:
            self._visualize(gridSize)
        
        # Buffer B-spline grid and return
        self._set_buffered_data((i,j), iterInfo, deform)
        return deform



class OriginalDemonsRegistration(BaseRegistration, BaseDemonsRegistration):
    """ OriginalDemonsRegistration(*images)
    
    Inherits from :class:`pirt.BaseRegistration`.
    
    The original version of the Demons algorithm. Uses Gaussian diffusion
    to regularize the deformation field. This is the implementation as 
    proposed by He Wang et al. in 2005 "Validation of an accelerated 'demons' 
    algorithm for deformable image registration in radiation therapy"
    
    See also :class:`pirt.DiffeomorphicDemonsRegistration`.
    
    The ``speed_factor`` and ``noise_factor`` parameters are specific to this
    algorithm. Other important parameters are also listed below.
    
    Parameters
    ----------
    speed_factor : scalar
        The relative force of the transform. This one of the most important
        parameters to tune. Default 3.0.
    noise_factor : scalar
        The noise factor. Default 2.5.
    final_scale : scalar
        The minimum scale used during the registration process. This is the
        scale at which the registration ends. Default 1.0. Because calculating
        differentials suffer from more errors as the scale decreases, the
        minimum value is limited at 0.5.
    scale_levels : integer
        The amount of scale levels to use during the registration. Each level
        represents a factor of two in scale. The default (4) works for
        most images, but for large images or large deformations a larger
        value can be used.
    scale_sampling : scalar
        The amount of iterations for each level (i.e. between each factor 
        two in scale). Values between 20 and 30 are reasonable in
        most situations. Default 25. Higher values yield better results in
        general. The speed of the algorithm scales linearly with this value.
    
    """
    
    def _defaultParams(self):
        """ Overload to create all default params.
        """
        params = BaseRegistration._defaultParams(self)
        
        # Iteration speed and noise
        params.speed_factor = 3.0
        params.noise_factor = 2.5
        
        # Change default values
        params.scale_sampling = 25
        
        # Set some things to make stuff original
        params.smooth_scale = False
        params.combine_deforms = 'add'
        params.mapping = 'backward'
        #params.deform_wise = 'pairwise' is of GDG
        
        # Smoothing factor
        params.final_smoothing = 8
        
        return params
    
    def _deform_for_image(self, i, iterInfo):
        # implement pairwise registration
        
        # Get wise
        deform_wise = 'pairwise'
        if 'deform_wise' in self.params:
            deform_wise = self.params.deform_wise.lower()
        
        # Decide which routine to use
        if deform_wise in ['pairwise', 'pairwise1']:
            if i==0:
                return self._deform_from_image_pair(0, 1, iterInfo)
        elif deform_wise == 'pairwise2':
            if i==1:
                return self._deform_from_image_pair(1, 0, iterInfo)
        elif deform_wise == 'groupwise':
            self._set_buffered_data((0,1), iterInfo, None)
            self._set_buffered_data((1,0), iterInfo, None)
            if i==0: 
                return self._deform_from_image_pair(0, 1, iterInfo)
            elif i==1:
                return self._deform_from_image_pair(1, 0, iterInfo)
            else:
                raise ValueError('Classic Demons only supports 2 images.')
        else:
            raise ValueError('Invalid value for param deform_wise: %s' % repr(deform_wise))


class DiffeomorphicDemonsRegistration(GDGRegistration, BaseDemonsRegistration):
    """ DiffeomorphicDemonsRegistration(*images)
    
    Inherits from :class:`pirt.GDGRegistration`.
    
    A variant of the Demons algorithm that is diffeomorphic. Based on the
    generice diffeomorphic groupwise registration (GDGRegistration) method .
    
    See also :class:`pirt.OriginalDemonsRegistration`.
    
    The ``speed_factor`` parameter is specific to this algorithm. The
    ``noise_factor`` works best set at 1.0, effectively disabling
    its use; it is made redundant by the B-spline based regularization.
    Other important parameters are also listed below.
    
    Parameters
    ----------
    speed_factor : scalar
        The relative force of the transform. This one of the most important
        parameters to tune. Default 3.0.
    mapping : {'forward', 'backward'}
        Whether forward or backward mapping is used. Default forward.
    final_scale : scalar
        The minimum scale used during the registration process. This is the
        scale at which the registration ends. Default 1.0. Because calculating
        differentials suffer from more errors as the scale decreases, the
        minimum value is limited at 0.5.
    scale_levels : integer
        The amount of scale levels to use during the registration. Each level
        represents a factor of two in scale. The default (4) works for
        most images, but for large images or large deformations a larger
        value can be used.
    scale_sampling : scalar
        The amount of iterations for each level (i.e. between each factor 
        two in scale). Values between 20 and 30 are reasonable in
        most situations. Default 25. Higher values yield better results in
        general. The speed of the algorithm scales linearly with this value.
    final_grid_sampling : scalar
        The grid sampling of the grid at the final level. During the 
        registration process, the B-spine grid sampling scales along 
        with the scale. This parameter is usually best coupled to final_scale.
        (When increasing final scale, this value should often be increased
        accordingly.)
    grid_sampling_factor : scalar between 0 and 1
        To what extent the grid sampling scales with the scale. By making
        this value lower than 1, the grid is relatively fine at the the
        higher scales, allowing for more deformations. The default is 0.5.
        Note that setting this value to 1 when using 'frozenedge' can cause
        the image to be 'stuck' at higher scales.
    
    """
    
    def _defaultParams(self):
        """ Overload to create all default params.
        """
        params = GDGRegistration._defaultParams(self)
        
        # Change default values
        params.scale_sampling = 25
        
        # Iteration speed and noise
        params.speed_factor = 3.0
        params.noise_factor = 1.0
        
        return params

