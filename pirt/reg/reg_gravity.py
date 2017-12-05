""" Registration using gravity registration.
"""

import numba
import numpy as np
import scipy.ndimage

from .. import Aarray, diffuse2
from .reg_base import GDGRegistration, create_grid_image


@numba.jit(nopython=True, nogil=True)
def near_root3(arr):
    """ near_root3(n)
    Calculates an approximation of the square root using
    (a few) Newton iterations.
    """
    for z in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            for x in range(arr.shape[2]):
                n = arr[z, y, x]
                v = 1.0    
                v = v - (v * v - n) / (2.0 * v)
                v = v - (v * v - n) / (2.0 * v)
                v = v - (v * v - n) / (2.0 * v)
                arr[z, y, x] = v


@numba.jit(nopython=True, nogil=True)
def near_exp3(arr):
    """ near_exp3(n)
    Calculates an approximation of the exp.
    """
    for z in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            for x in range(arr.shape[2]):
                v = arr[z, y, x]
                v = 1.0 + v / 256.0;
                v *= v; v *= v; v *= v; v *= v
                v *= v; v *= v; v *= v; v *= v
                arr[z, y, x] = v


class GravityRegistration(GDGRegistration):
    """ GravityRegistration(*images)
    
    Inherits from :class:`pirt.GDGRegistration`
    
    A registration algorithm based on attraction between masses in both 
    images, which is robust for large differences between the images.
    
    The most important parameters to tune the algorithm with are 
    scale_sampling, speed_factor and final_grid_sampling.
    
    The ``speed_factor`` and ``mass_transforms`` parameters are specific to
    this algorithm. Other important parameters are also listed below.
    
    Parameters
    ----------
    speed_factor : scalar
        The relative force of the transform. This one of the most important
        parameters to tune. Typical values are between 1 and 5. Default 1.
    mass_transforms : int or tuple of ints
        How the image is transformed to obtain the mass image. The number
        refers to the order of differentiation; 1 and 2 are gradient magnitude
        and Laplacian respectively. 0 only performs normalization to subtract
        the background. Can be specified for all images or for each image
        individually. Default 1.
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
        two in scale). For the coarse implementation, this is the amount of
        iterations performed before moving to the next scale (which is always
        a factor of two smaller). Values between 10 and 20 are reasonable in
        most situations. Default 15. Higher values yield better results in
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
        
        # The order of differentiation to calculate the mass images
        params.mass_transforms = 1
        
        # Iteration speed
        params.speed_factor = 1.0
        
        return params
    
    
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
    
    
    def _create_mass(self, image_id, im, scale):
        """ _create_mass(image_id, im, scale)
        
        Get the unnormalized mass image for the given image (which has
        the given image_id).
        
        This method can be overloaded to create the mass image in a
        custom way.
        
        """
        
        # Determine order of differentiation to create the mass image
        if isinstance(self.params.mass_transforms, (tuple,list)):
            order = self.params.mass_transforms[image_id]
        else:
            order = self.params.mass_transforms
        
        if order==0:
            # Plain image (but background is made "black")
            
            # Flip the intensities such that the part 
            # that is the background has the smallest values.
            mi, ma, me = im.min(), im.max(), im.mean()
            if (me-mi) > (ma-me):
                im = - im
            # And then, mass is the image
            mass = im
        
        elif order==1:
            # Gradient magnitude
            
            # Get squared derivative for each dimension
            massParts = []
            for d in range(im.ndim):
                tmp = self._get_derivative(im, d, order, 'nearest')
                massParts.append(tmp**2)
            
            # Sum and take square root
            mass = np.add(*massParts)**0.5
            # mass = np.add(*massParts)
            # near_root3(mass)  # mmm, does not seem to matter much
        
        elif order==2:
            # Laplacian
            
            # Get second order derivative for each dimension
            # edgemode nearest prevents artifacts at edges, and really
            # improves performance.
            massParts = []
            for d in range(im.ndim):
                tmp = self._get_derivative(im, d, 2, 'nearest')
                massParts.append(tmp)
            
            # Calculate Laplacian
            mass = np.add(*massParts)
            
            # Use only the positive part
            # Take abs, this results in kind of "sharp" mass data, but this
            # does not matter for the mass. The gravity field is smoothed
            # anyway.
            # Note that edges results in two masses that are very close, 
            # resulting in a smooth single "mass" for the gravity field.
            # Lines result in one strong mass with two side-lobes, which
            # becomes a single structure in the gravity field.
            mass = np.abs(mass)
        
        else:
            raise ValueError("This order is not implemented.")
        
        # Done
        return mass
    
    
    def _normalize_mass(self, mass):
        """ _normalize_mass(mass)
        
        Normalize the mass image. This method can be overloaded to implement
        custom normalization. This normalization should preferably be
        such that repeated calls won't change the result.
        
        """
        # The normalization is of crucial importance to this algorithm.
        # Sadly, it not trivial.
        mass *= (2/mass.std()) # i.e. make std of mass 2
        mass += (0.0-mass.mean()) # i.e. move mean to 0.0
        return mass
    
    
    def _get_mass_and_gradient(self, image_id, iterInfo):
        """ _get_mass_and_gradient(image_id, scale)
        
        Get the mass and the gradient for the given image id.
        Returns a tuple (mass, (gradz, grady, gradx))
        
        """
        scale = iterInfo[2]
        
        # Use buffered?
        buffered = self._get_buffered_data(image_id, iterInfo)
        if buffered is not None:
            mass = buffered
        
        else:
            # Get image, convert to mass, normalize
            im = self.get_deformed_image(image_id, scale)
            mass = self._create_mass(image_id, im, scale)
            mass = self._normalize_mass(mass)
            
            # Truncate the mass
            mass[mass<=0] = 0.0
            self._soft_limit1(mass, 1.0)
            
            # Buffer this mass image
            # We could also buffer the gradient, but that costs an
            # awefull lot of memory (for 3D images).
            self._set_buffered_data(image_id, iterInfo, mass)
        
        
        # Smooth to get the gravity field
        exta_smoothing = scale*1.0
        grav_field = diffuse2(mass, exta_smoothing)
        grav_field *= 1.0 / grav_field.mean()
        
        # Calculate gradient of mass fields.
        gradient = []
        for d in range(mass.ndim):
            tmp = self._get_derivative(grav_field, d, 1, 'nearest')
            gradient.append( tmp )
        
        # Done
        return mass, tuple(gradient)
    
    
    def _soft_limit1(self, data, limit):
        
        # Does not seem to be a bottleneck
        # if limit == 1:
        #     data = -data
        #     near_exp3(data)
        #     data[:] = 1.0 - data
        # else:
        #     data = -data/limit
        #     near_exp3(data)
        #     data[:] = -limit * (data-1)
        
        if limit == 1:
            data[:] = 1.0 - np.exp(-data)
        else:
            f = np.exp(-data/limit)
            data[:] = -limit * (f-1)
    
    def _soft_limit2(self, data, limit):
        f = np.exp(-np.abs(data)/limit)
        data[:] = -limit * (f-1) * np.sign(data)
    
    
#     def _grad_normalization(self, grad1, grad2, scale):
#         """ _grad_normalization(self, grad1, grad2, scale)
#         
#         Normalize the gradient vector field. This involves clipping
#         the values at the edges to prevent for bad values at the edges. 
#         This is a practical issue caused by the edge effects during 
#         convolution (convolution is ill-defined at the edges).
#         """
#         # todo: do I need this with frozen edges? NO
#         return
#         
#         for grad in [grad1, grad2]:
#             for tmp in grad:
#                 if tmp.ndim >= 1:
#                     m = int(np.ceil(scale/tmp.sampling[0]))+1 # Margin
#                     tmp[:m] = 0
#                     tmp[-m:] = 0
#                     tmp[m] *= 0.5 
#                     tmp[-(m+1)] *= 0.5 
#                 if tmp.ndim >= 2:
#                     m = int(np.ceil(scale/tmp.sampling[1]))+1 # Margin
#                     tmp[:,:m] = 0
#                     tmp[:,-m:] = 0
#                     tmp[:,m] *= 0.5 
#                     tmp[:,-(m+1)] *= 0.5 
#                 if tmp.ndim >= 3:
#                     m = int(np.ceil(scale/tmp.sampling[2]))+1 # Margin
#                     tmp[:,:,:m] = 0
#                     tmp[:,:,-m:] = 0
#                     tmp[:,:,m] *= 0.5 
#                     tmp[:,:,-(m+1)] *= 0.5 
    
    
    def _visualize(self, mass1=None, mass2=None, gridStep=10):
        """ _visualize(self,  mass1=None, mass2=None)
        
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
            ims = [ None, im1, grid1, mass1, 
                    None, im2, grid2, mass2]
            
            # Update textures
            for i in range(len(ims)):
                if ims[i] is not None:
                    t = self.visualizer.imshow((2,4,i+1), ims[i])
                    if i in [3, 7]:
                        t.SetClim(0,1)
            
        if firstpass:
            # Init
            
            # Show originals
            self.visualizer.imshow(241, self._ims[0])
            self.visualizer.imshow(245, self._ims[1])
            
            # Show titles
            #title_map = ['im 1', 'im 2', 'deformed 1', 'deformed 2', 'mass 1', 'mass 2']
            title_map = [   'im 1', 'deformed 1', 'grid 1', 'mass 1', 
                            'im 2', 'deformed 2', 'grid 2', 'mass 2']
            for i in range(len(title_map)):
                a = vv.subplot(2,4,i+1)
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
        
        # Get mass and its gradient
        self.timer.start('getting mass images and gradient')
        mass1, grad1 = self._get_mass_and_gradient(i, iterInfo)
        mass2, grad2 = self._get_mass_and_gradient(j, iterInfo)
        self.timer.stop('getting mass images and gradient')
        
#         # Prevent too high scales
#         if min(mass1.shape) < 16:
#             print 'skip because image too small'
#             return None
        
        # Calculate force vectors
        self.timer.start('calculating vectors')
        factor = float(self.params.speed_factor) * scale
        if not self.forward_mapping:
            factor *= -1
        dd_ = []
        for d in range(mass1.ndim):
            #dd_.append( grad1[d] * (-1 * factor) )
            #dd_.append( grad2[d] * (factor) )
            dd_.append( (grad2[d]-grad1[d]) * (factor) )
        self.timer.stop('calculating vectors')
        
        # Regularize using a B-spline grid
        deformForce = self.DeformationField(*dd_)
        deform = self._regularize_diffeomorphic(scale, deformForce, mass1*mass2)
        
        # Show
        if i==0 and j==1:
            self._visualize(mass1, mass2, self._get_grid_sampling(scale))
        elif self.params.deform_wise.lower() == 'pairwise2' and i==1:
            self._visualize(mass1, mass2, self._get_grid_sampling(scale))
        
        # Buffer B-spline grid and return
        self._set_buffered_data((i,j), iterInfo, deform)
        return deform
