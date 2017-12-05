""" 
Defines the base registration object.
"""

import time

import numpy as np

from .. import Aarray, Parameters, ScaleSpacePyramid, SplineGrid
from .. import (Deformation, DeformationField, DeformationIdentity,
                DeformationGridForward, DeformationFieldForward,
                DeformationGridBackward, DeformationFieldBackward)


class classproperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()


def create_grid_image(shape, sampling=None, step=10, bigstep=None):
    """ create_grid_image(shape, sampling=None, step=10, bigstep=5*step)
    
    Create an image depicting a grid. The first argument can also be an array.
    
    """ 
    step = int(step + 0.4999999)
    
    # Accept arrays
    if isinstance(shape, np.ndarray):
        im = shape
        shape = im.shape
        if hasattr(im, 'sampling'):
            sampling = im.sampling
    
    # Default sampling
    if sampling is None:
        sampling = [1 for s in shape]
    
    # Calculate bigstep
    if bigstep is None:
        bigstep = step * 5
    
    # Create image
    im = Aarray(shape, sampling, fill=0.0, dtype=np.float32)
    #
    im[1::step,:] = 1
    im[:,1::step] = 1
    im[::bigstep,:] = 1.2
    im[1::bigstep,:] = 1.2
    im[2::bigstep,:] = 1.2
    im[:,::bigstep] = 1.2
    im[:,1::bigstep] = 1.2
    im[:,2::bigstep] = 1.2
    # 
    return im


class Progress(object):
    """ Progress()
    
    Allows an algorithm to display the progress to the user.
    
    """
    
    def __init__(self):
        
        # Init variables to show progress
        self._progress_last_message = ''
        self._progress_iter = 0
        self._progress_max_iters = 0
    
    def start(self, message, max_iters=0):
        """ start(message, max_iters=0)
        
        Start a progress. The message should indicate what is being done.
        
        """
        
        # Init progress variables
        self._progress_last_message = ''
        self._progress_iter = 0
        self._progress_max_iters = max_iters
        
        print(message)
    
    
    def next(self, extra_info=''):
        """ next(extra_info='')
        
        Update progress to next iteration and show the new progress. 
        Optionally a message with extra information can be displayed.
        
        """
        self._progress_iter += 1
        self.show(extra_info)
    
    
    def show(self, extra_info=''):
        """ show(extra_info='')
        
        Show current progress, and optional extra info.
        
        """
        
        # Get series of backspaces to remove previous message
        rem = '\b' * (len(self._progress_last_message)+1)
        
        # Create message
        mes = ''
        if self._progress_max_iters:
            i1, i2 = self._progress_iter, self._progress_max_iters
            mes += 'iter %i/%i (%1.0f%%)' % (i1, i2, 100*(float(i1)/i2))
        else:
            mes += 'iter %i' % self._progress_iter
        if extra_info:
            mes += ', ' + str(extra_info)
        
        # Store and print message
        self._progress_last_message = mes
        print(rem + mes)


class Timer(object):
    """ Timer()
    
    Enables registration objects to time the different components.
    Can be used to optimize the speed of the registration algorithms,
    or to study the effect of a parameter on the speed.
    
    Multiple things can be timed simultaneously. Timers can also be
    started and stopped multiple times; the total time is measured.
    
    """
    
    def __init__(self):
        # Init variables to perform timing
        self._timers = {}
    
    def start(self, id):
        """ start(id)
        
        Start a timer for the given id, which should best be a string.
        
        The timer can be started and stopped multiple times. 
        In the end the total time spend on 'id' can be displayed 
        using show().
        
        """
        
        # Get timers for this type
        if id in self._timers:
            timers = self._timers[id]
        else:
            timers = []
            self._timers[id] = timers
        
        # Add start time
        timers.append( time.time() )
    
    
    def stop(self, id):
        """ stop(id)
        
        Stop the timer for 'id'.
        
        """
        
        # Get timers for this type
        if id in self._timers:
            timers = self._timers[id]
        else:
            return # No timer to stop
        
        # Stop the last timer
        t = timers[-1]
        if t > 946080000: # 30*365*24*60*60: a time after the year 2000
            timers[-1] = time.time() - t
        else:
            pass # An already stopped timer
    
    
    def get(self, id):
        """ get(id)
        
        Get the total time spend in seconds on 'id'.
        Returns -1 if the given id is not valid.
        
        """
        
        # Get timers for this type
        if id in self._timers:
            timers = self._timers[id]
        else:
            return -1# No timer to get
        
        # Sum all stopped timers (do not added running timers)
        t_sum = 0
        for t in timers:
            if t < 946080000: # 30*365*24*60*60: less than 30 years
                t_sum += t
        
        # Done
        return t_sum
    
    
    def show(self, id=None):
        """  show(id=None)
        
        Show (print) the results for the total timings of 'id'.
        If 'id' is not given, will print the result of all timers.
        
        """
        
        # Get what timers to display
        if id is None:
            ids = self._timers.keys()
        elif isinstance(id, (tuple,list)):
            ids = id
        else:
            ids = [id]
        
        # Display timers
        for id in ids:
            t = self.get(id)
            print('Total time spend on %s: %1.2f seconds' % (str(id), t))


class Visualizer(object):
    """ Visualize
    
    Tool to visualize the images during registration.
    
    """
    
    def __init__(self):
        self._f = None
    
    def init(self, fig):
        """ init(fig)
        
        Initialize by giving a figure.
        
        """
        self._f = fig
        if fig is not None:
            import visvis as vv  # noqa - so importerror is raised if visvis not available
    
    @property
    def fig(self):
        """ Get the figure instance (or None) if init() was not called.
        """
        return self._f
    
    def imshow(self, subplot, im, *args, **kwargs):
        """ imshow(subplot, im, *args, **kwargs)
        
        Show the given image in specified subplot. If possible, 
        updates the previous texture object.
        
        """
        
        # Check if we can show
        if not self.fig:
            return
        else:
            self._f.MakeCurrent()
        
        # Import visvis
        import visvis as vv
        
        # Get axes
        if isinstance(subplot, tuple):
            a = vv.subplot(*subplot)
        else:
            a = vv.subplot(subplot)
        
        # Get what's in there
        t = None
        if len(a.wobjects) == 2 and isinstance(a.wobjects[1], vv.BaseTexture):
            t = a.wobjects[1]
        
        # Reuse, or clear and replace
        if t is not None:
            t.SetData(im)
        else:
            a.Clear()
            t = vv.imshow(im, *args, **kwargs)
        
        # Done
        return t


class AbstractRegistration(object):
    """ AbstractRegistration(*images, makeFloat=True)
    
    Base class for registration of 2 or more images. This class only provides
    a common interface for the user.
    
    This base class can for example be inherited by registration classes 
    that wrap an external registration algorithm, such as Elastix.
    
    Also see :class:`pirt.BaseRegistration`.
    
    Implements:
    
      * progress, timer, and visualizer objects
      * properties to handle the mapping (forward or backward)
      * functionality to get and set parameters
      * basic functionality to get resulting deformations
      * functionality to show the result (2D only)
    
    Parameters
    ----------
    None
    """
    
    # Inherting methods should implement register()
    
    def __init__(self, *ims, makeFloat=True):
        
        # Init images
        self._ims = []
        
        # Check number of images
        if len(ims) < 2: 
            raise ValueError('Need at least two images at initialisation.')
        
        # Check all images
        for im in ims:
            # Check if numpy array
            if not isinstance(im, np.ndarray):
                raise ValueError('Images to register should be numpy arrays.')
            # Make float (if necessary)
            if makeFloat and im.dtype not in [np.float32, np.float64]:
                im = im.astype(np.float32)
            # Make anisotropic arrays (is view on same data, no data copying)
            if not isinstance(im, Aarray):
                im = Aarray(im)
            # Store
            self._ims.append(im)
        
        # Init deformations
        self._deforms = {}
        
        # Init params
        self._params = self._defaultParams()
        
        # Instantiate progress and timer
        self._progress = Progress()
        self._timer = Timer()
        self._visualizer = Visualizer()
    
    
    @classmethod
    def register_and_get_object(cls, *ims, **params):    
        """ register_get_object(*ims, **params)
        
        Classmethod to register the given images with the given
        parameters, and return the resulting registration object
        (after the registration has been performed).
        
        """
        # Create registration object
        ro = cls(*ims)
        
        # Set params
        for param in params.keys():
            ro.params[param] = params[param]
        
        # Register
        ro.register()
        
        # Done
        return ro
    
    
    ## Methods and properties for the registration and results
    
    
    @property
    def progress(self):
        """ The progress object, can be used by the algorithm to indicate
        its progress.
        """
        return self._progress
    
    @property
    def timer(self):
        """ The timer object, can be used by the algorithm to measure the
        processing time of the different steps in the registration algorithm.
        """
        return self._timer
    
    @property
    def visualizer(self):
        """ The visualizer object, can be used by the algorithm to display
        the images as they are deformed during registration.
        """
        return self._visualizer
    
    
    ## Properties to handle forward/backward mapping
    
    @property
    def forward_mapping(self):
        """ Whether forward (True) or backward (False) 
        mapping is to be used internally.
        """
        if self.params.mapping.lower() == 'forward':
            return True
        elif self.params.mapping.lower() == 'backward':
            return False
        else:
            raise ValueError('Invalid mapping.')
    
    
    @property
    def DeformationField(self):
        """ Depending on whether forward or backward mapping is used,
        returns the DeformationFieldForward or DeformationFieldBackward
        class.
        """
        if self.forward_mapping:
            return DeformationFieldForward
        else:
            return DeformationFieldBackward
    
    
    @property
    def DeformationGrid(self):
        """ Depending on whether forward or backward mapping is used,
        returns the DeformationGridForward or DeformationGridBackward
        class.
        """
        if self.forward_mapping:
            return DeformationGridForward
        else:
            return DeformationGridBackward
    
    
    ## Methods and props for params
    
    def _defaultParams(self):
        """ Overload to create all default params.
        """
        params = Parameters()
        params._class_name = self.__class__.__name__
        params.mapping = 'undefined'
        return params
    
    @classproperty
    @classmethod
    def defaultParams(cls):
        """ Class property to get the default params for this registration
        class.
        """
        # Instantiate a dummy class to obtain the default params
        im = np.zeros((10,10), 'float32')
        reg = cls(im,im)
        return reg._defaultParams()
    
    @property
    def params(self):
        """ Get params structure (as a Parameters object). 
        """
        return self._params
    
    
    def set_params(self, params=None, **kwargs):
        """ set_params(params=None, **kwargs)
        
        Set any parameters. The parameters are updated with the given 
        dict, Parameters object, and then with the parameters given
        via the keyword arguments.
        
        Note that the parameter structure can also be accessed directly via
        the 'params' propery.
        
        """
        # Combine user input
        D = {}
        if params:
            for key in params:
                D[key] = params[key]
        if True:
            for key in kwargs:
                D[key] = kwargs[key]
        
        # Set style elements
        invalidKeys = []
        for key in D:
            self._params[key] = D[key]
            if key not in self._params:
                invalidKeys.append(key)
        
        # Give warning for invalid keys
        if invalidKeys:
            print("Warning, invalid param given: " + ','.join(invalidKeys))
    
    
    ## Methods to get result and show result
    
    def get_deform(self, i=0):
        """ get_deform(i=0)
        
        Get the DeformationField instance for image with index i. If groupwise
        registration was used, this deformation field maps image i to the mean
        shape.
        
        """
        
        # Check
        if not isinstance(i, int):
            raise ValueError('The argument of get_deform must be an int.')
        
        try:
            return self._deforms[i]
        except KeyError:
            raise KeyError('The deformation for index %i' % i + 'is not available.')
    
    def get_final_deform(self, i=0, j=1, mapping=None):
        """ get_final_deform(i=0, j=1, mapping=None)
        
        Get the DeformationField instance that maps image with index i 
        to the image with index j. If groupwise registration was used,
        the deform is a composition of deform 'i' with the inverse of
        deform 'j'.
        
        Parameters
        ----------
        i : int
            The source image
        j : int
            The target image
        mapping : {'forward', 'backward', Deformation instance}
            Whether the result should be a forward or backward deform.
            When specified here, the result can be calculated with less
            errors than for example using result.as_forward(). If a 
            Deformation object is given, the mapping of that deform is used.
        
        """
        
        # Check
        for ij in [i,j]:
            if not isinstance(ij, int):
                raise ValueError('Both arguments must be indices to the images.')
        
        # Get individual deforms
        deform1 = self._deforms.get(i, None)
        deform2 = self._deforms.get(j, None)
        
        # Handle mapping
        if mapping is None:
            mapForward = self.forward_mapping
        elif isinstance(mapping, str):
            if mapping.lower() == 'forward':
                mapForward = True
            elif mapping.lower() == 'backward':
                mapForward = False
            else:
                raise ValueError('Invalid mapping specified.')
        elif isinstance(mapping, Deformation):
            mapForward = mapping.forward_mapping
        else: 
            raise ValueError('Invalid mapping specified.')
        
        # Compose
        if deform1 and deform2:
            if mapForward:
                deform1 = deform1.as_forward()
                deform2i = deform2.as_forward_inverse()
                deform = deform1.compose(deform2i)
            else:
                deform1 = deform1.as_backward()
                deform2i = deform2.as_backward_inverse()
                deform = deform1.compose(deform2i)
        elif deform1:
            if mapForward:
                deform = deform1.as_forward()
            else:
                deform = deform1.as_backward()
        elif deform2:
            if mapForward:
                deform = deform2.as_forward_inverse()
            else:
                deform = deform2.as_backward_inverse()
        else:
            raise ValueError('No deform available. Run register().')
        
        # Done
        return deform
    
    
    def show_result(self, how=None, fig=None):
        """ show_result(self, how=None, fig=None)
        
        Convenience method to show the registration result. Only 
        works for two dimensional data.
        Requires visvis.
        
        """
        
        # Check if dimension ok
        if self._ims[0].ndim != 2:
            raise RuntimeError('show_result only works for 2D data.')
        
        # Check if result is set
        if not self._deforms:
            raise RuntimeError('The result is not available; run register().')
        
        # Make how lower if string
        if isinstance(how, str):
            how = how.lower()
        
        # Import visvis
        import visvis as vv
        
        # Create figure
        if fig is None:
            fig = vv.figure()
        else:
            fig = vv.figure(fig.nr)
            fig.Clear()
        
        
        if how in [None, 'grid', 'diff', 'dx', 'dy']:
            
            # Title map
            title_map = ['Moving', 'Static', 'Deformed', 'Grid']
            
            # Get deform
            deform = self.get_final_deform(0, 1)
            
            # Create third image
            im1_ = deform.apply_deformation(self._ims[0])
            
            # Create fourth image
            if how in [None, 'grid']:
                im2_ = create_grid_image(im1_.shape, im1_.sampling)
                im2_ = deform.apply_deformation(im2_)
            elif how == 'diff':
                im2_ = np.abs(im1_-self._ims[1])
                title_map[3] = 'Diff'
            elif how == 'dx':
                im2_ = deform[1]
                title_map[3] = 'dx'
            elif how == 'dy':
                im2_ = deform[0]
                title_map[3] = 'dy'
            
            # Set images
            ims = [self._ims[0], self._ims[1], im1_, im2_]
            
            # Imshow all figures
            aa, tt = [],[]
            for i in range(len(ims)):
                a = vv.subplot(2,2,i+1)
                t = vv.imshow(ims[i])
                vv.title(title_map[i])
                a.axis.visible = False
                aa.append(a); tt.append(t)
            
            # Done
            return tuple(tt)
    
    
    ## Some basic registration methods (and its tools)
    
    
    def register(self, verbose=1, fig=None):
        """ register(verbose=1, fig=None)
        
        Perform the registration process. 
        
        Parameters
        ----------
        verbose : int
            Verbosity level. 0 means silent, 1 means print some, 2 means
            print a lot.
        fig : visvis figure or None
            If given, will display the registration progress in the given
            figure.
        """
        self._register(verbose, fig)
    
    def _register(self, verbose, fig):
        """ Inheriting classes should overload this method.
        """
        raise NotImplemented()



class NullRegistration(AbstractRegistration):
    """ NullRegistration(*images)
    
    Inherits from :class:`pirt.AbstractRegistration`.
    
    A registration algorithm that does nothing. This can be usefull to test
    the result if no registration would be applied.
    
    Parameters
    ----------
    None
    
    """
    def _defaultParams(self):
        """ Overload to create all default params.
        """
        params = AbstractRegistration._defaultParams(self)
        params.mapping = 'backward'
        return params
    
    def _register(self, *args, **kwargs):
        for i in range(len(self._ims)):
            shape = self._ims[i].shape
            fields = [np.zeros(shape, 'float32') for s in shape]
            self._deforms[i] = DeformationFieldForward(*fields)



class BaseRegistration(AbstractRegistration):
    """ BaseRegistration(*images)
    
    Inherits from :class:`pirt.AbstractRegistration`.
    
    An abstract registration class that provides common functionality
    shared by almost all registration algorithms.
    
    This class maintains an image pyramid for each image, implements methods
    to set the delta deform, and get the deformed image at a specified scale.
    Further, this class implements the high level aspect of the registration
    algorithm that iterates through scale space.
    
    Parameters
    ----------
    mapping : {'forward', 'backward'}
        Whether forward or backward mapping is used. Default forward.
    combine_deforms : {'compose', 'add'}
        How deformations are combined. Default compose. While add is used
        in some (older) registration algorithms, it is a coarse approximation.
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
        two in scale). What values are reasonable depends on the specific
        algorithm.
    smooth_scale : bool
        Whether a smooth scale space should be used (default) or the scale
        is reduced with a factor of two each scale_sampling iterations.
    
    """
    
    # Implements:
    #   * _register()
    #   * _set_delta_deform(i, deform)
    #   * get_deformed_image(i)
    #   * _set_buffered_data(key1, key2, data)
    #   * _get_buffered_data(key1, key2)
    #   
    # Inherting methods should implement:
    #   * _deform_for_image(i, iterInfo)
    
    def __init__(self, *ims):
        AbstractRegistration.__init__(self, *ims)
        
        # Init container for scale space for the images 
        self._pyramids = None
        
        # Buffer to store reusable data
        self._buffer = {}
        
        # Attributes to steer the registration process
        self._current_interp_order = 1
    
    
    def _defaultParams(self):
        """ Overload to create all default params.
        """
        
        # Get params struct
        params = AbstractRegistration._defaultParams(self)
        
        # Set default mapping
        params.mapping = 'backward' 
        
        # How transformations are combined
        params.combine_deforms = 'compose'
        
        # Parameters related to scale
        params.smooth_scale = True # if False uses factors of two
        params.scale_levels = 5
        params.final_scale = 1.0
        params.scale_sampling = 15
        
        # Done
        return params
    
    
    ## Methods and props to help the algorithm
    
    
    def _set_delta_deform(self, i, deform):
        """ _set_delta_deform(i, deform)
        
        Append the given delta deform for image i. It is combined with the
        current deformation for that image.
        
        """
        
        # Assume null deform
        if deform is None:
            return
        elif deform.is_identity:
            return
        
        # Get how to combine
        combine = self.params.combine_deforms.lower()
        
        # Get current (or null-deform)
        current = self._deforms.get(i, None)
        if current is None:
            current = self.DeformationField(self._ims[0].ndim)
        
        self.timer.start('Setting delta deforms')
        
        # Reshape
        current = current.resize_field(deform)
        
        # Track deformation amplitude?
        if hasattr(self, '_deform_tracking'):
            ff = []
            for f in deform:
                if not isinstance(f, np.ndarray):
                    f = f.knots
                ff.append(f)
            f = (ff[0]**2 + ff[1]**2) ** 0.5
            if i==0:
                self._deform_tracking.append(f.mean())
        
        # Combine
        if combine == 'compose':
            totalDeform = current.compose(deform)
        elif combine == 'add':
            totalDeform = current.add(deform)
        else:
            raise RuntimeError('param combine_deforms is invalid: %s' %
 str(combine))
        
        self.timer.stop('Setting delta deforms')
        
        # Store
        self._deforms[i] = totalDeform
    
    
    def get_deformed_image(self, i, s=0):
        """ get_deformed_image(i, s=0)
        
        Get the image i at scale s, deformed with its current deform. 
        Mainly intended for the registration algorithms, but can be of interest
        during development/debugging.
        
        """
        # Get image
        self.timer.start('getting raw images from pyramid')
        if not s:
            s = None
        im = self._pyramids[i].get_scale(s)
        self.timer.stop('getting raw images from pyramid')
        
        # Deform it
        self.timer.start('deforming raw images')
        deform = self._deforms.get(i, None)
        if deform is not None:
            deform = deform.resize_field(im)
            self._deforms[i] = deform # store, so we do not need to resize later
            im = deform.apply_deformation(im, self._current_interp_order)
        self.timer.stop('deforming raw images')
        
        # Done
        return im
    
    
    def _set_buffered_data(self, key1, key2, data):
        """ _set_buffered_data(key1, key2, data)
        
        Buffer the given data. key1 is where the data is stored under.
        key2 is a check. The most likely use case is using the image
        number as key1 and the scale as key2.
        
        Intended for the registration algorithm subclasses.
        
        """        
        self._buffer[key1] = key2, data
    
    
    def _get_buffered_data(self, key1, key2):
        """ _get_buffered_data(key1, key2)
        
        Retrieve buffered data.
        
        """
        key2_data = self._buffer.get(key1, None)
        if key2_data and key2_data[0] == key2:
            return key2_data[1]
        else:
            return None
    
    
    ## The actual methods
    
    def _register(self, verbose=1, fig=None):
        
        # For an illustration of the scale sampling, see the script
        # _smooth_scale_sampling.py.
        
        # Init visualizer
        self.visualizer.init(fig)
        
        # Init progress display
        if verbose >= 1:
            self.progress.start('%s: '% self.__class__.__name__)
        
        # Init interpolation order
        self._current_interp_order = 1
        
        # Scale parameters
        final_scale = float(self.params.final_scale)
        scale_sampling = int(self.params.scale_sampling)
        smooth_scale = bool(self.params.smooth_scale)
        
        # Calculate iter factor for smooth scale space
        iter_factor = 0.5**(1.0/scale_sampling)
        
        # Check
        pixel_scales = [final_scale/s for s in self._ims[0].sampling]   
        if min(pixel_scales) < 0.5:
            raise ValueError('final_scale expressed in pixel units should be at least 0.5.')
        
        # Create pyramids, using final_scale as an offset
        self._pyramids = [ScaleSpacePyramid(im, final_scale)
                                for im in self._ims]
        
        # Calculate max scale
        ranges = [sh*sa for sh, sa in zip(self._ims[0].shape, self._ims[0].sampling)]
        self._max_scale = max_scale = max(ranges) * 0.25
        
        # Calculate scale levels
        scale_levels = 1
        while final_scale * 2**(scale_levels-1) < max_scale:
            scale_levels += 1
        
        # Main loop
        for level in reversed(range(scale_levels)):
            
            # Set (initial) scale for this level
            scale = final_scale * 2**level
            if smooth_scale:
                scale *= 2 * iter_factor
            
            for iter in range(1, scale_sampling+1):
                
                # Skip highest scale
                if smooth_scale and level >= scale_levels-1:
                    continue
                
                # Set interpolation order higher in the final iterations
                if level==0 and iter>0.75*scale_sampling:
                    self._current_interp_order = 3
                
                # Do one iteration
                iterInfo = (level, iter, scale)
                self._register_iteration(iterInfo)
                
                # Print iteration info
                if verbose==1:
                    self.progress.next('(%i-%i) scale %1.2f' % iterInfo)
                elif verbose>1:
                    print("Registration iter %i-%i at scale %1.2f" % iterInfo)
                
                # Next iteration. When using a smooth scale space
                # the final scale is reached sooner.
                if smooth_scale:
                    scale = max(final_scale, scale*iter_factor)
    
    
    def _register_iteration(self, iterInfo):
        """ _register_iteration(iterInfo)
        
        Apply one iteration of the registration algorithm at 
        the specified scale (iterInfo[2]).
        
        """
        
        nims = len(self._ims)
        
        # Calculate deformation for each image
        deforms = []
        for i in range(nims):
            deform = self._deform_for_image(i, iterInfo)
            deforms.append(deform)
        
        # Apply deformations
        for i in range(nims):
            self._set_delta_deform(i, deforms[i])
    
    
    def _deform_for_image(self, i, iterInfo):
        """ _deform_for_image(i, iterInfo)
        
        Calculate the deform for the given image index. 
        
        """
        raise NotImplemented()


class GDGRegistration(BaseRegistration):
    """ GDGRegistration(*images)
    
    Inherits from :class:`pirt.BaseRegistration`.
    
    Generic Groupwise Diffeomorphic Registration. Abstract class that 
    provides a generic way to perform diffeomorphic groupwise registration.
    
    Parameters
    ----------
    deform_wise : {'groupwise', 'pairwise'}
        Whether all images are deformed simultaneously, or only the
        first image is deformed. When registering more than 2 images, 
        'groupwise' registration should be used. Default 'groupwise'.
    injective : bool
        Whether the injectivity constraint should be used. This value should
        only be set to False in specific (testing?) situation; the resulting
        deformations are only guaranteed to be diffeomorphic if injective=True.
    frozenEdge : bool
        Whether the deformation is set to zero at the edges. If True (default)
        the resulting deformation fields are *fully* diffeomorphic; no pixels
        are mapped from the inside to the outside nor vice versa.
    
    final_grid_sampling : scalar
        The grid sampling of the grid at the final level. During the 
        registration process, the B-spine grid sampling scales along 
        with the scale.
    grid_sampling_factor : scalar between 0 and 1
        To what extent the grid sampling scales with the scale. By making
        this value lower than 1, the grid is relatively fine at the the
        higher scales, allowing for more deformations. The default is 0.5.
        Note that setting this value to 1 when using 'frozenedge' can cause
        the image to be 'stuck' at higher scales.
    deform_limit : scalar
        If injective is True, the deformations at each iteration are 
        constraint by a "magic" limit. By making this limit tighter
        (relative to the scale), the deformations stay in reasonable bounds.
        This feature helps a lot for convergence. Default value is 1.
    """
    
    # Implements:
    #   * _regularize_diffeomorphic(scale, deform, weight=None)
    #   * get_final_deform(i, j, mapping)
    #   * _deform_for_image(i, iterInfo)
    # 
    # Inherting methods should implement:
    #   * _deform_for_image_pair(i, i, iterInfo)
    
    def _defaultParams(self):
        """ Overload to create all default params.
        """
        
        # Get params struct
        params = BaseRegistration._defaultParams(self)
        
        # Parameters related to diffeomorpic deforms and groupwise registration
        params.deform_wise = 'groupwise'
        params.injective = True
        params.frozenedge = True
        
        # Parameters related to B-spline based regularization
        params.final_grid_sampling = 16
        params.grid_sampling_factor = 0.5
        params.deform_limit = 1.0
        
        # Done
        return params
    
    
    def _get_grid_sampling_old(self, scale):
        # This method is stupid, as changing the final_scale changes
        # the grid sampling in a very unexpected ways. This is why
        # the final_scale and grid_sampling were so dependent in previous
        # versions.
        
        # Get values from params
        final_grid_sampling = float(self.params.final_grid_sampling)
        grid_factor = float(self.params.grid_sampling_factor)
        
        # Get factor and bias
        gsf = grid_factor * final_grid_sampling / self.params.final_scale
        gsb = (1-grid_factor) * final_grid_sampling
        
        # Compute!
        grid_sampling = scale * gsf + gsb
        return grid_sampling
    
    def _get_grid_sampling(self, scale):
        
        # Get values from params
        final_grid_sampling = float(self.params.final_grid_sampling)
        grid_factor = float(self.params.grid_sampling_factor)
        final_scale = float(self.params.final_scale)
        
        # Get factor and bias
        gsf = grid_factor * final_grid_sampling
        gsb = final_grid_sampling
        
        # Compute!
        grid_sampling = (scale-final_scale) * gsf + gsb
        return grid_sampling
    
    def _get_grid_sampling_full(self, scale):
        # One can determine a max grid sampling from the image shape, and
        # scale linearly between this and the final_grid_sampling, 
        # but this turns out not to work so nice, because it is so 
        # dependent on image shape etc.
        
        # Calculate max sampling (4x max_scale seems nice because we have 4 knots)
        max_scale = self._max_scale
        max_grid_sampling = max_scale * 4.0 
        
        # Get values from params
        final_grid_sampling = float(self.params.final_grid_sampling)
        final_scale = float(self.params.final_scale)
        
        # Compute!
        t = max(0.0,scale-final_scale) / (max_scale - final_scale)
        grid_sampling = t*max_grid_sampling + (1.0-t)*final_grid_sampling
        return grid_sampling
    
    
    def _regularize_diffeomorphic(self, scale, deform, weight=None):
        """ _regularize_diffeomorphic(scale, deform, weight=None)
        
        Regularize the given DeformationField in a way that makes it
        diffeomorphic. Returns the result as a DeformationGrid.
        
        """
        
        # Check
        if not isinstance(deform, DeformationField):
            raise ValueError('make_diffeomorphic needs a DeformationField.') 
        
        # Get grid sampling
        grid_sampling = self._get_grid_sampling(scale)
        
        # Calculate factor to limit the deformation
        injective, frozenedge = self.params.injective, self.params.frozenedge
        if injective:
            deform_limit = float(self.params.deform_limit) 
            injective = deform_limit * scale / grid_sampling
            injective = min(injective, 0.9) # must not be higher than 1!
        
        # Regularize using a B-spline grid
        self.timer.start('regularizing')
        grid = self.DeformationGrid.from_field(deform,  grid_sampling, weight,
                                                    injective, frozenedge)
        self.timer.stop('regularizing')
        
        return grid
    
    
    def _deform_for_image(self, i, iterInfo):
        """ _deform_for_image(i, iterInfo)
        
        Calculate the deform for the given image index. 
        
        """
        scale = iterInfo[2]
        
        # Check whether the grid would be too small anyway
        if self.params.frozenedge:
            # Get grid sampling
            grid_sampling = self._get_grid_sampling(scale)
            # Create dummy grid
            testGrid = SplineGrid(self._ims[0], grid_sampling)
            # Check
            if all( [s<4 for s in testGrid.grid_shape] ):
                print('skip because grid too small')
                return None
        
        # Get wise
        deform_wise = self.params.deform_wise.lower()
        
        # Decide which routine to use
        if deform_wise in ['pairwise', 'pairwise1']:
            return self._deform_for_image_pairwise1(i, iterInfo)
        elif deform_wise == 'pairwise2':
            return self._deform_for_image_pairwise2(i, iterInfo)
        elif deform_wise == 'groupwise':
            return self._deform_for_image_groupwise(i, iterInfo)
        else:
            raise ValueError('Invalid value for param deform_wise: %s' % repr(deform_wise))
    
    
    def _deform_for_image_groupwise(self, i, iterInfo):
        """ _deform_for_image_groupwise(i, iterInfo)
        
        Calculate the deform for the given image index. For each image,
        the deform between that image and all other images is 
        calculated. The total deform is the average of these deforms.
        
        """
        
        # todo: Suggestion by DJ: weight the contribution of the 
        # "nearest" images more.
        
        # Init deform
        totalDeform = DeformationIdentity()
        nims = len(self._ims)
        count = 0
        
        # Collect deform
        for j in range(nims):
            if i==j:
                continue
            # Get deform
            deform = self._deform_from_image_pair(i, j, iterInfo)
            
            # Add to total
            if deform is not None:
                count += 1
                totalDeform = totalDeform + deform
            
        #print 'totalDeform', totalDeform.__class__
        
        # Almost done
        if count > 1:
            totalDeform = totalDeform.scale(1.0/count)
        
        # Done
        return totalDeform
    
    
    def _deform_for_image_pairwise1(self, i, iterInfo):
        """ _deform_for_image_pairwise1(i, iterInfo)
        
        Get the deform for the image only if i is 0; the source image.
        This is what's classic registration does.
        
        """
        # Only for source to target
        if i != 0:
            return None
        
        # Return deform for image 0 only
        return self._deform_from_image_pair(0, 1, iterInfo)
    
    
    def _deform_for_image_pairwise2(self, i, iterInfo):
        """ _deform_for_image_pairwise2(i, iterInfo)
        
        Get the deform for the image only if i is 1.
        
        """
        # Only for source to target
        if i != 1:
            return None
        
        # Return deform for image 0 only
        return self._deform_from_image_pair(1, 0, iterInfo)

