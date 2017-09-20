import numpy as np
import pirt

from .gaussfun import diffuse2
from . import Aarray


class BasePyramid:
    pass # When implementing HaarPyramid, maybe use a base class


class HaarPyramid:
    pass # Implement using Haar wavelets.


class ScaleSpacePyramid:
    """ ScaleSpacePyramid(data, min_scale=None, scale_offset=0, 
                                            use_buffer=False, level_factor=2)
    
    The scale space pyramid class provides a way to manage a scale
    space pyramid. Given an input image (of arbitrary dimension),
    it provides two simple methods to obtain the image at the a specified
    scale or level. 
    
    Parameters
    ----------
    data : numpy array
        An array of any dimension. Should preferably be of float type.
    min_scale : scalar, optional
        The minimum scale to sample from the pyramid. If not given, 
        scale_offset is used. If larger than zero, the image is smoothed
        to this scale before creating the zeroth level. If the smoothness
        is sufficient, the data is also downsampled. This makes a registration
        algorithm much faster, because the image data for the final scales
        does not have a unnecessary high resolution. 
    scale_offset : scalar
        The scale of the given data. Use this if the data is already smooth.
        Be careful not to set this value too high, as aliasing artifacts
        may be introduced. Default zero.
    use_buffer : bool
        Whether a result obtained with get_scale() is buffered for later use.
        Only one image is buffered. Default False.
    level_factor : scalar
        The scale distance between two levels. A larger number means saving
        a bit of memory in trade of speed. You're probably fine with 2.0.
    
    Notes
    -----
    Note that this scale space representation handles anisotropic arrays
    and that scale is expressed in world units. 
    
    Note that images at higher levels do not always have a factor 2 sampling 
    difference with the original! This is because the first and last pixel
    are kept the same, and the number of pixels is decreased with factors 
    of two (or almost a factor of two if the number is uneven).
    
    The images always have the same offset though.
    
    We adopt the following concepts:
      * level: the level in the pyramid. Each level is a factor two smaller
        in size (in each dimension) than the previous.
      * scale: the scale in world coordinates
    
    """
    
    def __init__(self, data, min_scale=None, scale_offset=0, 
                use_buffer=False, level_factor=2):
        
        # Make sure data is an anisotropic array
        if not hasattr(data, 'sampling'):
            data = Aarray(data)
        
        # Check scale_offset
        scale_offset = float(scale_offset)
        if scale_offset < 0.0:
            raise ValueError('scale_offset should be >= 0.')
        
        # Check min_scale
        if min_scale is None:
            min_scale = scale_offset
        else:
            min_scale = float(min_scale)
        if min_scale < scale_offset:
            raise ValueError('min_scale should be >= scale_offset.')
        
        # Set lowest level image
        self._initialize_level0(data, min_scale, scale_offset)
        
        # Store level factor 
        self._level_factor = float(level_factor)
        if self._level_factor <= 1:
            raise ValueError('Level factor must be > 1.')
        
        # Buffer to store image for a specific scale
        self._use_buffer = bool(use_buffer)
        self._buffer = None
    
    
    def _initialize_level0(self, data, min_scale, scale_offset):
        """ _initialize_level0(data, min_scale, scale_offset)
        
        Smooth the input image if necessary so it is at min_scale.
        The data is resampled at lower resolution if the scale is 
        high enough.
        
        """
        
        # Make image float
        if data.dtype not in [np.float32, np.float64]:
            data = data.astype('float32')
        
        # Calculate sigma (in world coordinates): amount of smoothing to apply
        sigma1 = scale_offset # scale that the image already has
        sigma2 = min_scale # scale that we want the image to have
        sigma = (sigma2**2 - sigma1**2)**0.5
        
        # Smooth
        if sigma > 0:
            data = diffuse2(data, sigma)
        
        # Get scale in pixel coords
        pixel_scales = [min_scale/s for s in data.sampling]
        
        # Sample at lower rate?
        # This will make the data more isotropic if it was not
        if min_scale > 0:
            # Get zoom factors (should only be <= 1)
            zoom_factors = [min(1, 1.0/s) for s in pixel_scales]
            # Only resample if one dim can be reduced by more than 10%
            if min(zoom_factors) < 0.9:
                data = pirt.interp.zoom(data, zoom_factors, order=3, prefilter=False)
        
        # Set properties
        data._pyramid_scale = min_scale
        data._pyramid_level = 0
        
        # Store
        self._levels = [data]
    
    
    def calculate(self, levels=None, min_shape=None):
        """ calculate(levels=None, min_shape=None)
        
        Create the image pyramid now. Specify either the amount of levels,
        or the minimum shape component of the highest level.        
        If neither levels nor min_shape is given, uses min_shape=8.
        
        Returns (max_level, max_sigma) of the current pyramid.
        
        """
        
        # Check
        if None not in [levels, min_shape]:
            raise ValueError('You cannot specify both levels and min_shape')
        if levels is None and min_shape is None:
            min_shape = 8
        
        # Add levels 
        if levels is None:
            while min(self._levels[-1].shape) >= min_shape*2:
                self._add_Level()
        else:
            while len(self._levels) < levels:
                self._add_Level()
        
        # Return 
        maxLevel = len(self._levels)-1
        maxSigma = self._levels[-1]._pyramid_scale
        return maxLevel, maxSigma
    
    
    def get_scale(self, scale=None):
        """ get_scale(scale)
        
        Get the image at the specified scale (expressed in world units). 
        For higher scales, the image has a smaller shape than the original
        image. If min_scale and scale_offset are not used, a scale of 0 
        represents the original image.
        
        To calculate the result, the image at the level corresponding to
        the nearest lower scale is obtained, and diffused an extra bit
        to obtain the requested scale.
        
        The result is buffered (if the pyramid was instantiated with 
        use_buffer=True), such that calling this function multiple
        times with the same scale is much faster. Only buffers the last 
        used scale.
        
        The returned image has two added properties: _pyramid_scale and
        _pyramid_level, wich specify the image scale and level in the
        pyramid.
        
        """
        
        # Check
        min_scale = self._levels[0]._pyramid_scale
        if scale is None:
            scale = min_scale
        if scale < min_scale:
            raise ValueError("Scale should be at least min_scale (%1.2f)." % min_scale)
        
        # Can we use the buffer?
        if self._buffer and self._buffer[0] == scale:
            return self._buffer[1]
        
        # Determine level offset. We loop untill we are one level too
        # high and then use the level below that.
        level = -1
        baseScale = -1
        while baseScale <= scale:
            level += 1
            baseScale = self.get_level(level)._pyramid_scale
        
        # Correct (we went one level too far)
        level = max(level-1, 0)
        
        # Get data
        data = self.get_level(level)
        
        # Calculate sigma (in world coordinates)
        sigma1 = data._pyramid_scale
        sigma2 = scale
        sigma = (sigma2**2 - sigma1**2)**0.5
        
        # Smooth a bit more
        if sigma > 0:
            data = diffuse2(data, sigma)
            data._pyramid_scale = scale
            data._pyramid_level = level
        
        # Set buffer and return
        if self._use_buffer:
            self._buffer = scale, data
        return data
    
    
    def get_level(self, level):
        """ get_level(level):
        
        Get the image at the specified (integer) level, zero being the 
        lowest level (the original image).
        
        Each level is approximately a factor two smaller in size that the 
        previous level. All levels are buffered.
        
        The returned image has two added properties: _pyramid_scale and
        _pyramid_level, wich specify the image scale and level in the
        pyramid.
        
        """
        
        # Get integer level number and delta
        level_i = int(level)
        
        # Add levels if required
        while level_i >= len(self._levels):
            self._add_Level()
        
        # Get integer level data
        return self._levels[level_i]
    
    
    def _add_Level(self):
        """ _add_Level()
        
        Add a level to the scale space pyramid. 
        
        """
        
        # Get data
        data = self._levels[-1]
        
        # Calculate scales (in world coords) needed to make the pixel-scales 2.0
        scales = [self._level_factor*s for s in data.sampling]
        
        # Calculate the amount of required smoothing (in world coords)
        sigma1 = data._pyramid_scale
        sigma2 = max( max(scales), sigma1*2.0 )
        sigma = (sigma2**2 - sigma1**2)**0.5
        
        # Smooth
        data = diffuse2(data, sigma)
        
        # Downsample (do not take every other sample, because then we will
        # lose the last pixel if the shape is even!)
        if min(data.shape) > 8:
            factor = 1.0/self._level_factor
            data = pirt.interp.zoom(data, factor, order=3, prefilter=False)
        
        # Insert in levels
        data._pyramid_scale = sigma2
        data._pyramid_level = len(self._levels)
        self._levels.append(data)
