"""
The gaussfun module implements functions for diffusion and Gaussian
derivatives for data of any dimension.

Contents of this module:

  * gaussiankernel - Create a Gaussian kernel
  * gaussiankernel2 - Create a 2D Gaussian kernel
  * diffusionkernel - Create a discrete analog to the Gaussian kernel
  * gfilter - Filter data with Gaussian (derivative) kernels
  * diffuse - Filter data with true discrete diffusion kernels
  * gfilter2 - Filter data by specifying sigma in world coordinates
  * diffuse2 - Diffuse data by specifying sigma in world coordinates

"""

# SHORTED VERSION WITHOUT PYRAMID STUFF

import numpy as np
import scipy.ndimage

from . import Aarray


## Kernels

def _gaussiankernel(sigma, order, t):
    """ _gaussiankernel(sigma, order, t)
    Calculate a Gaussian kernel of the given sigma and with the given
    order, using the given t-values. 
    """
    
    # if sigma 0, kernel is a single 1.
    if sigma==0:
        return np.array([1.0])
    
    # precalculate some stuff
    sigma2 = sigma**2
    sqrt2  = np.sqrt(2)
    
    # Calculate the gaussian, it is unnormalized. We'll normalize at the end.
    basegauss = np.exp(- t**2 / (2*sigma2) )
    
    # Scale the t-vector, what we actually do is H( t/(sigma*sqrt2) ), 
    # where H() is the Hermite polynomial. 
    x = t / (sigma*sqrt2)
    
    # Depending on the order, calculate the Hermite polynomial (physicists 
    # notation). We let Mathematica calculate these, and put the first 20 
    # orders in here. 20 orders should be sufficient for most tasks :)
    if order<0: 
        raise Exception("The order should not be negative!")    
    elif order==0:
        part = 1
    elif order==1:
        part = 2*x
    elif order==2:
        part = -2 + 4*x**2
    elif order==3:
        part = -12*x + 8*x**3
    elif order==4:
        part = 12 - 48*x**2 + 16*x**4
    elif order==5:
        part = 120*x - 160*x**3 + 32*x**5
    elif order==6:
        part = -120 + 720*x**2 - 480*x**4 + 64*x**6
    elif order==7:  
        part = -1680*x + 3360*x**3 - 1344*x**5 + 128*x**7
    elif order==8:  
        part = 1680 - 13440*x**2 + 13440*x**4 - 3584*x**6 + 256*x**8
    elif order==9:  
        part = 30240*x - 80640*x**3 + 48384*x**5 - 9216*x**7 + 512*x**9
    elif order==10: 
        part = (-30240 + 302400*x**2 - 403200*x**4 + 161280*x**6 - 23040*x**8 
                + 1024*x**10)
    elif order==11: 
        part = (-665280*x + 2217600*x**3 - 1774080*x**5 + 506880*x**7 
                - 56320*x**9 + 2048*x**11)
    elif order==12: 
        part = (665280 - 7983360*x**2 + 13305600*x**4 - 7096320*x**6 
                + 1520640*x**8 - 135168*x**10 + 4096*x**12)
    elif order==13: 
        part = (17297280*x - 69189120*x**3 + 69189120*x**5 - 26357760*x**7 
                + 4392960*x**9 - 319488*x**11 + 8192*x**13)
    elif order==14: 
        part = (-17297280 + 242161920*x**2 - 484323840*x**4 + 322882560*x**6 
                - 92252160*x**8 + 12300288*x**10 - 745472*x**12 + 16384*x**14)
    elif order==15: 
        part = (-518918400*x + 2421619200*x**3 - 2905943040*x**5 
                + 1383782400*x**7 - 307507200*x**9 + 33546240*x**11 
                - 1720320*x**13 + 32768*x**15)
    elif order==16: 
        part = (518918400 - 8302694400*x**2 + 19372953600*x**4 
                - 15498362880*x**6 + 5535129600*x**8 
                - 984023040*x**10 + 89456640*x**12 - 3932160*x**14 
                + 65536*x**16)    
    else:
        raise Exception("This order is not implemented!")
        
    
    # Apply Hermite polynomial to gauss
    k = (-1)**order * part * basegauss
    
    ## Normalize
    
    # By calculating the normalization factor by integrating the gauss, rather
    # than using the expression 1/(sigma*sqrt(2pi)), we know that the KERNEL
    # volume is 1 when the order is 0.
    norm_default = 1 / basegauss.sum()
    #           == 1 / ( sigma * sqrt(2*pi) )

    # Here's another normalization term that we need because we use the Hermite
    # polynomials.
    norm_hermite = 1/(sigma*sqrt2)**order

    # A note on Gaussian derivatives: as sigma increases, the resulting
    # image (when smoothed) will have smaller intensities. To correct for
    # this (if this is necessary) a diffusion normalization term can be
    # applied: sigma**2

    # Normalize and return
    return k * ( norm_default * norm_hermite )



def gaussiankernel(sigma, order=0, N=None, returnt=False, warn=True):
    """ gaussiankernel(sigma, order=0, N=None, returnt=False, warn=True)
    
    Creates a 1D gaussian derivative kernel with the given sigma
    and the given order. (An order of 0 is a "regular" Gaussian.)
    
    The returned kernel is a column vector, thus working in the first 
    dimension (in images, this often is y). 
    
    The returned kernel is odd by default. Using N one can specify the
    full kernel size (if not int, the ceil operator is applied). By 
    specifying a negative value for N, the tail length (number of elements
    on both sides of the center element) can be specified.
    The total kernel size than becomes ceil(-N)*2+1. Though the method
    to supply it may be a bit obscure, this measure can be handy, since 
    the tail length if often related to the sigma. If not given, the 
    optimal N is determined automatically, depending on sigma and order. 
    
    If the given scale is a small for the given order, a warning is
    produced (unless warn==True).
    
    ----- Used Literature:

    Koenderink, J. J. 
    The structure of images. 
    Biological Cybernetics 50, 5 (1984), 363-370.

    Lindeberg, T. 
    Scale-space for discrete signals. 
    IEEE Transactions on Pattern Analysis and Machine Intelligence 12, 3 (1990), 234-254.

    Ter Haar Romeny, B. M., Niessen, W. J., Wilting, J., and Florack, L. M. J. 
    Differential structure of images: Accuracy of representation.
    In First IEEE International Conference on Image Processing, (Austin, TX) (1994).
    """
    
    # Check inputs
    if not N:
        # Calculate ratio that is small, but large enough to prevent errors
        ratio = 3 + 0.25 * order - 2.5/((order-6)**2+(order-9)**2)
        # Calculate N
        N = int( np.ceil( ratio*sigma ) ) * 2 + 1
    
    elif N > 0:
        if not isinstance(N, int):
            N = int( np.ceil(N) )
    
    elif N < 0:
        N = -N
        if not isinstance(N, int):
            N = int( np.ceil(N) )
        N = N * 2 + 1
    
    # Check whether given sigma is large enough 
    sigmaMin = 0.5 + order**(0.62) / 5
    if warn and sigma < sigmaMin:
        print('WARNING: The scale (sigma) is very small for the given order, '
              'better use a larger scale!')
    
    # Create t vector which indicates the x-position
    t = np.arange(-N/2.0+0.5, N/2.0, 1.0, dtype=np.float64)
    
    # Get kernel
    k = _gaussiankernel(sigma, order, t)
    
    # Done
    if returnt:
        return k, t
    else:
        return k



def gaussiankernel2(sigma, ox, oy, N=None):
    """ gaussiankernel2(sigma, ox, oy, N=-3*sigma)
    Create a 2D Gaussian kernel.
    """
    # Default N
    if N is None:
        N = -3*sigma
    
    # Calculate kernels
    k1 = gaussiankernel(sigma, ox, N)
    k2 = gaussiankernel(sigma, oy, N)
    
    # Matrix multiply
    k = np.matrix(k1).T * np.matrix(k2)
    return k.A 


def diffusionkernel(sigma, N=4, returnt=False):
    """ diffusionkernel(sigma, N=4, returnt=False)
    
    A discrete analog to the continuous Gaussian kernel, 
    as proposed by Toni Lindeberg.
    
    N is the tail length factor (relative to sigma).
    
    """
    
    # Make sure sigma is float
    sigma = float(sigma)
    
    # Often refered to as the scale parameter, or t
    sigma2 = sigma*sigma 
    
    # Where we start, from which we go backwards
    # This is also the tail length
    if N > 0:
        nstart = int(np.ceil(N*sigma)) + 1
    else:
        nstart  = abs(N) + 1
    
    # Allocate kernel and times
    t = np.arange(-nstart, nstart+1, dtype='float64')
    k = np.zeros_like(t)
    
    # Make a start
    n = nstart # center (t[nstart]==0)
    k[n+nstart] = 0
    n = n-1
    k[n+nstart] = 0.01
    
    # Iterate!
    for n in range(nstart-1,0,-1):   
        # Calculate previous
        k[(n-1)+nstart] = 2*n/sigma2 * k[n+nstart] + k[(n+1)+nstart]
    
    # The part at the left can be erroneous, so let's use the right part only
    k[:nstart] = np.flipud(k[-nstart:])
    
    # Remove the tail, which is zero
    k = k[1:-1]
    t = t[1:-1]
    
    # Normalize
    k = k / k.sum()
    
    # the function T that we look for is T = e^(-sigma2) * I(n,sigma2)
    # We found I(n,sigma2) and because we normalized it, the normalization term
    # e^(-sigma2) is no longer necesary.
    
    # Done
    if returnt:
        return k, t
    else:
        return k


## Filters

def gfilter(L, sigma, order=0, mode='constant', warn=True):
    """ gfilter(L, sigma, order=0, mode='constant', warn=True)
    
    Gaussian filterering and Gaussian derivative filters.
    
    Parameters
    ----------
    L : np.ndarray
        The input data to filter
    sigma : scalar or list-of-scalars
        The smoothing parameter, can be given for each dimension
    order : int or list-of-ints
        The order of the derivative, can be given for each dimension
    mode : {'reflect','constant','nearest','mirror', 'wrap'}
        Determines how edge effects are handled. (see scipy.ndimage.convolve1d)
    warn : boolean
        Whether to show a warning message if the sigma is too small to 
        represent the required derivative.
    
    Notes
    =====
    Makes use of the seperability property of the Gaussian by convolving
    1D kernels in each dimension. 
    
    
    Example
    =======
    # Calculate the second order derivative with respect to x (Lx) (if the
    # first dimension of the image is Y).
    result1 = gfilter( im, 2, [0,2] ) 
    # Calculate the first order derivative with respect to y and z (Lyz).
    result2 = gfilter( volume, 3, [0,1,1] ) 
    
    """
    
    # store original
    Lo = L
    
    # make sigma ok
    try:
        sigma = [sig for sig in sigma]
    except TypeError:
        sigma = [sigma for i in range(L.ndim)]
        
    # same for order
    if order is None:
        order = 0
    try:
        order = [o for o in order]
    except TypeError:
        order = [order for i in range(L.ndim)]
        
    # test sigma
    if len(sigma) != L.ndim:
        tmp = "the amount of sigmas given must match the dimensions of L!"
        raise Exception(tmp)    
    # test order
    if len(order) != L.ndim:
        tmp = "the amount of sigmas given must match the dimensions of L!"
        raise Exception(tmp)
    
    for d in range(L.ndim):
        # get kernel
        k = gaussiankernel(sigma[d], order[d], warn=warn)
        # convolve
        L = scipy.ndimage.convolve1d(L, k, d, mode=mode)
    
    
    # Make Aarray if we can
    if hasattr(Lo, 'sampling') and hasattr(Lo, 'origin'):
        L = Aarray(L, Lo.sampling, Lo.origin)
    
    # Done
    return L


def diffuse(L, sigma, mode='nearest'):
    """ diffuse(L, sigma)
    
    Diffusion using a discrete variant of the diffusion operator. 
    
    Parameters
    ----------
    L : np.ndarray
        The input data to filter
    sigma : scalar or list-of-scalars
        The smoothing parameter, can be given for each dimension
    
    Details
    -------
    In the continous domain, the Gaussian is the only true diffusion
    operator. However, by using a sampled Gaussian kernel in the 
    discrete domain, errors are introduced, particularly if for
    small sigma. 
    
    This implementation uses a a discrete variant of the diffusion
    operator, which is based on modified Bessel functions. This results
    in a better approximation of the diffusion process, particularly
    when applying the diffusion recursively. There are also advantages
    for calculating derivatives, see below.
    
    Based on:
    Lindeberg, T. "Discrete derivative approximations with scale-space
    properties: A basis for low-level feature extraction", 
    J. of Mathematical Imaging and Vision, 3(4), pp. 349--376, 1993.
    
    Calculating derivatives
    -----------------------
    Because this imeplementation applies diffusion using a discrete 
    representation of the diffusion kernel, one can calculate true
    derivatives using small-support derivative operators. For 1D:
      * Lx = 0.5 * ( L[x+1] - L[x-1] )
      * Lxx = L[x+1] - 2*L[x] + L(x-1)
    
    """
    
    # Store original
    Lo = L
    
    # Make sigma ok
    try:
        sigma = [sig for sig in sigma]
    except TypeError:
        sigma = [sigma for i in range(L.ndim)]
    
    # Test sigma
    if len(sigma) != L.ndim:
        tmp = "the amount of sigmas given must match the dimensions of L!"
        raise Exception(tmp)    
    
    # Diffuse
    for d in range(L.ndim):
        # get kernel
        k = diffusionkernel(sigma[d])
        # convolve
        L = scipy.ndimage.convolve1d(L, k, d, mode=mode)
    
    # Make Aarray if we can
    if hasattr(Lo, 'sampling') and hasattr(Lo, 'origin'):
        L = Aarray(L, Lo.sampling, Lo.origin)
    
    # Done
    return L


def gfilter2(L, scale, order=0, mode='reflect', warn=True):
    """ gfilter2(L, scale, order=0, mode='reflect', warn=True)
    
    Apply Gaussian filtering by specifying a scale in world coordinates
    rather than a sigma. This function determines the sigmas to apply,
    based on the sampling of the elements.
    
    See gfilter for more information.
    
    (If L is not an Aarray, this function yields the same result as gfilter.)
    
    """
    
    # Determine sigmas
    if hasattr(L, 'sampling'):
        sigmas = [float(scale)/s for s in L.sampling]
    else:
        sigmas = float(scale)
    
    # Filter
    return gfilter(L, sigmas, order, mode, warn)


def diffuse2(L, scale, mode='nearest'):
    """ diffuse2(L, scale, mode='nearest')
    
    Apply diffusion by specifying a scale in world coordinates
    rather than a sigma. This function determines the sigmas to apply,
    based on the sampling of the elements.
    
    See diffuse for more information.
    
    (If L is not an Aarray, this function yields the same result as diffuse.)
    
    """
    
    # Determine sigmas
    if hasattr(L, 'sampling'):
        sigmas = [float(scale)/s for s in L.sampling]
    else:
        sigmas = float(scale)
    
    # Filter
    return diffuse(L, sigmas, mode)

