"""
Cubic spline coefficients and lookup tables.
"""

import numpy as np
from numpy import sin, pi  # for Lanczos
import numba

# Keep a cache of calculated luts
LUTS = {}


def get_cubic_spline_coefs(t, spline_type=0.0):
    """ get_cubic_spline_coefs(t, spline_type='Catmull-Rom')
    
    Calculates the coefficients for a cubic spline and returns them as 
    a tuple. t is the ratio between "left" point and "right" point on the
    lattice.  If performance matters, consider using get_lut() instead.
    
    spline_type can be (case insensitive):
        
        <number between -1 and 1>: Gives a Cardinal spline with the
        specified number as its tension parameter. A Cardinal spline 
        is a type of Hermite spline, where the tangents are calculated 
        using points p0 and p3; the coefficients can be directly applied
        to p0 p1 p2 p3. 
        
        'Catmull-Rom' or 'Cardinal0': is a cardinal spline a tension of 0.
        An interesting note: if we would create two quadractic splines, that
        would fit a polynomial "f(t) = a*t*t + b*t + c" to the first three
        and last three knots respectively, and if we would then combine 
        the two using linear interpolation, we would obtain a catmull-rom
        spline. I don't know whether this is how the spline was designed,
        or if this is a "side-effect".
        
        'B-pline': Basis spline. Not an interpolating spline, but an
        approximating spline. Here too, the coeffs can be applied to p0-p3.
        
        'Hermite': Gives the Hermite coefficients to be applied to p1 m1 p2
        m2 (in this order), where p1 p2 are the closest knots and m1 m2 the
        tangents.
        
        'Lagrange': The Lagrange spline or Lagrange polynomial is an 
        interpolating spline. It is the same as Newton polynomials, but
        implemented in a different manner (wiki). The Newton implementation
        should be more efficient, but this implementation is much simpler, 
        and is very similar to the B-spline implementation (only the 
        coefficients are different!). Also, when for example interpolating 
        an image, coefficients are reused often and can be precalculated
        to enhance speed.
        
        'Lanczos': Lanczos interpolation (windowed sync funcion). Note that
        this is not really a spline, and that sum of the coefficients is
        not exactly 1. Often used in audio processing. The Lanczos spline is
        very similar to the Cardinal spline with a tension of -0.25.
        
        'quadratic': Quadratic interpolation with a support of 4, essentially
        the addition of the two quadratic polynoms.
        
        'linear': Linear interpolation. Effective support is 2. Added
        for completeness and testing.
        
        'nearest': Nearest neighbour interpolation. Added for completeness
        and testing.
    
    """
    spline_id = spline_type_to_id(spline_type)
    out = np.zeros((4, ), np.float64)
    set_cubic_spline_coefs(t, spline_id, out)
    return tuple(out)


@numba.jit(nopython=True)
def set_cubic_spline_coefs(t, spline_id, out):
    """ set_cubic_spline_coefs(t, spline_id, out)
    
    Calculate cubuc spline coefficients for the given spline_id, and
    store them in the given array. See get_cubic_spline_coefs() and
    spline_type_to_id() for details.
    """
    #if spline_id == 0.0:
    #    cubicsplinecoef_catmullRom(t, out)  == cubicsplinecoef_cardinal(t, out, 0)
    if spline_id <= 1.0:
        cubicsplinecoef_cardinal(t, out, spline_id)  # tension=spline_id
    elif spline_id == 2.0:
        cubicsplinecoef_basis(t, out)
    elif spline_id == 3.0:
        cubicsplinecoef_hermite(t, out)
    elif spline_id == 4.0:
        cubicsplinecoef_lagrange(t, out)
    elif spline_id == 5.0:
        cubicsplinecoef_lanczos(t, out)  # common in Audio, but slow because of sine
    elif spline_id == 97.0:
        cubicsplinecoef_nearest(t, out)
    elif spline_id == 98.0:
        cubicsplinecoef_linear(t, out)
    elif spline_id == 99.0:
        cubicsplinecoef_quadratic(t, out)


# Note: the lookup table was initially implemented to provide efficient
# calculation of the coefficient in the warp functions. However, with Numba
# it turned out to be more efficient to calculate the coefficients directly,
# possibly by LLVM making use of SIMD and/or the overhead of array management
# needed with a LUT.
def get_lut(spline_type, n=32768):
    """ get_lut(spline_type, n=32768)
    
    Calculate the look-up table for the specified spline type
    with n entries. Returns a float64 1D array that has a size of (n + 2 * 4)
    that can be used in get_coef().
    
    The spline_type can be 'cardinal' or a float between -1 and 1 for
    a Cardinal spline, or 'hermite', 'lagrange', 'lanczos', 'quadratic',
    'linear', 'nearest'. Note that the last three are available for
    completeness; its inefficient to do nearest, linear or quadratic
    interpolation with a cubic kernel.
    """
    
    spline_id = spline_type_to_id(spline_type)
    
    # Create lut if not existing yet
    key = spline_id, n
    if key not in LUTS.keys():
        lut = np.zeros((n + 2) * 4, np.float64)
        _calculate_lut(lut, spline_id)
        LUTS[key] = lut
    
    return LUTS[key]


@numba.jit
def _calculate_lut(lut, spline_id):
    
    n = lut.size // 4 - 2
    step = 1.0 / n
    t = 0.0
    
    for i in range(n):
        t += step
        out = lut[i * 4:]
        # For each possible t, calculate the coefficients
        set_cubic_spline_coefs(t, spline_id, out)


def spline_type_to_id(spline_type):
    """ spline_type_to_id(spline_type)
    
    Method to map a spline name to an integer ID. This is used so that
    set_cubic_spline_coefs() can be efficient.
    
    The spline_type can also be a number between -1 and 1, representing
    the tension for a Cardinal spline.
    """
    
    # Handle tension given for Cardinal spline
    tension = 0.0
    if isinstance(spline_type, (float, int)):           
        if spline_type >= -1 and spline_type <= 1:
            tension = float(spline_type)
            spline_type = 'Cardinal'
        else:
            raise ValueError('Tension parameter must be between -1 and 1.')
    
    # Get id
    if spline_type.lower() in ['c', 'card', 'cardinal', 'catmull–rom']:
        return tension # For catmull-rom, we use default tension 0
    elif spline_type.lower() in ['b', 'basis', 'basic']:
        return 2.0
    elif spline_type.lower() in ['herm', 'hermite']:
        return 3.0
    elif spline_type.lower() in ['lag', 'lagrange']:
        return 4.0
    elif spline_type.lower() in ['lanc', 'lanczos']:
        return 5.0
    elif spline_type.lower() in ['near', 'nearest']:
        return 97.0    
    elif spline_type.lower() in ['lin', 'linear']:
        return 98.0        
    elif spline_type.lower() in ['quad', 'quadratic']:
        return 99.0
    else:
        raise ValueError('Unknown spline type: ' + str(spline_type))


@numba.jit(nopython=True)
def get_coef(lut, t):
    """ get_coef(lut, t)
    
    Get the coefficients for given value of t. This simply obtains
    the nearest coefficients in the table. For a more accurate result,
    use the AccurateCoef class.
    """
    n = lut.size // 4 - 2
    i1 = int(t * n + 0.5) * 4
    return lut[i1:i1 + 4]


## The coefficient functions

@numba.jit(nopython=True)
def cubicsplinecoef_catmullRom(t, out):
    # See the doc for the catmull-rom spline, this is how the two splines
    # are combined by simply adding (and dividing by two) 
    out[0] = - 0.5*t**3 + t**2 - 0.5*t
    out[1] =   1.5*t**3 - 2.5*t**2 + 1
    out[2] = - 1.5*t**3 + 2*t**2 + 0.5*t
    out[3] =   0.5*t**3 - 0.5*t**2


@numba.jit(nopython=True)
def cubicsplinecoef_cardinal(t, out, tension):
    tau = 0.5 * (1 - tension)
    out[0] = - tau * (   t**3 - 2*t**2 + t )
    out[3] =   tau * (   t**3 -   t**2     )
    out[1] =           2*t**3 - 3*t**2 + 1  - out[3]
    out[2] = -         2*t**3 + 3*t**2      - out[0]


@numba.jit(nopython=True)
def cubicsplinecoef_basis(t, out):
    out[0] = (1-t)**3                     /6.0
    out[1] = ( 3*t**3 - 6*t**2 +       4) /6.0
    out[2] = (-3*t**3 + 3*t**2 + 3*t + 1) /6.0
    out[3] = (  t)**3                     /6.0


@numba.jit(nopython=True)
def cubicsplinecoef_hermite(t, out):
    out[0] =   2*t**3 - 3*t**2 + 1
    out[1] =     t**3 - 2*t**2 + t
    out[2] = - 2*t**3 + 3*t**2
    out[3] =     t**3 -   t**2


@numba.jit(nopython=True)
def cubicsplinecoef_lagrange(t, out):
    k = -1.0  
    out[0] =               (t  )/(k  ) * (t-1)/(k-1) * (t-2)/(k-2)
    k= 0  
    out[1] = (t+1)/(k+1) *               (t-1)/(k-1) * (t-2)/(k-2)
    k= 1  
    out[2] = (t+1)/(k+1) * (t  )/(k  ) *               (t-2)/(k-2)
    k= 2  
    out[3] = (t+1)/(k+1) * (t  )/(k  ) * (t-1)/(k-1)


@numba.jit(nopython=True)
def cubicsplinecoef_lanczos(t, out):
    tt= (1+t)
    out[0] = 2*sin(pi*tt)*sin(pi*tt/2) / (pi*pi*tt*tt)
    tt= (2-t)  
    out[3] = 2*sin(pi*tt)*sin(pi*tt/2) / (pi*pi*tt*tt)
    if t!=0:
        tt= t
        out[1] = 2*sin(pi*tt)*sin(pi*tt/2) / (pi*pi*tt*tt)
    else:  
        out[1] =1
    if t!=1:
        tt= (1-t)  
        out[2] = 2*sin(pi*tt)*sin(pi*tt/2) / (pi*pi*tt*tt)  
    else:
        out[2] =1


@numba.jit(nopython=True)
def cubicsplinecoef_nearest(t, out):
    out[0] = 0.0
    out[1] = int(t < 0.5)
    out[2] = int(t >= 0.5)
    out[3] = 0.0


@numba.jit(nopython=True)
def cubicsplinecoef_linear(t, out):
    out[0] = 0.0
    out[1] = (1.0-t)
    out[2] = t
    out[3] = 0.0


@numba.jit(nopython=True)
def cubicsplinecoef_quadratic(t, out):
    # This corresponds to adding the two quadratic polynoms,
    # thus keeping genuine quadratic interpolation. However,
    # it has the same support as a cubic spline, so why use this?
    # Catmull-rom is similar, except it linearly interpolates the two
    # quadratic functions instead of adding them.
    out[0] = 0.25*t**2 - 0.25*t
    out[1] = -0.25*t**2 - 0.75*t + 1
    out[2] = -0.25*t**2 + 1.25*t
    out[3] = 0.25*t**2 - 0.25*t
