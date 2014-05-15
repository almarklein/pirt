""" Cython module interpolation_

Provides functionality for interpolation. 
Support for float32 and float64 data of 1,2 and 3 dimension.
Interpolation using nearest neighbour, linear, quasi-linear,
and cubic interpolation using various (spline) kernels.

Copyright 2010 (C) Almar Klein, University of Twente.

"""

# Cython specific imports
import numpy as np
cimport numpy as np
import cython 

# Enable low level memory management
# from libc.stdlib cimport malloc, free
cdef extern from "stdlib.h": # The cimport does not work on my Linux Laptop
   void free(void* ptr)
   void* malloc(size_t size)


cdef double pi = 3.1415926535897931
cdef extern from "math.h":
    double sin(double val)

# Type defs, we support float32 and float64
FLOAT32 = np.float32
FLOAT64 = np.float64
FLOATi16 = np.int16 # Temporay the most confusion name ever; its just a 16  bit int
SAMPLE = np.float32

# Default amount of coefficients to calculate for LUT.
# When using the AccurateCoef class, it can be shown that the errors are...
#
# N                 Memory      Error
# ===================================================================
# 2**12 (4096)      128KiB      3.73e-08 (smaller then eps32)
# 2**13 (8192)      256KiB      9.31e-09
# 2**14 (16384)     512KiB      2.33e-09
# 2**15 (32768)     1MiB        5.82e-10 (much smaller then eps32)
#
NCOEFS = 32768


# Floor operator (deal with negative numbers)
cdef inline int floor(double a) nogil: return <int>a if a>=0.0 else (<int>a)-1
cdef inline int ceil(double a): return <int>(a+0.999999999999999) if a>=0.0 else (<int>a)
cdef inline int round(double a): return <int>(a+0.49999) if a>=0.0 else (<int>(a-0.49999))
cdef inline double dabs(double a): return a if a>=0 else -a
cdef inline int iabs(int a): return a if a>=0 else -a
cdef inline double dmax(double a, double b): return a if a>b else b
cdef inline double dmin(double a, double b): return a if a<b else b
cdef inline int imax(int a, int b): return a if a>b else b
cdef inline int imin(int a, int b): return a if a<b else b

cdef inline double wrongabs(double a): return <double>(<int>a)

## Cubic spline coefficients

def get_cubic_spline_coefs(t, spline_type=0.0):
    """ get_cubic_spline_coefs(t, spline_type='Catmull-Rom')
    
    Calculates the coefficients for a cubic spline and returns them as 
    a tuple. t is the ratio between "left" point and "right" point on the
    lattice. 
    
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
        
        'linear': Linear interpolation. This is not really cubic interpolation
        as the first and last coefficient are always 0. For testing purposes
        only!
    
    """
    
    # Init spline object and determine type
    cdef CoefLut tmp = CoefLut()
    cdef double splineId = tmp.spline_type_to_id(spline_type)
    
    # Obtain coefficients
    cdef double *out = <double *> malloc(4 * sizeof(double))
    tmp.cubicsplinecoef(splineId, t, out)
    
    # Pack results as tuple and free memory
    result = out[0], out[1], out[2], out[3]
    free(out)
    
    # Done
    return result


cdef class CoefLut:
    """ CoefLut()
    
    This class is a collection of methods to manage/create cubic spline
    coefficients. Further, it can be used to represent a look up table 
    (LUT) of cubic spline coefficients, so that the coefficients for
    arbitrary t-values can be obtained fast.
    
    This class is not ment to be exposed to Python directly. The function
    get_cubic_spline_coefs() can be used from Python to get the coefficients.
    It uses this class, so that the code to calculate the coefficients does
    not have to be repeated.
    
    Use the calculate_lut() to calculate the coefficients. Even better
    is to use the get_lut() classmethod, that stores all created tables,
    so they can be reused, thereby saving quite a bit of time.
    
    Example
    =======
    lut = CoefLut.get_lut('B-spline')
    lut.get_coef(0.3) # Uses the nearest value in the table
    
    # To get a (way) more accurate result (using linear interpolation
    # in the lut) at only a small performance penalty, use the AccurateCoef
    # class.
    coef = AccurateCoef(lut)
    coef.get_coef(0.3)
    
    """
    
    def __init__(self):
        pass
    
    def __cinit__(self):
        self._LUT = NULL
    
    def __dealloc__(self):
        if self._LUT is not NULL:
            free(self._LUT)
    
    
    cdef void calculate_lut(self, spline_type, int N):
        """ calculate_lut(spline_type, int N)
        
        Calculate the look-up table for the specified spline type
        with N entries.
        
        """
        
        # Clear any previous LUT
        if self._LUT is not NULL:
            free(self._LUT)
        
        # The actial length is 1 larger, so also t=1.0 is in the table
        cdef int N1 = N + 2 # and an extra bit for if we use linear interp.
        
        # Allocate array (first clear)
        self._LUT = <double *> malloc(N1 * 4 * sizeof(double))
        self.N = N
        
        # Prepare
        cdef double step = 1.0 / N
        cdef double t = 0.0
        cdef int i
        cdef double splineId = self.spline_type_to_id(spline_type)
        
        # For each possible t, calculate the coefficients
        # The exact coefficients depend on the type of spline
        for i in range(N1):
            self.cubicsplinecoef(splineId, t, (self._LUT + i*4))
            t += step
    
    
    cdef double spline_type_to_id(self, spline_type) except *:
        """ spline_type_to_id(spline_type)
        
        Method to map a spline name to an integer ID. This is used
        so that the LUT can be created relatively fast without having to
        repeat the loop for each spline type.
        
        spline_type can also be a number between -1 and 1, representing
        the tension for a Cardinal spline.
        
        """
        
        # Handle tension given for Cardinal spline
        cdef double tension = 0.0
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
        elif spline_type.lower() in ['lin', 'linear']:
            return 98.0        
        elif spline_type.lower() in ['quad', 'quadratic']:
            return 99.0
        else:
            raise ValueError('Unknown spline type: ' + str(spline_type))
    
    
    cdef cubicsplinecoef(self, double splineId, double t, double* out):
        """ cubicsplinecoef(double splineId, double t, double* out)
        
        Method that wraps the specific cubicsplinecoef_* methods, 
        by mapping the splineId.
        
        """
        if splineId == 0.0:
            self.cubicsplinecoef_catmullRom(t, out)
        elif splineId <= 1.0:
            self.cubicsplinecoef_cardinal(t, out, splineId) # tension=splineId
        elif splineId == 2.0:
            self.cubicsplinecoef_basis(t, out)
        elif splineId == 3.0:
            self.cubicsplinecoef_hermite(t, out)
        elif splineId == 4.0:
            self.cubicsplinecoef_lagrange(t, out)
        elif splineId == 5.0:
            self.cubicsplinecoef_lanczos(t, out)
        elif splineId == 98.0:
            self.cubicsplinecoef_linear(t, out)
        elif splineId == 99.0:
            self.cubicsplinecoef_quadratic(t, out)
    
    
    cdef cubicsplinecoef_catmullRom(self, double t, double *out):
        # See the doc for the catmull-rom spline, this is how the two splines
        # are combined by simply adding (and dividing by two) 
        out[0] = - 0.5*t**3 + t**2 - 0.5*t        
        out[1] =   1.5*t**3 - 2.5*t**2 + 1
        out[2] = - 1.5*t**3 + 2*t**2 + 0.5*t
        out[3] =   0.5*t**3 - 0.5*t**2
    
    cdef cubicsplinecoef_cardinal(self, double t, double *out, double tension):
        cdef double tau = 0.5 * (1 - tension)
        out[0] = - tau * (   t**3 - 2*t**2 + t )
        out[3] =   tau * (   t**3 -   t**2     )
        out[1] =           2*t**3 - 3*t**2 + 1  - out[3]
        out[2] = -         2*t**3 + 3*t**2      - out[0]
    
    cdef cubicsplinecoef_basis(self, double t, double *out):
        out[0] = (1-t)**3                     /6.0
        out[1] = ( 3*t**3 - 6*t**2 +       4) /6.0
        out[2] = (-3*t**3 + 3*t**2 + 3*t + 1) /6.0
        out[3] = (  t)**3                     /6.0
    
    cdef cubicsplinecoef_hermite(self, double t, double *out):
        out[0] =   2*t**3 - 3*t**2 + 1
        out[1] =     t**3 - 2*t**2 + t
        out[2] = - 2*t**3 + 3*t**2
        out[3] =     t**3 -   t**2
    
    cdef cubicsplinecoef_lagrange(self, double t, double *out):
        cdef double k
        k = -1.0  
        out[0] =               (t  )/(k  ) * (t-1)/(k-1) * (t-2)/(k-2)
        k= 0  
        out[1] = (t+1)/(k+1) *               (t-1)/(k-1) * (t-2)/(k-2)
        k= 1  
        out[2] = (t+1)/(k+1) * (t  )/(k  ) *               (t-2)/(k-2)
        k= 2  
        out[3] = (t+1)/(k+1) * (t  )/(k  ) * (t-1)/(k-1)
    
    cdef cubicsplinecoef_lanczos(self, double t, double *out):
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
    
    cdef cubicsplinecoef_linear(self, double t, double *out):
        out[0] = 0.0
        out[1] = (1.0-t)
        out[2] = t
        out[3] = 0.0
    
    cdef cubicsplinecoef_quadratic(self, double t, double *out):
        # This corresponds to adding the two quadratic polynoms,
        # thus keeping genuine quadratic interpolation. However,
        # it has the same support as a cubic spline, so why use this?
        out[0] = 0.25*t**2 - 0.25*t
        out[1] = -0.25*t**2 - 0.75*t + 1
        out[2] = -0.25*t**2 + 1.25*t
        out[3] = 0.25*t**2 - 0.25*t
    
#     cdef cubicsplinecoef_cardinal_edge(self, double t, double *out): 
#         # This is to handle the edges for a Cardinal spline.
#         # It uses quadratic interpolation. Based on the value of t, which
#         # goes below 0.0 or above 1.0, left or right quadratic
#         # polynomals are returned.
#         if t < 0.0:
#             t += 1.0
#             out[0] = 0.5*t**2 - 1.5*t + 1
#             out[1] = -t**2 + 2*t
#             out[2] = 0.5*t**2 - 0.5*t
#             out[3] = 0.0
#         elif t > 1.0:
#             t -= 1.0
#             out[0] = 0.0
#             out[1] = 0.5*t**2 - 0.5*t
#             out[2] = -t**2 + 1
#             out[3] = 0.5*t**2 + 0.5*t
#         else:
#             out[0] = 0.0
#             out[1] = 0.0
#             out[2] = 0.0
#             out[3] = 0.0
#         
#     cdef quadraticsplinecoef_left(self, double t, double *out): 
#         # A quadratic spline is well defined: there's just this one
#         # equation (well, two actually).
#         # A quadratic spline uses three points x0 x1 x2. So either your are 
#         # interpolating between x0 and x1 (left), or between x1 and x2 (right).
#         # You can also extraporlate actually.
#         out[0] = 0.5*t**2 - 1.5*t + 1
#         out[1] = -t**2 + 2*t
#         out[2] = 0.5*t**2 - 0.5*t
#     
#     cdef quadraticsplinecoef_right(self, double t, double *out): 
#         out[0] = 0.5*t**2 - 0.5*t
#         out[1] = -t**2 + 1
#         out[2] = 0.5*t**2 + 0.5*t
    
    
    cdef double* get_coef(self, double t) nogil:
        """ get_coef(t)
        
        Get the coefficients for given value of t. This simply obtains
        the nearest coefficients in the table. For a more accurate result,
        use the AccurateCoef class.
        
        """
        cdef int i = <int>(t * self.N + 0.5) # Round
        return (self._LUT + 4*i)
    
    
    cdef double* get_coef_from_index(self, int i):
        """ get_coef_from_index(i)
        
        Get the spline coefficients using the index in the table.
        
        """
        return (self._LUT + 4*i)
    
    
    @classmethod
    def get_lut(cls, spline_type, N=32768):
        """ get_lut(spline_type, N=32768)
        
        Classmethod to get a lut of the given spline type and with
        the given amount of elements (default 2**15).
        
        This method uses a global buffer to store previously 
        calculated LUT's; if the requested LUT was created 
        earlier, it does not have to be re-calculated.
        
        """
        
        # Get list instance
        D = _global_coef_lut_dict
        
        # Get id
        cdef CoefLut lut = CoefLut()
        key = lut.spline_type_to_id(spline_type), N
        
        # Create lut if not existing yet
        if key not in D.keys():
            lut.calculate_lut(spline_type, NCOEFS)
            D[key] = lut
        
        # Return lut
        return D[key] 


cdef class AccurateCoef:
    """ AccurateCoef(lut)
    
    A simple class that can be used to obtain accurate values from 
    a CoefLut instance, by using linear interpolation.
    
    This is implemented in a separate class because a piece of memory
    needs to be managed to contain the four interpolated coefficients.
    
    """
    
    def __init__(self, CoefLut lut):
        self._lut = lut
    
    def __cinit__(self):
        self._out = <double *> malloc(4 * sizeof(double))
    
    def __dealloc__(self):
        if self._out is not NULL:
            free(self._out)
            self._out = NULL
    
    
    cdef double* get_coef(self, double t) nogil:
        """ get_coef(t)
        
        Get the coefficients for given value of t, using linear 
        interpolation.
        
        """
        
        # Get i1, i2 and t1, t2
        t = t * self._lut.N
        cdef int i1 = <int>t
        cdef int i2 = i1+1
        cdef double t2 = t-i1
        cdef double t1 = (1.0-t2)
        
        # Correct indices
        i1 *= 4
        i2 *= 4
        
        # Fill values
        cdef int i
        cdef double* LUT = self._lut._LUT  
        for i in range(4):
            self._out[i] = LUT[i1+i] * t1 + LUT[i2+i] * t2
        
        # Return
        return self._out


# Global dict with LUT instances. See CoefLut.get_lut().
_global_coef_lut_dict = {}



## Misc


def meshgrid(*args):
    """ Meshgrid implementation for 2D and 3D. 
    
    meshgrid(nx,ny) will create meshgrids with the specified shapes.
    
    meshgrid(ndarray) will create meshgrids corresponding to the shape
    of the given array (which must have 2 or 3 dimension).
    
    meshgrid([2,3,4], [1,2,3], [4,1,2]) uses the supplied values to 
    create the grids. These lists can also be numpy arrays.
    
    Returns a tuple with the grids in x-y-z order, with arrays of type float32.
    """
    
    # Test args
    if len(args) == 1:
        args = args[0]
    if isinstance(args, np.ndarray) and args.ndim in [2,3]:
        args = tuple(reversed(args.shape))
    if not isinstance(args, (list, tuple)):
        raise ValueError('Invalid argument for meshgrid.')
    iterators = []
    for arg in args:
        if isinstance(arg, int):
            iterators.append( np.arange(arg, dtype=SAMPLE) )
        elif isinstance(arg, list):
            iterators.append( np.array(arg, dtype=SAMPLE) )
        elif isinstance(arg, np.ndarray):
            iterators.append( arg )
        else:
            raise ValueError('Invalid argument for meshgrid.')
    
    # Reverse iterators so its in z-y-x order
    iterators.reverse()
    
    # Determine shape
    shape = tuple([len(tmp) for tmp in iterators])
    
    # Init variables
    cdef int d, i
    cdef np.ndarray[SAMPLE_T, ndim=1] iterator
    
    # Init grids
    grids = []
    
    for d in range(len(shape)):
        
        # Init array
        grid = np.empty(shape, dtype=SAMPLE)
        
        # Get iterator
        tmp = iterators[d].ravel()
        if tmp.dtype != SAMPLE:
            tmp = tmp.astype(SAMPLE)
        iterator = tmp
        
        # Walk along that dimension
        if d == 0:
            for i in range(shape[d]):
                grid[i] = iterator[i]
        if d == 1:
            for i in range(shape[d]):
                grid[:,i] = iterator[i]
        if d == 2:
            for i in range(shape[d]):
                grid[:,:,i] = iterator[i]
        
        # Store
        grids.append(grid)
    
    # Done
    return tuple(reversed(grids))
    

@cython.boundscheck(False)
@cython.wraparound(False)
def slice_from_volume(data, pos, vec1, vec2, Npatch, order=3):
    """ slice_from_volume(data, pos, vec1, vec2, Npatch, order=3)
    Samples a 2D slice from a 3D volume, using a center position
    and two vectors that span the patch. The length of the vectors
    specify the sample distance for the patch.
    """
    
    # Init sample arrays
    samplesx = np.empty((Npatch, Npatch), dtype=SAMPLE)
    samplesy = np.empty((Npatch, Npatch), dtype=SAMPLE)
    samplesz = np.empty((Npatch, Npatch), dtype=SAMPLE)
    #
    cdef np.ndarray[SAMPLE_T, ndim=2] samplesx_ = samplesx
    cdef np.ndarray[SAMPLE_T, ndim=2] samplesy_ = samplesy
    cdef np.ndarray[SAMPLE_T, ndim=2] samplesz_ = samplesz
    
    # Init iterator variables and offset
    cdef int u, v
    cdef double ud, vd
    cdef double Npatch2 = Npatch/2.0
    
    # Set start position
    cdef double x = pos.x
    cdef double y = pos.y
    cdef double z = pos.z
    
    # Get anisotropy factors
    cdef double sam_x = 1.0
    cdef double sam_y = 1.0
    cdef double sam_z = 1.0
    cdef double ori_x = 0.0
    cdef double ori_y = 0.0
    cdef double ori_z = 0.0
    if hasattr(data, 'sampling'):
        sam_x = 1.0 / data.sampling[2]  # Do the division here
        sam_y = 1.0 / data.sampling[1]
        sam_z = 1.0 / data.sampling[0]
        ori_x = data.origin[2]
        ori_y = data.origin[1]
        ori_z = data.origin[0]
    
    # Make vectors quick
    cdef double v1x = vec1.x
    cdef double v1y = vec1.y
    cdef double v1z = vec1.z
    #
    cdef double v2x = vec2.x
    cdef double v2y = vec2.y
    cdef double v2z = vec2.z
    
    # Loop
    for v in range(Npatch):
        vd = <double>(v-Npatch2)
        for u in range(Npatch):
            ud = <double>(u-Npatch2)
            
            # Determine sample positions
            samplesx_[v,u] = ( (x + vd*v1x + ud*v2x) - ori_x) * sam_x
            samplesy_[v,u] = ( (y + vd*v1y + ud*v2y) - ori_y) * sam_y
            samplesz_[v,u] = ( (z + vd*v1z + ud*v2z) - ori_z) * sam_z
    
    # Almost done
    return interp(data, [samplesx, samplesy, samplesz], order)


cdef double uglyRoot(double n):
    """ uglyRoot(double n)
    Calculates an approximation of the square root using
    (a few) Newton iterations.
    """
    cdef double x
    cdef int iter
    
    x = 1.0    
    for iter in range(3):
        x = x - (x*x - n) / (2.0 * x)
    return x


## For deformations


def make_samples_absolute(samples):
    """ make_samples_absolute(samples)
    
    Note: this function is intended for sampes that represent a 
    deformation; the number of dimensions of each array should 
    match the number of arrays.
    
    Given a tuple of arrays that represent a relative deformation 
    expressed in world coordinates (x,y,z order), returns a tuple
    of sample arrays that represents the absolute sample locations in pixel
    coordinates. It is assumed that the sampling of the data is the same
    as for the sample arrays. The origin property is not used.
    
    This process can also be done with relative ease by adding a meshgrid
    and then using ainterp() or aproject(). But by combining it in 
    one step, this process becomes faster and saves memory. Note that
    the deform_*() functions use this function.
    
    """
    
    ndim = len(samples)
    absolute_samples = []
    for i in range(ndim):
        
        # Get array and check
        sample_array = samples[i]
        if sample_array.ndim != ndim:
            raise ValueError("make_samples_absolute: the number of dimensions"+
                " of each array should  match the number of arrays.")
        
        # Get dimension corresponding to this sampling array
        d = ndim-i-1
        
        # Get sampling
        sampling = 1.0
        if hasattr(sample_array, 'sampling'):
            sampling = sample_array.sampling[d]
        
        # Instantiate result
        result = np.empty_like(sample_array)
        absolute_samples.append(result)
        
        # Apply
        if sample_array.ndim == 1:
            make_samples_absolute1(sample_array, result, sampling, d)
        if sample_array.ndim == 2:
            make_samples_absolute2(sample_array, result, sampling, d)
        if sample_array.ndim == 3:
            make_samples_absolute3(sample_array, result, sampling, d)
    
    # Done
    return absolute_samples

@cython.boundscheck(False)
@cython.wraparound(False)
def make_samples_absolute1(samples, result, sampling, dim=0):
    
    # Cast arrays
    cdef np.ndarray[SAMPLE_T, ndim=1] samples_ = samples
    cdef np.ndarray[SAMPLE_T, ndim=1] result_ = result
    
    # Define variables
    cdef int x, y, z
    cdef double sampling_i = 1.0/sampling
    cdef int Nx = samples.shape[0]
    
    if dim == 0:
        for x in range(Nx):
            result_[x] = (<double>x) +  samples_[x] * sampling_i

@cython.boundscheck(False)
@cython.wraparound(False)
def make_samples_absolute2(samples, result, sampling, dim=0):
    
    # Cast arrays
    cdef np.ndarray[SAMPLE_T, ndim=2] samples_ = samples
    cdef np.ndarray[SAMPLE_T, ndim=2] result_ = result
    
    # Define variables
    cdef int x, y, z
    cdef double sampling_i = 1.0/sampling
    cdef int Ny = samples.shape[0]
    cdef int Nx = samples.shape[1]
    
    if dim == 0:
        for y in range(Ny):
            for x in range(Nx):
                result_[y,x] = (<double>y) +  samples_[y,x] * sampling_i
    elif dim == 1:
        for y in range(Ny):
            for x in range(Nx):
                result_[y,x] = (<double>x) +  samples_[y,x] * sampling_i

@cython.boundscheck(False)
@cython.wraparound(False)
def make_samples_absolute3(samples, result, sampling, dim=0):
    
    # Cast arrays
    cdef np.ndarray[SAMPLE_T, ndim=3] samples_ = samples
    cdef np.ndarray[SAMPLE_T, ndim=3] result_ = result
    
    # Define variables
    cdef int x, y, z
    cdef double sampling_i = 1.0/sampling
    cdef int Nz = samples.shape[0]
    cdef int Ny = samples.shape[1]
    cdef int Nx = samples.shape[2]
    
    if dim == 0:
        for z in range(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    result_[z,y,x] = (<double>z) +  samples_[z,y,x] * sampling_i
    elif dim == 1:
        for z in range(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    result_[z,y,x] = (<double>y) +  samples_[z,y,x] * sampling_i
    elif dim == 2:
        for z in range(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    result_[z,y,x] = (<double>x) +  samples_[z,y,x] * sampling_i



# todo: this is deprecated
def fix_samples_edges(samples):
    """ fix_samples_edges(samples)
    
    Limit the values of the sample arrays to they do not go beyond edges.
    
    
    """
    
    ndim = len(samples)
    for d in range(ndim):
        
        # Get array and check
        sample_array = samples[d]
        if sample_array.ndim != ndim:
            raise ValueError("fix_samples_edges: the number of dimensions"+
                " of each array should  match the number of arrays.")
        
        # Get sampling
        sampling = 1.0
        if hasattr(sample_array, 'sampling'):
            sampling = sample_array.sampling[d]
        
        # Apply
        if sample_array.ndim == 1:
            fix_samples_edges1(sample_array, sampling, d)
        if sample_array.ndim == 2:
            fix_samples_edges2(sample_array, sampling, d)
        if sample_array.ndim == 3:
            fix_samples_edges3(sample_array, sampling, d)

@cython.boundscheck(False)
@cython.wraparound(False)
def fix_samples_edges1(samples, sampling, dim=0):
    
    # Cast arrays
    cdef np.ndarray[SAMPLE_T, ndim=1] samples_ = samples
    
    # Define variables
    cdef int x, y, z
    cdef double sampling_ = sampling
    cdef double limit
    cdef int Nx = samples.shape[0]
    
    if dim == 0:
        for x in range(Nx):
            limit = 0.9 * dmin(x, Nx-x-1) * sampling_
            samples_[x] = dmin(samples_[x], limit)

@cython.boundscheck(False)
@cython.wraparound(False)
def fix_samples_edges2(samples, sampling, dim=0):
    
    # Cast arrays
    cdef np.ndarray[SAMPLE_T, ndim=2] samples_ = samples
    
    # Define variables
    cdef int x, y, z
    cdef double sampling_ = sampling
    cdef double limit
    cdef int Ny = samples.shape[0]
    cdef int Nx = samples.shape[1]
    
    if dim == 0:
        for y in range(Ny):
            for x in range(Nx):
                limit = 0.9 * dmin(y, Ny-y-1) * sampling_
                samples_[y,x] = dmin(samples_[y,x], limit)
    elif dim == 1:
        for y in range(Ny):
            for x in range(Nx):
                limit = 0.9 * dmin(x, Nx-x-1) * sampling_
                samples_[y,x] = dmin(samples_[y,x], limit)

@cython.boundscheck(False)
@cython.wraparound(False)
def fix_samples_edges3(samples, sampling, dim=0):
    
    # Cast arrays
    cdef np.ndarray[SAMPLE_T, ndim=3] samples_ = samples
    
    # Define variables
    cdef int x, y, z
    cdef double sampling_ = sampling
    cdef double limit
    cdef int Nz = samples.shape[0]
    cdef int Ny = samples.shape[1]
    cdef int Nx = samples.shape[2]
    
    if dim == 0:
        for z in range(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    limit = 0.9 * dmin(z, Nz-z-1) * sampling_
                    samples_[z,y,x] = dmin(samples_[z,y,x], limit)
    elif dim == 1:
        for z in range(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    limit = 0.9 * dmin(y, Ny-y-1) * sampling_
                    samples_[z,y,x] = dmin(samples_[z,y,x], limit)
    elif dim == 2:
        for z in range(Nz):
            for y in range(Ny):
                for x in range(Nx):
                    limit = 0.9 * dmin(x, Nx-x-1) * sampling_
                    samples_[z,y,x] = dmin(samples_[z,y,x], limit)


## Entry functions for intep and project


# Needs to be defined in the pyx file because it is used by other functions
def interp(data, samples, order=1, spline_type=0.0):
    """ interp(data, samples, order='linear', spline_type=0.0)
    
    Interpolate (sample) data at the positions specified by samples 
    (pixel coordinates).
    
    Parameters
    ----------
    data : array (float32 or float64)
        Data to interpolate, can be 1D, 2D or 3D.
    sample : tuple with numpy arrays
        Each array specifies the sample position for one dimension (in 
        x-y-z order). 
    order : integer or string
        Order of interpolation. Can be 0:'nearest', 1:'linear', 
        2:'quasi-linear', 3:'cubic'. 
    spline_type : float or string
        Only for cubic interpolation. Specifies the type of spline.
        Can be 'Basis', 'Hermite', 'Cardinal', 'Catmull-rom', 'Lagrange', 
        'Lanczos', 'quadratic', or a float, specifying the tension 
        parameter for the Cardinal spline. See the docs of 
        get_cubic_spline_coefs() for more information.
    
    Returns
    -------
    result : array 
        The result is of the same shape of the samples arrays, which 
        can be of any shape. This flexibility makes this function suitable
        as a base function used by many of the other high-level functions
        defined in this module.
    
    Notes
    -----------
    The input data should be of float32 or float64 and can have up to 
    three dimensions.
    
    An order of interpolation of 2 would naturally correspond to
    quadratic interpolation. However, due to its uneven coefficients
    it reques the same support (and speed) as a cubic interpolant, 
    while producing less satisfactory results in general. 
    
    Quasi-linear corresponds to a cardinal spine with tension 1, which is 
    calculated using cubic polynomials, but has the support of linear 
    interpolation (in other words: it uses only two coefficients). 
    It produces results wich are C2 continouos and look smoother than linear
    interpolation, at only a slight decrease of speed.
    
    It can be shown (see Thevenaz et al. 2000 "Interpolation Revisited") 
    that interpolation using a Cardinal spline is equivalent to 
    interpolating B-spline interpolation.
    
    """
    
    # Check data
    if not isinstance(data, np.ndarray):
        raise ValueError('data must be a numpy array.')
    elif data.ndim > 3:
        raise ValueError('can not interpolate data with such many dimensions.')
    elif data.dtype not in [np.float32, np.float64, np.int16]:
        raise ValueError('data must be float32 or float64.')
    
    # Check samples
    if isinstance(samples, tuple):
        pass
    elif isinstance(samples, list):
        samples = tuple(samples)
    elif data.ndim==1:
        samples = (samples,)
    else:
        raise ValueError("samples must be a tuple of arrays.")
    if len(samples) != data.ndim:
        tmp = "samples must contain as many arrays as data has dimensions."
        raise ValueError(tmp)
    for s in samples:
        if not isinstance(data, np.ndarray):
            raise ValueError("values in samples must all be numpy arrays.")
        if s.shape != samples[0].shape:
            raise ValueError("sample arrays must all have the same shape.")
        if s.dtype != np.float32:
            raise ValueError("sample arrays must be of type float32.")
    
    # Check order
    orders = {'nearest':0, 'linear':1, 'quasi-linear':2, 'cubic':3}
    if isinstance(order, basestring):
        try:
            order = orders[order]
        except KeyError:
            raise ValueError('Unknown order of interpolation.')
    if order not in [0, 1, 2, 3]:
        raise ValueError('Invalid order of interpolation.')
    
    # Go
    result = None
    if data.dtype == np.float32:
        if data.ndim == 1:
            result = interp1_32(data, samples[0], order, spline_type)
        elif data.ndim == 2:
            result = interp2_32(data, samples[0], samples[1], order, spline_type)
        elif data.ndim == 3:
            tmp = samples
            result = interp3_32(data, tmp[0], tmp[1], tmp[2], order, spline_type)
    
    elif data.dtype == np.float64:
        if data.ndim == 1:
            result = interp1_64(data, samples[0], order, spline_type)
        elif data.ndim == 2:
            result = interp2_64(data, samples[0], samples[1], order, spline_type)
        elif data.ndim == 3:
            tmp = samples
            result = interp3_64(data, tmp[0], tmp[1], tmp[2], order, spline_type)
    
#     elif data.dtype == np.int16:
#         if data.ndim == 1:
#             result = interp1_i16(data, samples[0], order, spline_type)
#         elif data.ndim == 2:
#             result = interp2_i16(data, samples[0], samples[1], order, spline_type)
#         elif data.ndim == 3:
#             tmp = samples
#             result = interp3_i16(data, tmp[0], tmp[1], tmp[2], order, spline_type)
    
    # Make Anisotropic array if input data was too
    # --> No: We do not know what the sample points are
    
    # Done
    return result


def ainterp(data, samples, *args, **kwargs):
    """ ainterp(data, samples, order='linear', spline_type=0.0)
    
    Interpolation in anisotropic array. Like interp(), but the
    samples are expressed in world coordimates.    
    
    """
    
    # Check
    if not (hasattr(data, 'sampling') and hasattr(data, 'origin')):
        raise ValueError('ainterp() needs the data to be an Aarray.')
    
    # Correct samples
    samples2 = []
    ndim = len(samples)
    for i in range(ndim):
        d = ndim-i-1
        origin = data.origin[d]
        sampling = data.sampling[d]
        samples2.append( (samples[i]-origin) / sampling )
    
    # Interpolate
    return interp(data, samples2, *args, **kwargs)


def project(data, deltas):
    """ project(data, deltas)
    
    Interpolate data according to the deformations specified in deltas.
    Deltas should be a tuple of numpy arrays similar to 'samples' in
    the interp() function. They represent the relative sample positions.
    
    Applies forward deformations, moving the pixels/voxels to the given
    locations, rather than getting the pixel values from the given 
    locations.
    
    """
    
    # Check
    if len(deltas) != data.ndim:
        tmp = "Samples must contain as many arrays as data has dimensions."
        raise ValueError(tmp)
    
    if data.dtype == np.float32:
        if data.ndim == 1:
            return project1_32(data, *deltas)
        elif data.ndim == 2:
            return project2_32(data, *deltas)
        elif data.ndim == 3:
            return project3_32(data, *deltas)
        else:
            raise RuntimeError('deform_forward not implemented for that dimension.')
    
    elif data.dtype == np.float64:
        if data.ndim == 1:
            return project1_64(data, *deltas)
        elif data.ndim == 2:
            return project2_64(data, *deltas)
        elif data.ndim == 3:
            return project3_64(data, *deltas)
        else:
            raise RuntimeError('deform_forward not implemented for that dimension.')
    else:
        raise RuntimeError('deform_forward not implemented for that data type.')



def aproject(data, samples, *args, **kwargs):
    """ ainterp(data, samples, order='linear', spline_type=0.0)
    
    Interpolation in anisotropic array. Like interp(), but the
    samples are expressed in world coordimates.    
    
    """
    
    # Check
    if not (hasattr(data, 'sampling') and hasattr(data, 'origin')):
        raise ValueError('ainterp() needs the data to be an Aarray.')
    
    # Correct samples
    samples2 = []
    ndim = len(samples)
    for i in range(ndim):
        d = ndim-i-1
        origin = data.origin[d]
        sampling = data.sampling[d]
        samples2.append( (samples[i]-origin) / sampling )
    
    # Interpolate
    return project(data, samples2, *args, **kwargs)


## Float32 backward and forward mapping functions

@cython.boundscheck(False)
@cython.wraparound(False)
def interp1_32(data, samplesx, order, spline_type=0.0):
    """ Interpolation of 1D base data of 32bit floats
    """
    
    # Create result array with the shape of the samples
    result = np.empty((samplesx.size,), dtype=FLOAT32)
    
    # Typecast (flatten samples)
    cdef np.ndarray[FLOAT32_T, ndim=1] data_ = data
    cdef np.ndarray[FLOAT32_T, ndim=1] result_ = result
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesx_ = samplesx.ravel()
    
    # Create and init cubic interpolator
    cdef CoefLut lut
    cdef AccurateCoef coeffx
    cdef double *ccx
    cdef double splineId = 0.0
    if order > 1:
        lut = CoefLut.get_lut(spline_type)
        splineId = lut.spline_type_to_id(spline_type)
        coeffx = AccurateCoef(lut)
    
    # Prepare sample location variables
    cdef double dx, tx, tx_
    cdef int ix
    cdef int cx, cx1, cx2
    
    # Prepare indices and bounds, etc
    cdef int i
    cdef int Ni = samplesx.shape[0]
    cdef int Nx = data.shape[0]
    cdef double val
    
    
    if order == 3:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            
            if ix >= 1 and ix < Nx-2:
                # Cubic interpolation
                ccx = coeffx.get_coef(tx)
                val =  data_[ix-1] * ccx[0]
                val += data_[ix  ] * ccx[1]
                val += data_[ix+1] * ccx[2]
                val += data_[ix+2] * ccx[3]
                result_[i] = val
            
            elif dx>=-0.5 and dx<=Nx-0.5:
                # Edge effects
                
                # Get coefficients
                ccx = coeffx.get_coef(tx)
                
                # Correct stuff: calculate offset (max 2)
                cx1, cx2 = 0, 4
                #
                if ix<1: cx1+=1-ix;
                if ix>Nx-3: cx2+=(Nx-3)-ix;
                
                # Correct coefficients, so that the sum is one
                val = 0.0
                for cx in range(cx1, cx2):  val += ccx[cx]
                val = 1.0/val
                for cx in range(cx1, cx2):  ccx[cx] *= val
                
                # Combine elements
                val = 0.0
                for cx in range(cx1, cx2):
                    val += data_[ix+cx-1] * ccx[cx]
                result_[i] = val
                
                # Code below produces nice results too, but can only be 
                # used for 1D; higher dimensions get weird results
                
#                 # Correct indices
#                 if ix<1: tx+=ix-1; ix=1; 
#                 if ix>Nx-3: tx+=ix-(Nx-3); ix=Nx-3;
#                 
#                 # Get coefficients (first fill, so we can use its memory)
#                 ccx = coeffx.get_coef(0.0)
#                 if splineId < 2.0:
#                     lut.cubicsplinecoef_cardinal_edge(tx, ccx)
#                 else:
#                     lut.cubicsplinecoef(splineId, tx, ccx)
#                 
#                 # Combine elements
#                 val = 0.0
#                 for cx in range(0,4):
#                     cx2 = ix + cx - 1 
#                     val += data_[cx2] * ccx[cx]
#                 result_[i] = val
            
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 2:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            
            if ix >= 0 and ix < Nx-1:
                # Quasi-linear interpolation
                tx_ = -2*tx**3 + 3*tx**2
                val =  data_[ix] * (1.0-tx_)
                val += data_[ix+1] * tx_
                result_[i] = val
            elif dx>=-0.5 and dx<=Nx-0.5:                
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                # Quasi-linear interpolation (edges)
                tx_ = -2*tx**3 + 3*tx**2
                val =  data_[ix] * (1.0-tx_)
                val += data_[ix+1] * tx_
                result_[i] = val
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 1:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            
            if ix >= 0 and ix < Nx-1:
                # Linear interpolation
                val =  data_[ix] * (1.0-tx)
                val += data_[ix+1] * tx
                result_[i] = val
            elif dx>=-0.5 and dx<=Nx-0.5:
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                # Linear interpolation (edges)
                val =  data_[ix] * (1.0-tx)
                val += data_[ix+1] * tx
                result_[i] = val
            else:
                # Out of range
                result_[i] = 0.0
    
    else:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location
            dx = samplesx_[i]; ix = floor(dx+0.5)
            
            if ix >= 0 and ix < Nx:
                # Nearest neighbour interpolation
                result_[i] = data_[ix]
            else:
                # Out of range
                result_[i] = 0.0
    
    # Done
    result.shape = samplesx.shape
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def interp2_32(data, samplesx, samplesy, order, spline_type=0.0):
    """ Interpolation of 2D base data of 32bit floats
    """    
    
    # Create result array with the shape of the samples
    result = np.empty((samplesx.size,), dtype=FLOAT32)
    
    # Flatten and type the samples and result
    cdef np.ndarray[FLOAT32_T, ndim=2] data_ = data
    cdef np.ndarray[FLOAT32_T, ndim=1] result_ = result
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesx_ = samplesx.ravel()
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesy_ = samplesy.ravel()
    
    # Create and init cubic interpolator
    cdef CoefLut lut
    cdef AccurateCoef coeffx, coeffy
    cdef double splineId = 0.0
    cdef double *ccx
    cdef double *ccy
    if order > 1:
        lut = CoefLut.get_lut(spline_type)
        splineId = lut.spline_type_to_id(spline_type)
        coeffx = AccurateCoef(lut)
        coeffy = AccurateCoef(lut)
    
    # Prepare sample location variables
    cdef double dx, tx, tx_, dy, ty, ty_
    cdef int ix, iy
    cdef int cx, cy
    cdef int cx1, cx2, cy1, cy2
    cdef double valFactor
    
    # Prepare indices and bounds, etc
    cdef int i
    cdef int Ni = samplesx.size
    cdef int Ny = data.shape[0]
    cdef int Nx = data.shape[1]
    cdef double val
    
    
    if order == 3:
       
        # with nogil: does not make it faster
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            
            if (    ix >= 1 and ix < Nx-2 and 
                    iy >= 1 and iy < Ny-2       ):
                # Cubic interpolation
                ccx = coeffx.get_coef(tx)
                ccy = coeffy.get_coef(ty)
                val = 0.0
                for cy in range(4):
                    for cx in range(4):
                        val += data_[iy+cy-1,ix+cx-1] * ccy[cy] * ccx[cx]
                result_[i] = val
            
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5     ):
                # Edge effects
                
                # Get coefficients
                ccx = coeffx.get_coef(tx)
                ccy = coeffy.get_coef(ty)
                
                # Correct stuff: calculate offset (max 2)
                cx1, cx2 = 0, 4
                cy1, cy2 = 0, 4
                #
                if ix<1: cx1+=1-ix;
                if ix>Nx-3: cx2+=(Nx-3)-ix;
                #
                if iy<1: cy1+=1-iy;
                if iy>Ny-3: cy2+=(Ny-3)-iy;
                
                # Correct coefficients, so that the sum is one
                val = 0.0
                for cx in range(cx1, cx2):  val += ccx[cx]
                val = 1.0/val
                for cx in range(cx1, cx2):  ccx[cx] *= val
                #
                val = 0.0
                for cy in range(cy1, cy2):  val += ccy[cy]
                val = 1.0/val
                for cy in range(cy1, cy2):  ccy[cy] *= val
                
                # Combine elements
                # No need to pre-calculate indices: the C compiler is well
                # capable of making these optimizations.
                val = 0.0
                for cy in range(cy1, cy2):
                    for cx in range(cx1, cx2):
                        val += data_[iy+cy-1,ix+cx-1] * ccy[cy] * ccx[cx]
                result_[i] = val
            
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 2:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            
            if (    ix >= 0 and ix < Nx-1 and
                    iy >= 0 and iy < Ny-1     ):
                # Quasi-linear interpolation
                tx_ = -2*tx**3 + 3*tx**2
                ty_ = -2*ty**3 + 3*ty**2
                val =  data_[iy,  ix  ] * (1.0-ty_) * (1.0-tx_)
                val += data_[iy,  ix+1] * (1.0-ty_) *      tx_
                val += data_[iy+1,ix  ] *      ty_  * (1.0-tx_)
                val += data_[iy+1,ix+1] *      ty_  *      tx_
                result_[i] = val
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5     ):
                # Edge effects
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                #
                if iy<0: ty+=iy; iy=0; 
                if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
                # Quasi-linear interpolation (edges)
                tx_ = -2*tx**3 + 3*tx**2
                ty_ = -2*ty**3 + 3*ty**2
                val =  data_[iy,  ix  ] * (1.0-ty_) * (1.0-tx_)
                val += data_[iy,  ix+1] * (1.0-ty_) *      tx_
                val += data_[iy+1,ix  ] *      ty_  * (1.0-tx_)
                val += data_[iy+1,ix+1] *      ty_  *      tx_
                result_[i] = val
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 1:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            
            if (    ix >= 0 and ix < Nx-1 and
                    iy >= 0 and iy < Ny-1     ):
                # Linear interpolation
                val =  data_[iy,  ix  ] * (1.0-ty) * (1.0-tx)
                val += data_[iy,  ix+1] * (1.0-ty) *      tx
                val += data_[iy+1,ix  ] *      ty  * (1.0-tx)
                val += data_[iy+1,ix+1] *      ty  *      tx
                result_[i] = val
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5     ):
                # Edge effects
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                #
                if iy<0: ty+=iy; iy=0; 
                if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
                # Linear interpolation (edges)
                val =  data_[iy,  ix  ] * (1.0-ty) * (1.0-tx)
                val += data_[iy,  ix+1] * (1.0-ty) *      tx
                val += data_[iy+1,ix  ] *      ty  * (1.0-tx)
                val += data_[iy+1,ix+1] *      ty  *      tx
                result_[i] = val
            else:
                # Out of range
                result_[i] = 0.0
    
    else:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location
            dx = samplesx_[i]; ix = floor(dx+0.5)
            dy = samplesy_[i]; iy = floor(dy+0.5)
            
            if (    ix >= 0 and ix < Nx and
                    iy >= 0 and iy < Ny     ):
                # Nearest neighbour interpolation
                result_[i] = data_[iy,ix]
            else:
                # Out of range
                result_[i] = 0.0
    
    # Done
    result.shape = samplesx.shape
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def interp3_32(data, samplesx, samplesy, samplesz, order, spline_type=0.0):
    """ Interpolation of 3D base data of 32bit floats
    """    
    
    # Create result array with the shape of the samples
    result = np.empty((samplesx.size,), dtype=FLOAT32)
    
    # Flatten and type the samples and result
    cdef np.ndarray[FLOAT32_T, ndim=3] data_ = data
    cdef np.ndarray[FLOAT32_T, ndim=1] result_ = result
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesx_ = samplesx.ravel()
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesy_ = samplesy.ravel()
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesz_ = samplesz.ravel()
    
    # Create and init cubic interpolator
    cdef CoefLut lut
    cdef AccurateCoef coeffx, coeffy, coeffz
    cdef double splineId = 0.0
    cdef double *ccx
    cdef double *ccy
    cdef double *ccz
    if order > 1:
        lut = CoefLut.get_lut(spline_type)
        splineId = lut.spline_type_to_id(spline_type)
        coeffx = AccurateCoef(lut)
        coeffy = AccurateCoef(lut)
        coeffz = AccurateCoef(lut)
    
    # Prepare sample location variables
    cdef double dx, tx, tx_, dy, ty, ty_, dz, tz, tz_
    cdef int ix, iy, iz
    cdef int cx, cy, cz
    cdef int cx1, cx2, cy1, cy2, cz1, cz2
    cdef double valFactor
    
    # Prepare indices and bounds, etc
    cdef int i
    cdef int Ni = samplesx.size
    cdef int Nz = data.shape[0]
    cdef int Ny = data.shape[1]
    cdef int Nx = data.shape[2]
    cdef double val
    
    
    if order == 3:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            dz = samplesz_[i]; iz = floor(dz); tz = dz-iz
            
            if (    ix >= 1 and ix < Nx-2 and 
                    iy >= 1 and iy < Ny-2 and
                    iz >= 1 and iz < Nz-2       ):
                # Cubic interpolation
                ccx = coeffx.get_coef(tx)
                ccy = coeffy.get_coef(ty)
                ccz = coeffz.get_coef(tz)
                val = 0.0
                for cz in range(4):
                    for cy in range(4):
                        for cx in range(4):
                            val += data_[iz+cz-1,iy+cy-1,ix+cx-1] * (
                                            ccz[cz] * ccy[cy] * ccx[cx] )
                result_[i] = val
            
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5 and
                    dz>=-0.5 and dz<=Nz-0.5     ):
                # Edge effects
                
                # Get coefficients
                ccx = coeffx.get_coef(tx)
                ccy = coeffy.get_coef(ty)
                ccz = coeffz.get_coef(tz)
                
                # Correct stuff: calculate offset (max 2)
                cx1, cx2 = 0, 4
                cy1, cy2 = 0, 4
                cz1, cz2 = 0, 4
                #
                if ix<1: cx1+=1-ix;
                if ix>Nx-3: cx2+=(Nx-3)-ix;
                #
                if iy<1: cy1+=1-iy;
                if iy>Ny-3: cy2+=(Ny-3)-iy;
                #
                if iz<1: cz1+=1-iz;
                if iz>Nz-3: cz2+=(Nz-3)-iz;
                
                # Correct coefficients, so that the sum is one
                val = 0.0
                for cx in range(cx1, cx2):  val += ccx[cx]
                val = 1.0/val
                for cx in range(cx1, cx2):  ccx[cx] *= val
                #
                val = 0.0
                for cy in range(cy1, cy2):  val += ccy[cy]
                val = 1.0/val
                for cy in range(cy1, cy2):  ccy[cy] *= val
                #
                val = 0.0
                for cz in range(cz1, cz2):  val += ccz[cz]
                val = 1.0/val
                for cz in range(cz1, cz2):  ccz[cz] *= val
                
                # Combine elements 
                # No need to pre-calculate indices: the C compiler is well
                # capable of making these optimizations.
                val = 0.0
                for cz in range(cz1, cz2):
                    for cy in range(cy1, cy2):
                        for cx in range(cx1, cx2):
                            val += data_[iz+cz-1,iy+cy-1,ix+cx-1] * ccz[cz] * ccy[cy] * ccx[cx]
                result_[i] = val
            
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 2:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            dz = samplesz_[i]; iz = floor(dz); tz = dz-iz
            
            if (    ix >= 0 and ix < Nx-1 and
                    iy >= 0 and iy < Ny-1 and
                    iz >= 0 and iz < Nz-1       ):
                # Quasi-linear interpolation
                tx_ = -2*tx**3 + 3*tx**2
                ty_ = -2*ty**3 + 3*ty**2
                tz_ = -2*tz**3 + 3*tz**2
                #
                val =  data_[iz  ,iy,  ix  ] * (1.0-tz_) * (1.0-ty_) *(1.0-tx_)
                val += data_[iz  ,iy,  ix+1] * (1.0-tz_) * (1.0-ty_) *     tx_
                val += data_[iz  ,iy+1,ix  ] * (1.0-tz_) *      ty_  *(1.0-tx_)
                val += data_[iz  ,iy+1,ix+1] * (1.0-tz_) *      ty_  *     tx_
                #
                val += data_[iz+1,iy,  ix  ] *      tz_  * (1.0-ty_) *(1.0-tx_)
                val += data_[iz+1,iy,  ix+1] *      tz_  * (1.0-ty_) *     tx_
                val += data_[iz+1,iy+1,ix  ] *      tz_  *      ty_  *(1.0-tx_)
                val += data_[iz+1,iy+1,ix+1] *      tz_  *      ty_  *     tx_
                result_[i] = val
            
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5 and
                    dz>=-0.5 and dz<=Nz-0.5    ):
                # Edge effects
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                #
                if iy<0: ty+=iy; iy=0; 
                if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
                #
                if iz<0: tz+=iz; iz=0; 
                if iz>Nz-2: tz+=iz-(Nz-2); iz=Nz-2; 
                # Quasi-linear interpolation (edges)
                tx_ = -2*tx**3 + 3*tx**2
                ty_ = -2*ty**3 + 3*ty**2
                tz_ = -2*tz**3 + 3*tz**2
                #
                val =  data_[iz  ,iy,  ix  ] * (1.0-tz_) * (1.0-ty_) *(1.0-tx_)
                val += data_[iz  ,iy,  ix+1] * (1.0-tz_) * (1.0-ty_) *     tx_
                val += data_[iz  ,iy+1,ix  ] * (1.0-tz_) *      ty_  *(1.0-tx_)
                val += data_[iz  ,iy+1,ix+1] * (1.0-tz_) *      ty_  *     tx_
                #
                val += data_[iz+1,iy,  ix  ] *      tz_  * (1.0-ty_) *(1.0-tx_)
                val += data_[iz+1,iy,  ix+1] *      tz_  * (1.0-ty_) *     tx_
                val += data_[iz+1,iy+1,ix  ] *      tz_  *      ty_  *(1.0-tx_)
                val += data_[iz+1,iy+1,ix+1] *      tz_  *      ty_  *     tx_
                result_[i] = val
            
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 1:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            dz = samplesz_[i]; iz = floor(dz); tz = dz-iz
            
            if (    ix >= 0 and ix < Nx-1 and
                    iy >= 0 and iy < Ny-1 and
                    iz >= 0 and iz < Nz-1       ):
                # Linear interpolation
                val =  data_[iz  ,iy,  ix  ] * (1.0-tz) * (1.0-ty) * (1.0-tx)
                val += data_[iz  ,iy,  ix+1] * (1.0-tz) * (1.0-ty) *      tx
                val += data_[iz  ,iy+1,ix  ] * (1.0-tz) *      ty  * (1.0-tx)
                val += data_[iz  ,iy+1,ix+1] * (1.0-tz) *      ty  *      tx
                #
                val += data_[iz+1,iy,  ix  ] *      tz  * (1.0-ty) * (1.0-tx)
                val += data_[iz+1,iy,  ix+1] *      tz  * (1.0-ty) *      tx
                val += data_[iz+1,iy+1,ix  ] *      tz  *      ty  * (1.0-tx)
                val += data_[iz+1,iy+1,ix+1] *      tz  *      ty  *      tx
                result_[i] = val
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5 and
                    dz>=-0.5 and dz<=Nz-0.5    ):
                # Edge effects
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                #
                if iy<0: ty+=iy; iy=0; 
                if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
                #
                if iz<0: tz+=iz; iz=0; 
                if iz>Nz-2: tz+=iz-(Nz-2); iz=Nz-2; 
                # Linear interpolation (edges)
                val =  data_[iz  ,iy,  ix  ] * (1.0-tz) * (1.0-ty) * (1.0-tx)
                val += data_[iz  ,iy,  ix+1] * (1.0-tz) * (1.0-ty) *      tx
                val += data_[iz  ,iy+1,ix  ] * (1.0-tz) *      ty  * (1.0-tx)
                val += data_[iz  ,iy+1,ix+1] * (1.0-tz) *      ty  *      tx
                #
                val += data_[iz+1,iy,  ix  ] *      tz  * (1.0-ty) * (1.0-tx)
                val += data_[iz+1,iy,  ix+1] *      tz  * (1.0-ty) *      tx
                val += data_[iz+1,iy+1,ix  ] *      tz  *      ty  * (1.0-tx)
                val += data_[iz+1,iy+1,ix+1] *      tz  *      ty  *      tx
                result_[i] = val
            else:
                # Out of range
                result_[i] = 0.0
    
    else:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location
            dx = samplesx_[i]; ix = floor(dx+0.5)
            dy = samplesy_[i]; iy = floor(dy+0.5)
            dz = samplesz_[i]; iz = floor(dz+0.5)
            
            if (    ix >= 0 and ix < Nx and
                    iy >= 0 and iy < Ny and
                    iz >= 0 and iz < Nz     ):
                # Nearest neighbour interpolation
                result_[i] = data_[iz,iy,ix]
            else:
                # Out of range
                result_[i] = 0.0
    
    # Done
    result.shape = samplesx.shape
    return result



@cython.boundscheck(False)
@cython.wraparound(False)
def project1_32(data, deformx):
    """ Forward deformation of 1D base data of 32bit floats.
    """
    
    # Create result array with the shape of the samples
    result1 = np.zeros(data.shape, dtype=FLOAT32)
    
    # Create coefficient array to be able to test coverage 
    coeff1 = np.zeros(data.shape, dtype=FLOAT32)
    
    # Typecast, activate result1 as the result
    cdef np.ndarray[FLOAT32_T, ndim=1] result1_ = result1
    cdef np.ndarray[FLOAT32_T, ndim=1] coeff1_ = coeff1
    cdef np.ndarray[FLOAT32_T, ndim=1] data_ = data
    cdef np.ndarray[SAMPLE_T, ndim=1] deformx_ = deformx
    
    # Prepare sample location variables
    cdef int iz1, iy1, ix1 # integer pixel locations in source
    cdef int iz2, iy2, ix2 # integer pixel locations in dest
    cdef int iz3, iy3, ix3 # integer sub-locations in source
    cdef double z1, y1, x1 # full pixel locations in source
    cdef double z2, y2, x2 # full pixel locations in dest
    
    # For the bounding box
    cdef double z2_min, z2_max, y2_min, y2_max, x2_min, x2_max
    cdef int  iz2_min, iz2_max, iy2_min, iy2_max, ix2_min, ix2_max
    
    # More ...
    cdef double wx, wy, wz, w  # Weights
    cdef double rangeMax  # max range to determine kernel size
    cdef double val, c # For storing the temporary values
        
    # Get bounds
    cdef int Nz, Ny, Nx
    Nx = data.shape[0]
    
    
    for ix1 in range(0, Nx):
        
        # Calculate location to map to
        x2 = deformx_[ix1]
        
        # Select where the surrounding pixels map to.
        # This defines the region that we should fill in. This region
        # is overestimated as it is assumed rectangular in the destination,
        # which is not true in general.
        x2_min = x2
        x2_max = x2        
        for ix3 in range(-1,2):
            if ix3==0:
                continue
            if ix1+ix3 < 0:
                x2_min = x2 - 1000000.0 # Go minimal
                continue
            if ix1+ix3 >= Nx:
                x2_max = x2 + 1000000.0 # Go maximal
                continue
            # x border
            val = deformx_[ix1+ix3]
            x2_min = dmin(x2_min, val)
            x2_max = dmax(x2_max, val)
        
        
        # Limit to bounds and make integer
        x2_min = dmax(0, dmin(Nx-1, x2_min ))
        x2_max = dmax(0, dmin(Nx-1, x2_max ))
        #
        ix2_min = imax(0, floor(x2_min) )   # use max/min again to be sure
        ix2_max = imin(Nx-1, ceil(x2_max) )
        
        # Calculate max range to determine kernel size
        rangeMax = 0.1
        rangeMax = dmax( rangeMax, dmax(dabs(x2_min-x2), dabs(x2_max-x2)) )
        rangeMax = 1.0 / rangeMax # pre-divide
        
        # Sample value
        val = data_[ix1]
        
        # Splat value in destination
        for ix2 in range(ix2_min, ix2_max+1):
            
            # Calculate weights and make sure theyre > 0
            wx = 1.0 - rangeMax * dabs( <double>ix2 - x2)
            w = dmax(0.0, wx)
            
            # Assign values
            result1_[ix2  ] += val * w
            coeff1_[ ix2  ] +=       w
    
    
    # Divide by the coeffeicients
    for ix2 in range(Nx):
        
        c = coeff1_[ix2]
        if c>0:
            result1_[ix2] = result1_[ix2] / c
    
    # Done
    return result1_


@cython.boundscheck(False)
@cython.wraparound(False)
def project2_32(data, deformx, deformy):
    """ Forward deformation of 2D base data of 32bit floats.
    """
    
    # Create result array with the shape of the samples
    result1 = np.zeros(data.shape, dtype=FLOAT32)
    
    # Create coefficient array to be able to test coverage 
    coeff1 = np.zeros(data.shape, dtype=FLOAT32)
    
    # Typecast, activate result1 as the result
    cdef np.ndarray[FLOAT32_T, ndim=2] result1_ = result1
    cdef np.ndarray[FLOAT32_T, ndim=2] coeff1_ = coeff1
    cdef np.ndarray[FLOAT32_T, ndim=2] data_ = data
    cdef np.ndarray[SAMPLE_T, ndim=2] deformx_ = deformx
    cdef np.ndarray[SAMPLE_T, ndim=2] deformy_ = deformy
    
    # Prepare sample location variables
    cdef int iz1, iy1, ix1 # integer pixel locations in source
    cdef int iz2, iy2, ix2 # integer pixel locations in dest
    cdef int iz3, iy3, ix3 # integer sub-locations in source
    cdef double z1, y1, x1 # full pixel locations in source
    cdef double z2, y2, x2 # full pixel locations in dest
    
    # For the bounding box
    cdef double z2_min, z2_max, y2_min, y2_max, x2_min, x2_max
    cdef int  iz2_min, iz2_max, iy2_min, iy2_max, ix2_min, ix2_max
    
    # More ...
    cdef double wx, wy, wz, w  # Weights
    cdef double rangeMax  # max range to determine kernel size
    cdef double val, c # For storing the temporary values
        
    # Get bounds
    cdef int Nz, Ny, Nx
    Ny = data.shape[0]
    Nx = data.shape[1]
    
    cdef double rangeMax_sum = 0.0
    
    for iy1 in range(0, Ny):
        for ix1 in range(0, Nx):
            
            # Calculate location to map to
            y2 = deformy_[iy1,ix1]
            x2 = deformx_[iy1,ix1]
            
            # Select where the surrounding pixels map to.
            # This defines the region that we should fill in. This region
            # is overestimated as it is assumed rectangular in the destination,
            # which is not true in general.
            x2_min = x2
            x2_max = x2
            y2_min = y2
            y2_max = y2
            for iy3 in range(-1,2):
                for ix3 in range(-1,2):
                    if iy3*ix3==0:
                        continue
                    if iy1+iy3 < 0:
                        y2_min = y2 - 1000000.0 # Go minimal
                        continue
                    if ix1+ix3 < 0:
                        x2_min = x2 - 1000000.0 # Go minimal
                        continue
                    if iy1+iy3 >= Ny:
                        y2_max = y2 + 1000000.0 # Go maximal
                        continue
                    if ix1+ix3 >= Nx:
                        x2_max = x2 + 1000000.0 # Go maximal
                        continue
                    # x border
                    val = deformx_[iy1+iy3,ix1+ix3]
                    x2_min = dmin(x2_min, val)
                    x2_max = dmax(x2_max, val)
                    # y border
                    val = deformy_[iy1+iy3,ix1+ix3]
                    y2_min = dmin(y2_min, val)
                    y2_max = dmax(y2_max, val)
            
            # Limit to bounds and make integer
            x2_min = dmax(0, dmin(Nx-1, x2_min ))
            x2_max = dmax(0, dmin(Nx-1, x2_max ))
            y2_min = dmax(0, dmin(Ny-1, y2_min ))
            y2_max = dmax(0, dmin(Ny-1, y2_max ))
            #
            ix2_min = imax(0, floor(x2_min) )   # use max/min again to be sure
            ix2_max = imin(Nx-1, ceil(x2_max) )
            iy2_min = imax(0, floor(y2_min) )
            iy2_max = imin(Ny-1, ceil(y2_max) )
            
            # Calculate max range to determine kernel size
            rangeMax = 0.1
            rangeMax = dmax( rangeMax, dmax(dabs(y2_min-y2), dabs(y2_max-y2)) )
            rangeMax = dmax( rangeMax, dmax(dabs(x2_min-x2), dabs(x2_max-x2)) )
            
#             rangeMax_sum += rangeMax
            rangeMax = 1.0 / rangeMax # pre-divide
            
            # Sample value
            val = data_[iy1,ix1]
            
            # Splat value in destination
            for iy2 in range(iy2_min, iy2_max+1):
                for ix2 in range(ix2_min, ix2_max+1):
                    
                    # Calculate weights and make sure theyre > 0
                    wy = 1.0 - rangeMax * dabs( <double>iy2 - y2)
                    wx = 1.0 - rangeMax * dabs( <double>ix2 - x2)
                    w = dmax(0.0, wy) * dmax(0.0, wx)
                    
                    # Assign values
                    result1_[iy2  ,ix2  ] += val * w
                    coeff1_[ iy2  ,ix2  ] +=       w
    
    
#     print rangeMax_sum / (Nx*Ny)
    
    # Divide by the coeffeicients
    for iy2 in range(Ny):
        for ix2 in range(Nx):
            
            c = coeff1_[iy2,ix2]
            if c>0:
                result1_[iy2,ix2] /= c
    
    # Done
    return result1_

# todo: apply for 1D and 3D as well, modify project to use this if ata is a tuple
@cython.boundscheck(False)
@cython.wraparound(False)
def project22_32(datax, datay, deformx, deformy):
    """ Forward deformation of 2D base data of 32bit floats.
    """
    
    # Create result array with the shape of the samples
    resultx = np.zeros(datax.shape, dtype=FLOAT32) # DIFF
    resulty = np.zeros(datax.shape, dtype=FLOAT32)
    
    # Create coefficient array to be able to test coverage 
    coeff = np.zeros(datax.shape, dtype=FLOAT32) # DIFF
    
    
    # Typecast, activate resultxyz as the result
    cdef np.ndarray[FLOAT32_T, ndim=2] resultx_ = resultx # DIFF
    cdef np.ndarray[FLOAT32_T, ndim=2] resulty_ = resulty
    cdef np.ndarray[FLOAT32_T, ndim=2] coeff_ = coeff
    cdef np.ndarray[FLOAT32_T, ndim=2] datax_ = datax
    cdef np.ndarray[FLOAT32_T, ndim=2] datay_ = datay
    cdef np.ndarray[SAMPLE_T, ndim=2] deformx_ = deformx
    cdef np.ndarray[SAMPLE_T, ndim=2] deformy_ = deformy
    
    # Prepare sample location variables
    cdef int iz1, iy1, ix1 # integer pixel locations in source
    cdef int iz2, iy2, ix2 # integer pixel locations in dest
    cdef int iz3, iy3, ix3 # integer sub-locations in source
    cdef double z1, y1, x1 # full pixel locations in source
    cdef double z2, y2, x2 # full pixel locations in dest
    
    # For the bounding box
    cdef double z2_min, z2_max, y2_min, y2_max, x2_min, x2_max
    cdef int  iz2_min, iz2_max, iy2_min, iy2_max, ix2_min, ix2_max
    
    # More ...
    cdef double wx, wy, wz, w  # Weights
    cdef double rangeMax  # max range to determine kernel size
    cdef double valx, valy, valz, c # For storing the temporary values DIFF
        
    # Get bounds
    cdef int Nz, Ny, Nx
    Ny = datax.shape[0]
    Nx = datax.shape[1]
    
    cdef double rangeMax_sum = 0.0
    
    for iy1 in range(0, Ny):
        for ix1 in range(0, Nx):
            
            # Calculate location to map to
            y2 = deformy_[iy1,ix1]
            x2 = deformx_[iy1,ix1]
            
            # Select where the surrounding pixels map to.
            # This defines the region that we should fill in. This region
            # is overestimated as it is assumed rectangular in the destination,
            # which is not true in general.
            x2_min = x2
            x2_max = x2
            y2_min = y2
            y2_max = y2
            for iy3 in range(-1,2):
                for ix3 in range(-1,2):
                    if iy3*ix3==0:
                        continue
                    if iy1+iy3 < 0:
                        y2_min = y2 - 1000000.0 # Go minimal
                        continue
                    if ix1+ix3 < 0:
                        x2_min = x2 - 1000000.0 # Go minimal
                        continue
                    if iy1+iy3 >= Ny:
                        y2_max = y2 + 1000000.0 # Go maximal
                        continue
                    if ix1+ix3 >= Nx:
                        x2_max = x2 + 1000000.0 # Go maximal
                        continue
                    # x border
                    valx = deformx_[iy1+iy3,ix1+ix3]
                    x2_min = dmin(x2_min, valx)
                    x2_max = dmax(x2_max, valx)
                    # y border
                    valy = deformy_[iy1+iy3,ix1+ix3]
                    y2_min = dmin(y2_min, valy)
                    y2_max = dmax(y2_max, valy)
            
            # Limit to bounds and make integer
            x2_min = dmax(0, dmin(Nx-1, x2_min ))
            x2_max = dmax(0, dmin(Nx-1, x2_max ))
            y2_min = dmax(0, dmin(Ny-1, y2_min ))
            y2_max = dmax(0, dmin(Ny-1, y2_max ))
            #
            ix2_min = imax(0, floor(x2_min) )   # use max/min again to be sure
            ix2_max = imin(Nx-1, ceil(x2_max) )
            iy2_min = imax(0, floor(y2_min) )
            iy2_max = imin(Ny-1, ceil(y2_max) )
            
            # Calculate max range to determine kernel size
            rangeMax = 0.1
            rangeMax = dmax( rangeMax, dmax(dabs(y2_min-y2), dabs(y2_max-y2)) )
            rangeMax = dmax( rangeMax, dmax(dabs(x2_min-x2), dabs(x2_max-x2)) )
            
#             rangeMax_sum += rangeMax
            rangeMax = 1.0 / rangeMax # pre-divide
            
            # Sample value
            valx = datax_[iy1,ix1] # DIFF
            valy = datay_[iy1,ix1]
            
            # Splat value in destination
            for iy2 in range(iy2_min, iy2_max+1):
                for ix2 in range(ix2_min, ix2_max+1):
                    
                    # Calculate weights and make sure theyre > 0
                    wy = 1.0 - rangeMax * dabs( <double>iy2 - y2)
                    wx = 1.0 - rangeMax * dabs( <double>ix2 - x2)
                    w = dmax(0.0, wy) * dmax(0.0, wx)
                    
                    # Assign values
                    resultx_[iy2  ,ix2  ] += valx * w # DIFF
                    resulty_[iy2  ,ix2  ] += valy * w
                    coeff_[  iy2  ,ix2  ] +=        w
    
    
#     print rangeMax_sum / (Nx*Ny)
    
    # Divide by the coeffeicients
    for iy2 in range(Ny):
        for ix2 in range(Nx):
            
            c = coeff_[iy2,ix2]
            if c>0:
                c = 1.0/c
                resultx_[iy2,ix2] *= c # DIFF
                resulty_[iy2,ix2] *= c
    
    # Done
    return resultx_, resulty_ # DIFF


@cython.boundscheck(False)
@cython.wraparound(False)
def project3_32(data, deformx, deformy, deformz):
    """ Forward deformation of 3D base data of 32bit floats.
    """
    
    # Create result array with the shape of the samples
    result1 = np.zeros(data.shape, dtype=FLOAT32)
    
    # Create coefficient array to be able to test coverage 
    coeff1 = np.zeros(data.shape, dtype=FLOAT32)
    
    # Typecast, activate result1 as the result
    cdef np.ndarray[FLOAT32_T, ndim=3] result1_ = result1
    cdef np.ndarray[FLOAT32_T, ndim=3] coeff1_ = coeff1
    cdef np.ndarray[FLOAT32_T, ndim=3] data_ = data
    cdef np.ndarray[SAMPLE_T, ndim=3] deformx_ = deformx
    cdef np.ndarray[SAMPLE_T, ndim=3] deformy_ = deformy
    cdef np.ndarray[SAMPLE_T, ndim=3] deformz_ = deformz
    
    # Prepare sample location variables
    cdef int iz1, iy1, ix1 # integer pixel locations in source
    cdef int iz2, iy2, ix2 # integer pixel locations in dest
    cdef int iz3, iy3, ix3 # integer sub-locations in source
    cdef double z1, y1, x1 # full pixel locations in source
    cdef double z2, y2, x2 # full pixel locations in dest
    
    # For the bounding box
    cdef double z2_min, z2_max, y2_min, y2_max, x2_min, x2_max
    cdef int  iz2_min, iz2_max, iy2_min, iy2_max, ix2_min, ix2_max
    
    # More ...
    cdef double wx, wy, wz, w  # Weights
    cdef double rangeMax  # max range to determine kernel size
    cdef double val, c # For storing the temporary values
        
    # Get bounds
    cdef int Nz, Ny, Nx
    Nz = data.shape[0]
    Ny = data.shape[1]
    Nx = data.shape[2]
    
    
    for iz1 in range(0, Nz):
        for iy1 in range(0, Ny):
            for ix1 in range(0, Nx):
                
                # Calculate location to map to
                z2 = deformz_[iz1,iy1,ix1]
                y2 = deformy_[iz1,iy1,ix1]
                x2 = deformx_[iz1,iy1,ix1]
                
                # Select where the surrounding pixels map to.
                # This defines the region that we should fill in. This region
                # is overestimated as it is assumed rectangular in the destination,
                # which is not true in general.
                x2_min = x2
                x2_max = x2
                y2_min = y2
                y2_max = y2
                z2_min = z2
                z2_max = z2
                for iz3 in range(-1,2):
                    for iy3 in range(-1,2):
                        for ix3 in range(-1,2):
                            if iz3*iy3*ix3==0:
                                continue
                            if iz1+iz3 < 0:
                                z2_min = z2 - 1000000.0 # Go minimal
                                continue
                            if iy1+iy3 < 0:
                                y2_min = y2 - 1000000.0 # Go minimal
                                continue
                            if ix1+ix3 < 0:
                                x2_min = x2 - 1000000.0 # Go minimal
                                continue
                            if iz1+iz3 >= Nz:
                                z2_max = z2 + 1000000.0 # Go maximal
                                continue
                            if iy1+iy3 >= Ny:
                                y2_max = y2 + 1000000.0 # Go maximal
                                continue
                            if ix1+ix3 >= Nx:
                                x2_max = x2 + 1000000.0 # Go maximal
                                continue
                            # x border
                            val = deformx_[iz1+iz3,iy1+iy3,ix1+ix3]
                            x2_min = dmin(x2_min, val)
                            x2_max = dmax(x2_max, val)
                            # y border
                            val = deformy_[iz1+iz3,iy1+iy3,ix1+ix3]
                            y2_min = dmin(y2_min, val)
                            y2_max = dmax(y2_max, val)
                            # z border
                            val = deformz_[iz1+iz3,iy1+iy3,ix1+ix3]
                            z2_min = dmin(z2_min, val)
                            z2_max = dmax(z2_max, val)
                
                # Limit to bounds and make integer
                x2_min = dmax(0, dmin(Nx-1, x2_min ))
                x2_max = dmax(0, dmin(Nx-1, x2_max ))
                y2_min = dmax(0, dmin(Ny-1, y2_min ))
                y2_max = dmax(0, dmin(Ny-1, y2_max ))
                z2_min = dmax(0, dmin(Nz-1, z2_min ))
                z2_max = dmax(0, dmin(Nz-1, z2_max ))
                #
                ix2_min = imax(0, floor(x2_min) )   # use max/min again to be sure
                ix2_max = imin(Nx-1, ceil(x2_max) )
                iy2_min = imax(0, floor(y2_min) )
                iy2_max = imin(Ny-1, ceil(y2_max) )
                iz2_min = imax(0, floor(z2_min) )
                iz2_max = imin(Nz-1, ceil(z2_max) )
                
                # Calculate max range to determine kernel size
                rangeMax = 0.1
                rangeMax = dmax( rangeMax, dmax(dabs(z2_min-z2), dabs(z2_max-z2)) )
                rangeMax = dmax( rangeMax, dmax(dabs(y2_min-y2), dabs(y2_max-y2)) )
                rangeMax = dmax( rangeMax, dmax(dabs(x2_min-x2), dabs(x2_max-x2)) )
                rangeMax = 1.0 / rangeMax # pre-divide
                
                # Sample value
                val = data_[iz1,iy1,ix1]
                
                # Splat value in destination
                for iz2 in range(iz2_min, iz2_max+1):
                    for iy2 in range(iy2_min, iy2_max+1):
                        for ix2 in range(ix2_min, ix2_max+1):
                            
                            # Calculate weights and make sure theyre > 0
                            wz = 1.0 - rangeMax * dabs( <double>iz2 - z2)
                            wy = 1.0 - rangeMax * dabs( <double>iy2 - y2)
                            wx = 1.0 - rangeMax * dabs( <double>ix2 - x2)
                            w = dmax(0.0, wz) * dmax(0.0, wy) * dmax(0.0, wx)
                            
                            # Assign values
                            result1_[iz2, iy2, ix2] += val * w
                            coeff1_[ iz2, iy2, ix2] +=       w
    
    
    # Divide by the coeffeicients
    for iz2 in range(Nz):
        for iy2 in range(Ny):
            for ix2 in range(Nx):
                
                c = coeff1_[iz2,iy2,ix2]
                if c>0:
                    result1_[iz2,iy2,ix2] = result1_[iz2,iy2,ix2] / c
    
    # Done
    return result1_


## Float64 backward and forward mapping functions
# - Copy the whole above cell
# - replace 32 with 64 (for the rest of the file)


@cython.boundscheck(False)
@cython.wraparound(False)
def interp1_64(data, samplesx, order, spline_type=0.0):
    """ Interpolation of 1D base data of 64bit floats
    """
    
    # Create result array with the shape of the samples
    result = np.empty((samplesx.size,), dtype=FLOAT64)
    
    # Typecast (flatten samples)
    cdef np.ndarray[FLOAT64_T, ndim=1] data_ = data
    cdef np.ndarray[FLOAT64_T, ndim=1] result_ = result
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesx_ = samplesx.ravel()
    
    # Create and init cubic interpolator
    cdef CoefLut lut
    cdef AccurateCoef coeffx
    cdef double *ccx
    cdef double splineId = 0.0
    if order > 1:
        lut = CoefLut.get_lut(spline_type)
        splineId = lut.spline_type_to_id(spline_type)
        coeffx = AccurateCoef(lut)
    
    # Prepare sample location variables
    cdef double dx, tx, tx_
    cdef int ix
    cdef int cx, cx1, cx2
    
    # Prepare indices and bounds, etc
    cdef int i
    cdef int Ni = samplesx.shape[0]
    cdef int Nx = data.shape[0]
    cdef double val
    
    
    if order == 3:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            
            if ix >= 1 and ix < Nx-2:
                # Cubic interpolation
                ccx = coeffx.get_coef(tx)
                val =  data_[ix-1] * ccx[0]
                val += data_[ix  ] * ccx[1]
                val += data_[ix+1] * ccx[2]
                val += data_[ix+2] * ccx[3]
                result_[i] = val
            
            elif dx>=-0.5 and dx<=Nx-0.5:
                # Edge effects
                
                # Get coefficients
                ccx = coeffx.get_coef(tx)
                
                # Correct stuff: calculate offset (max 2)
                cx1, cx2 = 0, 4
                #
                if ix<1: cx1+=1-ix;
                if ix>Nx-3: cx2+=(Nx-3)-ix;
                
                # Correct coefficients, so that the sum is one
                val = 0.0
                for cx in range(cx1, cx2):  val += ccx[cx]
                val = 1.0/val
                for cx in range(cx1, cx2):  ccx[cx] *= val
                
                # Combine elements
                val = 0.0
                for cx in range(cx1, cx2):
                    val += data_[ix+cx-1] * ccx[cx]
                result_[i] = val
                
                # Code below produces nice results too, but can only be 
                # used for 1D; higher dimensions get weird results
                
#                 # Correct indices
#                 if ix<1: tx+=ix-1; ix=1; 
#                 if ix>Nx-3: tx+=ix-(Nx-3); ix=Nx-3;
#                 
#                 # Get coefficients (first fill, so we can use its memory)
#                 ccx = coeffx.get_coef(0.0)
#                 if splineId < 2.0:
#                     lut.cubicsplinecoef_cardinal_edge(tx, ccx)
#                 else:
#                     lut.cubicsplinecoef(splineId, tx, ccx)
#                 
#                 # Combine elements
#                 val = 0.0
#                 for cx in range(0,4):
#                     cx2 = ix + cx - 1 
#                     val += data_[cx2] * ccx[cx]
#                 result_[i] = val
            
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 2:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            
            if ix >= 0 and ix < Nx-1:
                # Quasi-linear interpolation
                tx_ = -2*tx**3 + 3*tx**2
                val =  data_[ix] * (1.0-tx_)
                val += data_[ix+1] * tx_
                result_[i] = val
            elif dx>=-0.5 and dx<=Nx-0.5:                
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                # Quasi-linear interpolation (edges)
                tx_ = -2*tx**3 + 3*tx**2
                val =  data_[ix] * (1.0-tx_)
                val += data_[ix+1] * tx_
                result_[i] = val
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 1:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            
            if ix >= 0 and ix < Nx-1:
                # Linear interpolation
                val =  data_[ix] * (1.0-tx)
                val += data_[ix+1] * tx
                result_[i] = val
            elif dx>=-0.5 and dx<=Nx-0.5:
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                # Linear interpolation (edges)
                val =  data_[ix] * (1.0-tx)
                val += data_[ix+1] * tx
                result_[i] = val
            else:
                # Out of range
                result_[i] = 0.0
    
    else:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location
            dx = samplesx_[i]; ix = floor(dx+0.5)
            
            if ix >= 0 and ix < Nx:
                # Nearest neighbour interpolation
                result_[i] = data_[ix]
            else:
                # Out of range
                result_[i] = 0.0
    
    # Done
    result.shape = samplesx.shape
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def interp2_64(data, samplesx, samplesy, order, spline_type=0.0):
    """ Interpolation of 2D base data of 64bit floats
    """    
    
    # Create result array with the shape of the samples
    result = np.empty((samplesx.size,), dtype=FLOAT64)
    
    # Flatten and type the samples and result
    cdef np.ndarray[FLOAT64_T, ndim=2] data_ = data
    cdef np.ndarray[FLOAT64_T, ndim=1] result_ = result
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesx_ = samplesx.ravel()
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesy_ = samplesy.ravel()
    
    # Create and init cubic interpolator
    cdef CoefLut lut
    cdef AccurateCoef coeffx, coeffy
    cdef double splineId = 0.0
    cdef double *ccx 
    cdef double *ccy
    if order > 1:
        lut = CoefLut.get_lut(spline_type)
        splineId = lut.spline_type_to_id(spline_type)
        coeffx = AccurateCoef(lut)
        coeffy = AccurateCoef(lut)
    
    # Prepare sample location variables
    cdef double dx, tx, tx_, dy, ty, ty_
    cdef int ix, iy
    cdef int cx, cy
    cdef int cx1, cx2, cy1, cy2
    cdef double valFactor
    
    # Prepare indices and bounds, etc
    cdef int i
    cdef int Ni = samplesx.size
    cdef int Ny = data.shape[0]
    cdef int Nx = data.shape[1]
    cdef double val
    
    
    if order == 3:
       
        # with nogil: does not make it faster
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            
            if (    ix >= 1 and ix < Nx-2 and 
                    iy >= 1 and iy < Ny-2       ):
                # Cubic interpolation
                ccx = coeffx.get_coef(tx)
                ccy = coeffy.get_coef(ty)
                val = 0.0
                for cy in range(4):
                    for cx in range(4):
                        val += data_[iy+cy-1,ix+cx-1] * ccy[cy] * ccx[cx]
                result_[i] = val
            
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5     ):
                # Edge effects
                
                # Get coefficients
                ccx = coeffx.get_coef(tx)
                ccy = coeffy.get_coef(ty)
                
                # Correct stuff: calculate offset (max 2)
                cx1, cx2 = 0, 4
                cy1, cy2 = 0, 4
                #
                if ix<1: cx1+=1-ix;
                if ix>Nx-3: cx2+=(Nx-3)-ix;
                #
                if iy<1: cy1+=1-iy;
                if iy>Ny-3: cy2+=(Ny-3)-iy;
                
                # Correct coefficients, so that the sum is one
                val = 0.0
                for cx in range(cx1, cx2):  val += ccx[cx]
                val = 1.0/val
                for cx in range(cx1, cx2):  ccx[cx] *= val
                #
                val = 0.0
                for cy in range(cy1, cy2):  val += ccy[cy]
                val = 1.0/val
                for cy in range(cy1, cy2):  ccy[cy] *= val
                
                # Combine elements
                # No need to pre-calculate indices: the C compiler is well
                # capable of making these optimizations.
                val = 0.0
                for cy in range(cy1, cy2):
                    for cx in range(cx1, cx2):
                        val += data_[iy+cy-1,ix+cx-1] * ccy[cy] * ccx[cx]
                result_[i] = val
            
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 2:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            
            if (    ix >= 0 and ix < Nx-1 and
                    iy >= 0 and iy < Ny-1     ):
                # Quasi-linear interpolation
                tx_ = -2*tx**3 + 3*tx**2
                ty_ = -2*ty**3 + 3*ty**2
                val =  data_[iy,  ix  ] * (1.0-ty_) * (1.0-tx_)
                val += data_[iy,  ix+1] * (1.0-ty_) *      tx_
                val += data_[iy+1,ix  ] *      ty_  * (1.0-tx_)
                val += data_[iy+1,ix+1] *      ty_  *      tx_
                result_[i] = val
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5     ):
                # Edge effects
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                #
                if iy<0: ty+=iy; iy=0; 
                if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
                # Quasi-linear interpolation (edges)
                tx_ = -2*tx**3 + 3*tx**2
                ty_ = -2*ty**3 + 3*ty**2
                val =  data_[iy,  ix  ] * (1.0-ty_) * (1.0-tx_)
                val += data_[iy,  ix+1] * (1.0-ty_) *      tx_
                val += data_[iy+1,ix  ] *      ty_  * (1.0-tx_)
                val += data_[iy+1,ix+1] *      ty_  *      tx_
                result_[i] = val
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 1:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            
            if (    ix >= 0 and ix < Nx-1 and
                    iy >= 0 and iy < Ny-1     ):
                # Linear interpolation
                val =  data_[iy,  ix  ] * (1.0-ty) * (1.0-tx)
                val += data_[iy,  ix+1] * (1.0-ty) *      tx
                val += data_[iy+1,ix  ] *      ty  * (1.0-tx)
                val += data_[iy+1,ix+1] *      ty  *      tx
                result_[i] = val
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5     ):
                # Edge effects
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                #
                if iy<0: ty+=iy; iy=0; 
                if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
                # Linear interpolation (edges)
                val =  data_[iy,  ix  ] * (1.0-ty) * (1.0-tx)
                val += data_[iy,  ix+1] * (1.0-ty) *      tx
                val += data_[iy+1,ix  ] *      ty  * (1.0-tx)
                val += data_[iy+1,ix+1] *      ty  *      tx
                result_[i] = val
            else:
                # Out of range
                result_[i] = 0.0
    
    else:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location
            dx = samplesx_[i]; ix = floor(dx+0.5)
            dy = samplesy_[i]; iy = floor(dy+0.5)
            
            if (    ix >= 0 and ix < Nx and
                    iy >= 0 and iy < Ny     ):
                # Nearest neighbour interpolation
                result_[i] = data_[iy,ix]
            else:
                # Out of range
                result_[i] = 0.0
    
    # Done
    result.shape = samplesx.shape
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def interp3_64(data, samplesx, samplesy, samplesz, order, spline_type=0.0):
    """ Interpolation of 3D base data of 64bit floats
    """    
    
    # Create result array with the shape of the samples
    result = np.empty((samplesx.size,), dtype=FLOAT64)
    
    # Flatten and type the samples and result
    cdef np.ndarray[FLOAT64_T, ndim=3] data_ = data
    cdef np.ndarray[FLOAT64_T, ndim=1] result_ = result
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesx_ = samplesx.ravel()
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesy_ = samplesy.ravel()
    cdef np.ndarray[SAMPLE_T, ndim=1] samplesz_ = samplesz.ravel()
    
    # Create and init cubic interpolator
    cdef CoefLut lut
    cdef AccurateCoef coeffx, coeffy, coeffz
    cdef double splineId = 0.0
    cdef double *ccx
    cdef double *ccy
    cdef double *ccz
    if order > 1:
        lut = CoefLut.get_lut(spline_type)
        splineId = lut.spline_type_to_id(spline_type)
        coeffx = AccurateCoef(lut)
        coeffy = AccurateCoef(lut)
        coeffz = AccurateCoef(lut)
    
    # Prepare sample location variables
    cdef double dx, tx, tx_, dy, ty, ty_, dz, tz, tz_
    cdef int ix, iy, iz
    cdef int cx, cy, cz
    cdef int cx1, cx2, cy1, cy2, cz1, cz2
    cdef double valFactor
    
    # Prepare indices and bounds, etc
    cdef int i
    cdef int Ni = samplesx.size
    cdef int Nz = data.shape[0]
    cdef int Ny = data.shape[1]
    cdef int Nx = data.shape[2]
    cdef double val
    
    
    if order == 3:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            dz = samplesz_[i]; iz = floor(dz); tz = dz-iz
            
            if (    ix >= 1 and ix < Nx-2 and 
                    iy >= 1 and iy < Ny-2 and
                    iz >= 1 and iz < Nz-2       ):
                # Cubic interpolation
                ccx = coeffx.get_coef(tx)
                ccy = coeffy.get_coef(ty)
                ccz = coeffz.get_coef(tz)
                val = 0.0
                for cz in range(4):
                    for cy in range(4):
                        for cx in range(4):
                            val += data_[iz+cz-1,iy+cy-1,ix+cx-1] * (
                                            ccz[cz] * ccy[cy] * ccx[cx] )
                result_[i] = val
            
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5 and
                    dz>=-0.5 and dz<=Nz-0.5     ):
                # Edge effects
                
                # Get coefficients
                ccx = coeffx.get_coef(tx)
                ccy = coeffy.get_coef(ty)
                ccz = coeffz.get_coef(tz)
                
                # Correct stuff: calculate offset (max 2)
                cx1, cx2 = 0, 4
                cy1, cy2 = 0, 4
                cz1, cz2 = 0, 4
                #
                if ix<1: cx1+=1-ix;
                if ix>Nx-3: cx2+=(Nx-3)-ix;
                #
                if iy<1: cy1+=1-iy;
                if iy>Ny-3: cy2+=(Ny-3)-iy;
                #
                if iz<1: cz1+=1-iz;
                if iz>Nz-3: cz2+=(Nz-3)-iz;
                
                # Correct coefficients, so that the sum is one
                val = 0.0
                for cx in range(cx1, cx2):  val += ccx[cx]
                val = 1.0/val
                for cx in range(cx1, cx2):  ccx[cx] *= val
                #
                val = 0.0
                for cy in range(cy1, cy2):  val += ccy[cy]
                val = 1.0/val
                for cy in range(cy1, cy2):  ccy[cy] *= val
                #
                val = 0.0
                for cz in range(cz1, cz2):  val += ccz[cz]
                val = 1.0/val
                for cz in range(cz1, cz2):  ccz[cz] *= val
                
                # Combine elements 
                # No need to pre-calculate indices: the C compiler is well
                # capable of making these optimizations.
                val = 0.0
                for cz in range(cz1, cz2):
                    for cy in range(cy1, cy2):
                        for cx in range(cx1, cx2):
                            val += data_[iz+cz-1,iy+cy-1,ix+cx-1] * ccz[cz] * ccy[cy] * ccx[cx]
                result_[i] = val
            
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 2:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            dz = samplesz_[i]; iz = floor(dz); tz = dz-iz
            
            if (    ix >= 0 and ix < Nx-1 and
                    iy >= 0 and iy < Ny-1 and
                    iz >= 0 and iz < Nz-1       ):
                # Quasi-linear interpolation
                tx_ = -2*tx**3 + 3*tx**2
                ty_ = -2*ty**3 + 3*ty**2
                tz_ = -2*tz**3 + 3*tz**2
                #
                val =  data_[iz  ,iy,  ix  ] * (1.0-tz_) * (1.0-ty_) *(1.0-tx_)
                val += data_[iz  ,iy,  ix+1] * (1.0-tz_) * (1.0-ty_) *     tx_
                val += data_[iz  ,iy+1,ix  ] * (1.0-tz_) *      ty_  *(1.0-tx_)
                val += data_[iz  ,iy+1,ix+1] * (1.0-tz_) *      ty_  *     tx_
                #
                val += data_[iz+1,iy,  ix  ] *      tz_  * (1.0-ty_) *(1.0-tx_)
                val += data_[iz+1,iy,  ix+1] *      tz_  * (1.0-ty_) *     tx_
                val += data_[iz+1,iy+1,ix  ] *      tz_  *      ty_  *(1.0-tx_)
                val += data_[iz+1,iy+1,ix+1] *      tz_  *      ty_  *     tx_
                result_[i] = val
            
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5 and
                    dz>=-0.5 and dz<=Nz-0.5    ):
                # Edge effects
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                #
                if iy<0: ty+=iy; iy=0; 
                if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
                #
                if iz<0: tz+=iz; iz=0; 
                if iz>Nz-2: tz+=iz-(Nz-2); iz=Nz-2; 
                # Quasi-linear interpolation (edges)
                tx_ = -2*tx**3 + 3*tx**2
                ty_ = -2*ty**3 + 3*ty**2
                tz_ = -2*tz**3 + 3*tz**2
                #
                val =  data_[iz  ,iy,  ix  ] * (1.0-tz_) * (1.0-ty_) *(1.0-tx_)
                val += data_[iz  ,iy,  ix+1] * (1.0-tz_) * (1.0-ty_) *     tx_
                val += data_[iz  ,iy+1,ix  ] * (1.0-tz_) *      ty_  *(1.0-tx_)
                val += data_[iz  ,iy+1,ix+1] * (1.0-tz_) *      ty_  *     tx_
                #
                val += data_[iz+1,iy,  ix  ] *      tz_  * (1.0-ty_) *(1.0-tx_)
                val += data_[iz+1,iy,  ix+1] *      tz_  * (1.0-ty_) *     tx_
                val += data_[iz+1,iy+1,ix  ] *      tz_  *      ty_  *(1.0-tx_)
                val += data_[iz+1,iy+1,ix+1] *      tz_  *      ty_  *     tx_
                result_[i] = val
            
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 1:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
            dz = samplesz_[i]; iz = floor(dz); tz = dz-iz
            
            if (    ix >= 0 and ix < Nx-1 and
                    iy >= 0 and iy < Ny-1 and
                    iz >= 0 and iz < Nz-1       ):
                # Linear interpolation
                val =  data_[iz  ,iy,  ix  ] * (1.0-tz) * (1.0-ty) * (1.0-tx)
                val += data_[iz  ,iy,  ix+1] * (1.0-tz) * (1.0-ty) *      tx
                val += data_[iz  ,iy+1,ix  ] * (1.0-tz) *      ty  * (1.0-tx)
                val += data_[iz  ,iy+1,ix+1] * (1.0-tz) *      ty  *      tx
                #
                val += data_[iz+1,iy,  ix  ] *      tz  * (1.0-ty) * (1.0-tx)
                val += data_[iz+1,iy,  ix+1] *      tz  * (1.0-ty) *      tx
                val += data_[iz+1,iy+1,ix  ] *      tz  *      ty  * (1.0-tx)
                val += data_[iz+1,iy+1,ix+1] *      tz  *      ty  *      tx
                result_[i] = val
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5 and
                    dz>=-0.5 and dz<=Nz-0.5    ):
                # Edge effects
                if ix<0: tx+=ix; ix=0; 
                if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
                #
                if iy<0: ty+=iy; iy=0; 
                if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
                #
                if iz<0: tz+=iz; iz=0; 
                if iz>Nz-2: tz+=iz-(Nz-2); iz=Nz-2; 
                # Linear interpolation (edges)
                val =  data_[iz  ,iy,  ix  ] * (1.0-tz) * (1.0-ty) * (1.0-tx)
                val += data_[iz  ,iy,  ix+1] * (1.0-tz) * (1.0-ty) *      tx
                val += data_[iz  ,iy+1,ix  ] * (1.0-tz) *      ty  * (1.0-tx)
                val += data_[iz  ,iy+1,ix+1] * (1.0-tz) *      ty  *      tx
                #
                val += data_[iz+1,iy,  ix  ] *      tz  * (1.0-ty) * (1.0-tx)
                val += data_[iz+1,iy,  ix+1] *      tz  * (1.0-ty) *      tx
                val += data_[iz+1,iy+1,ix  ] *      tz  *      ty  * (1.0-tx)
                val += data_[iz+1,iy+1,ix+1] *      tz  *      ty  *      tx
                result_[i] = val
            else:
                # Out of range
                result_[i] = 0.0
    
    else:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location
            dx = samplesx_[i]; ix = floor(dx+0.5)
            dy = samplesy_[i]; iy = floor(dy+0.5)
            dz = samplesz_[i]; iz = floor(dz+0.5)
            
            if (    ix >= 0 and ix < Nx and
                    iy >= 0 and iy < Ny and
                    iz >= 0 and iz < Nz     ):
                # Nearest neighbour interpolation
                result_[i] = data_[iz,iy,ix]
            else:
                # Out of range
                result_[i] = 0.0
    
    # Done
    result.shape = samplesx.shape
    return result



@cython.boundscheck(False)
@cython.wraparound(False)
def project1_64(data, deformx):
    """ Forward deformation of 1D base data of 64bit floats.
    """
    
    # Create result array with the shape of the samples
    result1 = np.zeros(data.shape, dtype=FLOAT64)
    
    # Create coefficient array to be able to test coverage 
    coeff1 = np.zeros(data.shape, dtype=FLOAT64)
    
    # Typecast, activate result1 as the result
    cdef np.ndarray[FLOAT64_T, ndim=1] result1_ = result1
    cdef np.ndarray[FLOAT64_T, ndim=1] coeff1_ = coeff1
    cdef np.ndarray[FLOAT64_T, ndim=1] data_ = data
    cdef np.ndarray[SAMPLE_T, ndim=1] deformx_ = deformx
    
    # Prepare sample location variables
    cdef int iz1, iy1, ix1 # integer pixel locations in source
    cdef int iz2, iy2, ix2 # integer pixel locations in dest
    cdef int iz3, iy3, ix3 # integer sub-locations in source
    cdef double z1, y1, x1 # full pixel locations in source
    cdef double z2, y2, x2 # full pixel locations in dest
    
    # For the bounding box
    cdef double z2_min, z2_max, y2_min, y2_max, x2_min, x2_max
    cdef int  iz2_min, iz2_max, iy2_min, iy2_max, ix2_min, ix2_max
    
    # More ...
    cdef double wx, wy, wz, w  # Weights
    cdef double rangeMax  # max range to determine kernel size
    cdef double val, c # For storing the temporary values
        
    # Get bounds
    cdef int Nz, Ny, Nx
    Nx = data.shape[0]
    
    
    for ix1 in range(0, Nx):
        
        # Calculate location to map to
        x2 = deformx_[ix1]
        
        # Select where the surrounding pixels map to.
        # This defines the region that we should fill in. This region
        # is overestimated as it is assumed rectangular in the destination,
        # which is not true in general.
        x2_min = x2
        x2_max = x2        
        for ix3 in range(-1,2):
            if ix3==0:
                continue
            if ix1+ix3 < 0:
                x2_min = x2 - 1000000.0 # Go minimal
                continue
            if ix1+ix3 >= Nx:
                x2_max = x2 + 1000000.0 # Go maximal
                continue
            # x border
            val = deformx_[ix1+ix3]
            x2_min = dmin(x2_min, val)
            x2_max = dmax(x2_max, val)
        
        
        # Limit to bounds and make integer
        x2_min = dmax(0, dmin(Nx-1, x2_min ))
        x2_max = dmax(0, dmin(Nx-1, x2_max ))
        #
        ix2_min = imax(0, floor(x2_min) )   # use max/min again to be sure
        ix2_max = imin(Nx-1, ceil(x2_max) )
        
        # Calculate max range to determine kernel size
        rangeMax = 0.1
        rangeMax = dmax( rangeMax, dmax(dabs(x2_min-x2), dabs(x2_max-x2)) )
        rangeMax = 1.0 / rangeMax # pre-divide
        
        # Sample value
        val = data_[ix1]
        
        # Splat value in destination
        for ix2 in range(ix2_min, ix2_max+1):
            
            # Calculate weights and make sure theyre > 0
            wx = 1.0 - rangeMax * dabs( <double>ix2 - x2)
            w = dmax(0.0, wx)
            
            # Assign values
            result1_[ix2  ] += val * w
            coeff1_[ ix2  ] +=       w
    
    
    # Divide by the coeffeicients
    for ix2 in range(Nx):
        
        c = coeff1_[ix2]
        if c>0:
            result1_[ix2] = result1_[ix2] / c
    
    # Done
    return result1_


@cython.boundscheck(False)
@cython.wraparound(False)
def project2_64(data, deformx, deformy):
    """ Forward deformation of 2D base data of 64bit floats.
    """
    
    # Create result array with the shape of the samples
    result1 = np.zeros(data.shape, dtype=FLOAT64)
    
    # Create coefficient array to be able to test coverage 
    coeff1 = np.zeros(data.shape, dtype=FLOAT64)
    
    # Typecast, activate result1 as the result
    cdef np.ndarray[FLOAT64_T, ndim=2] result1_ = result1
    cdef np.ndarray[FLOAT64_T, ndim=2] coeff1_ = coeff1
    cdef np.ndarray[FLOAT64_T, ndim=2] data_ = data
    cdef np.ndarray[SAMPLE_T, ndim=2] deformx_ = deformx
    cdef np.ndarray[SAMPLE_T, ndim=2] deformy_ = deformy
    
    # Prepare sample location variables
    cdef int iz1, iy1, ix1 # integer pixel locations in source
    cdef int iz2, iy2, ix2 # integer pixel locations in dest
    cdef int iz3, iy3, ix3 # integer sub-locations in source
    cdef double z1, y1, x1 # full pixel locations in source
    cdef double z2, y2, x2 # full pixel locations in dest
    
    # For the bounding box
    cdef double z2_min, z2_max, y2_min, y2_max, x2_min, x2_max
    cdef int  iz2_min, iz2_max, iy2_min, iy2_max, ix2_min, ix2_max
    
    # More ...
    cdef double wx, wy, wz, w  # Weights
    cdef double rangeMax  # max range to determine kernel size
    cdef double val, c # For storing the temporary values
        
    # Get bounds
    cdef int Nz, Ny, Nx
    Ny = data.shape[0]
    Nx = data.shape[1]
    
    cdef double rangeMax_sum = 0.0
    
    for iy1 in range(0, Ny):
        for ix1 in range(0, Nx):
            
            # Calculate location to map to
            y2 = deformy_[iy1,ix1]
            x2 = deformx_[iy1,ix1]
            
            # Select where the surrounding pixels map to.
            # This defines the region that we should fill in. This region
            # is overestimated as it is assumed rectangular in the destination,
            # which is not true in general.
            x2_min = x2
            x2_max = x2
            y2_min = y2
            y2_max = y2
            for iy3 in range(-1,2):
                for ix3 in range(-1,2):
                    if iy3*ix3==0:
                        continue
                    if iy1+iy3 < 0:
                        y2_min = y2 - 1000000.0 # Go minimal
                        continue
                    if ix1+ix3 < 0:
                        x2_min = x2 - 1000000.0 # Go minimal
                        continue
                    if iy1+iy3 >= Ny:
                        y2_max = y2 + 1000000.0 # Go maximal
                        continue
                    if ix1+ix3 >= Nx:
                        x2_max = x2 + 1000000.0 # Go maximal
                        continue
                    # x border
                    val = deformx_[iy1+iy3,ix1+ix3]
                    x2_min = dmin(x2_min, val)
                    x2_max = dmax(x2_max, val)
                    # y border
                    val = deformy_[iy1+iy3,ix1+ix3]
                    y2_min = dmin(y2_min, val)
                    y2_max = dmax(y2_max, val)
            
            # Limit to bounds and make integer
            x2_min = dmax(0, dmin(Nx-1, x2_min ))
            x2_max = dmax(0, dmin(Nx-1, x2_max ))
            y2_min = dmax(0, dmin(Ny-1, y2_min ))
            y2_max = dmax(0, dmin(Ny-1, y2_max ))
            #
            ix2_min = imax(0, floor(x2_min) )   # use max/min again to be sure
            ix2_max = imin(Nx-1, ceil(x2_max) )
            iy2_min = imax(0, floor(y2_min) )
            iy2_max = imin(Ny-1, ceil(y2_max) )
            
            # Calculate max range to determine kernel size
            rangeMax = 0.1
            rangeMax = dmax( rangeMax, dmax(dabs(y2_min-y2), dabs(y2_max-y2)) )
            rangeMax = dmax( rangeMax, dmax(dabs(x2_min-x2), dabs(x2_max-x2)) )
            
#             rangeMax_sum += rangeMax
            rangeMax = 1.0 / rangeMax # pre-divide
            
            # Sample value
            val = data_[iy1,ix1]
            
            # Splat value in destination
            for iy2 in range(iy2_min, iy2_max+1):
                for ix2 in range(ix2_min, ix2_max+1):
                    
                    # Calculate weights and make sure theyre > 0
                    wy = 1.0 - rangeMax * dabs( <double>iy2 - y2)
                    wx = 1.0 - rangeMax * dabs( <double>ix2 - x2)
                    w = dmax(0.0, wy) * dmax(0.0, wx)
                    
                    # Assign values
                    result1_[iy2  ,ix2  ] += val * w
                    coeff1_[ iy2  ,ix2  ] +=       w
    
    
#     print rangeMax_sum / (Nx*Ny)
    
    # Divide by the coeffeicients
    for iy2 in range(Ny):
        for ix2 in range(Nx):
            
            c = coeff1_[iy2,ix2]
            if c>0:
                result1_[iy2,ix2] /= c
    
    # Done
    return result1_


@cython.boundscheck(False)
@cython.wraparound(False)
def project3_64(data, deformx, deformy, deformz):
    """ Forward deformation of 3D base data of 64bit floats.
    """
    
    # Create result array with the shape of the samples
    result1 = np.zeros(data.shape, dtype=FLOAT64)
    
    # Create coefficient array to be able to test coverage 
    coeff1 = np.zeros(data.shape, dtype=FLOAT64)
    
    # Typecast, activate result1 as the result
    cdef np.ndarray[FLOAT64_T, ndim=3] result1_ = result1
    cdef np.ndarray[FLOAT64_T, ndim=3] coeff1_ = coeff1
    cdef np.ndarray[FLOAT64_T, ndim=3] data_ = data
    cdef np.ndarray[SAMPLE_T, ndim=3] deformx_ = deformx
    cdef np.ndarray[SAMPLE_T, ndim=3] deformy_ = deformy
    cdef np.ndarray[SAMPLE_T, ndim=3] deformz_ = deformz
    
    # Prepare sample location variables
    cdef int iz1, iy1, ix1 # integer pixel locations in source
    cdef int iz2, iy2, ix2 # integer pixel locations in dest
    cdef int iz3, iy3, ix3 # integer sub-locations in source
    cdef double z1, y1, x1 # full pixel locations in source
    cdef double z2, y2, x2 # full pixel locations in dest
    
    # For the bounding box
    cdef double z2_min, z2_max, y2_min, y2_max, x2_min, x2_max
    cdef int  iz2_min, iz2_max, iy2_min, iy2_max, ix2_min, ix2_max
    
    # More ...
    cdef double wx, wy, wz, w  # Weights
    cdef double rangeMax  # max range to determine kernel size
    cdef double val, c # For storing the temporary values
        
    # Get bounds
    cdef int Nz, Ny, Nx
    Nz = data.shape[0]
    Ny = data.shape[1]
    Nx = data.shape[2]
    
    
    for iz1 in range(0, Nz):
        for iy1 in range(0, Ny):
            for ix1 in range(0, Nx):
                
                # Calculate location to map to
                z2 = deformz_[iz1,iy1,ix1]
                y2 = deformy_[iz1,iy1,ix1]
                x2 = deformx_[iz1,iy1,ix1]
                
                # Select where the surrounding pixels map to.
                # This defines the region that we should fill in. This region
                # is overestimated as it is assumed rectangular in the destination,
                # which is not true in general.
                x2_min = x2
                x2_max = x2
                y2_min = y2
                y2_max = y2
                z2_min = z2
                z2_max = z2
                for iz3 in range(-1,2):
                    for iy3 in range(-1,2):
                        for ix3 in range(-1,2):
                            if iz3*iy3*ix3==0:
                                continue
                            if iz1+iz3 < 0:
                                z2_min = z2 - 1000000.0 # Go minimal
                                continue
                            if iy1+iy3 < 0:
                                y2_min = y2 - 1000000.0 # Go minimal
                                continue
                            if ix1+ix3 < 0:
                                x2_min = x2 - 1000000.0 # Go minimal
                                continue
                            if iz1+iz3 >= Nz:
                                z2_max = z2 + 1000000.0 # Go maximal
                                continue
                            if iy1+iy3 >= Ny:
                                y2_max = y2 + 1000000.0 # Go maximal
                                continue
                            if ix1+ix3 >= Nx:
                                x2_max = x2 + 1000000.0 # Go maximal
                                continue
                            # x border
                            val = deformx_[iz1+iz3,iy1+iy3,ix1+ix3]
                            x2_min = dmin(x2_min, val)
                            x2_max = dmax(x2_max, val)
                            # y border
                            val = deformy_[iz1+iz3,iy1+iy3,ix1+ix3]
                            y2_min = dmin(y2_min, val)
                            y2_max = dmax(y2_max, val)
                            # z border
                            val = deformz_[iz1+iz3,iy1+iy3,ix1+ix3]
                            z2_min = dmin(z2_min, val)
                            z2_max = dmax(z2_max, val)
                
                # Limit to bounds and make integer
                x2_min = dmax(0, dmin(Nx-1, x2_min ))
                x2_max = dmax(0, dmin(Nx-1, x2_max ))
                y2_min = dmax(0, dmin(Ny-1, y2_min ))
                y2_max = dmax(0, dmin(Ny-1, y2_max ))
                z2_min = dmax(0, dmin(Nz-1, z2_min ))
                z2_max = dmax(0, dmin(Nz-1, z2_max ))
                #
                ix2_min = imax(0, floor(x2_min) )   # use max/min again to be sure
                ix2_max = imin(Nx-1, ceil(x2_max) )
                iy2_min = imax(0, floor(y2_min) )
                iy2_max = imin(Ny-1, ceil(y2_max) )
                iz2_min = imax(0, floor(z2_min) )
                iz2_max = imin(Nz-1, ceil(z2_max) )
                
                # Calculate max range to determine kernel size
                rangeMax = 0.1
                rangeMax = dmax( rangeMax, dmax(dabs(z2_min-z2), dabs(z2_max-z2)) )
                rangeMax = dmax( rangeMax, dmax(dabs(y2_min-y2), dabs(y2_max-y2)) )
                rangeMax = dmax( rangeMax, dmax(dabs(x2_min-x2), dabs(x2_max-x2)) )
                rangeMax = 1.0 / rangeMax # pre-divide
                
                # Sample value
                val = data_[iz1,iy1,ix1]
                
                # Splat value in destination
                for iz2 in range(iz2_min, iz2_max+1):
                    for iy2 in range(iy2_min, iy2_max+1):
                        for ix2 in range(ix2_min, ix2_max+1):
                            
                            # Calculate weights and make sure theyre > 0
                            wz = 1.0 - rangeMax * dabs( <double>iz2 - z2)
                            wy = 1.0 - rangeMax * dabs( <double>iy2 - y2)
                            wx = 1.0 - rangeMax * dabs( <double>ix2 - x2)
                            w = dmax(0.0, wz) * dmax(0.0, wy) * dmax(0.0, wx)
                            
                            # Assign values
                            result1_[iz2, iy2, ix2] += val * w
                            coeff1_[ iz2, iy2, ix2] +=       w
    
    
    # Divide by the coeffeicients
    for iz2 in range(Nz):
        for iy2 in range(Ny):
            for ix2 in range(Nx):
                
                c = coeff1_[iz2,iy2,ix2]
                if c>0:
                    result1_[iz2,iy2,ix2] = result1_[iz2,iy2,ix2] / c
    
    # Done
    return result1_


## Intt16 backward and forward

# 
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def interp1_i16(data, samplesx, order, spline_type=0.0):
#     """ Interpolation of 1D base data of i16bit floats
#     """
#     
#     # Create result array with the shape of the samples
#     result = np.empty((samplesx.size,), dtype=FLOATi16)
#     
#     # Typecast (flatten samples)
#     cdef np.ndarray[FLOATi16_T, ndim=1] data_ = data
#     cdef np.ndarray[FLOATi16_T, ndim=1] result_ = result
#     cdef np.ndarray[SAMPLE_T, ndim=1] samplesx_ = samplesx.ravel()
#     
#     # Create and init cubic interpolator
#     cdef CoefLut lut
#     cdef AccurateCoef coeffx
#     cdef double *ccx
#     cdef double splineId = 0.0
#     if order > 1:
#         lut = CoefLut.get_lut(spline_type)
#         splineId = lut.spline_type_to_id(spline_type)
#         coeffx = AccurateCoef(lut)
#     
#     # Prepare sample location variables
#     cdef double dx, tx, tx_
#     cdef int ix
#     cdef int cx, cx1, cx2
#     
#     # Prepare indices and bounds, etc
#     cdef int i
#     cdef int Ni = samplesx.shape[0]
#     cdef int Nx = data.shape[0]
#     cdef double val
#     
#     
#     if order == 3:
#         
#         # Iterate over all samples
#         for i in range(0, Ni):
#             
#             # Get integer sample location and t-factor
#             dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
#             
#             if ix >= 1 and ix < Nx-2:
#                 # Cubic interpolation
#                 ccx = coeffx.get_coef(tx)
#                 val =  data_[ix-1] * ccx[0]
#                 val += data_[ix  ] * ccx[1]
#                 val += data_[ix+1] * ccx[2]
#                 val += data_[ix+2] * ccx[3]
#                 result_[i] = val
#             
#             elif dx>=-0.5 and dx<=Nx-0.5:
#                 # Edge effects
#                 
#                 # Get coefficients
#                 ccx = coeffx.get_coef(tx)
#                 
#                 # Correct stuff: calculate offset (max 2)
#                 cx1, cx2 = 0, 4
#                 #
#                 if ix<1: cx1+=1-ix;
#                 if ix>Nx-3: cx2+=(Nx-3)-ix;
#                 
#                 # Correct coefficients, so that the sum is one
#                 val = 0.0
#                 for cx in range(cx1, cx2):  val += ccx[cx]
#                 val = 1.0/val
#                 for cx in range(cx1, cx2):  ccx[cx] *= val
#                 
#                 # Combine elements
#                 val = 0.0
#                 for cx in range(cx1, cx2):
#                     val += data_[ix+cx-1] * ccx[cx]
#                 result_[i] = val
#                 
#                 # Code below produces nice results too, but can only be 
#                 # used for 1D; higher dimensions get weird results
#                 
# #                 # Correct indices
# #                 if ix<1: tx+=ix-1; ix=1; 
# #                 if ix>Nx-3: tx+=ix-(Nx-3); ix=Nx-3;
# #                 
# #                 # Get coefficients (first fill, so we can use its memory)
# #                 ccx = coeffx.get_coef(0.0)
# #                 if splineId < 2.0:
# #                     lut.cubicsplinecoef_cardinal_edge(tx, ccx)
# #                 else:
# #                     lut.cubicsplinecoef(splineId, tx, ccx)
# #                 
# #                 # Combine elements
# #                 val = 0.0
# #                 for cx in range(0,4):
# #                     cx2 = ix + cx - 1 
# #                     val += data_[cx2] * ccx[cx]
# #                 result_[i] = val
#             
#             else:
#                 # Out of range
#                 result_[i] = 0.0
#     
#     elif order == 2:
#         
#         # Iterate over all samples
#         for i in range(0, Ni):
#             
#             # Get integer sample location and t-factor
#             dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
#             
#             if ix >= 0 and ix < Nx-1:
#                 # Quasi-linear interpolation
#                 tx_ = -2*tx**3 + 3*tx**2
#                 val =  data_[ix] * (1.0-tx_)
#                 val += data_[ix+1] * tx_
#                 result_[i] = val
#             elif dx>=-0.5 and dx<=Nx-0.5:                
#                 if ix<0: tx+=ix; ix=0; 
#                 if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
#                 # Quasi-linear interpolation (edges)
#                 tx_ = -2*tx**3 + 3*tx**2
#                 val =  data_[ix] * (1.0-tx_)
#                 val += data_[ix+1] * tx_
#                 result_[i] = val
#             else:
#                 # Out of range
#                 result_[i] = 0.0
#     
#     elif order == 1:
#         
#         # Iterate over all samples
#         for i in range(0, Ni):
#             
#             # Get integer sample location and t-factor
#             dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
#             
#             if ix >= 0 and ix < Nx-1:
#                 # Linear interpolation
#                 val =  data_[ix] * (1.0-tx)
#                 val += data_[ix+1] * tx
#                 result_[i] = val
#             elif dx>=-0.5 and dx<=Nx-0.5:
#                 if ix<0: tx+=ix; ix=0; 
#                 if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
#                 # Linear interpolation (edges)
#                 val =  data_[ix] * (1.0-tx)
#                 val += data_[ix+1] * tx
#                 result_[i] = val
#             else:
#                 # Out of range
#                 result_[i] = 0.0
#     
#     else:
#         
#         # Iterate over all samples
#         for i in range(0, Ni):
#             
#             # Get integer sample location
#             dx = samplesx_[i]; ix = floor(dx+0.5)
#             
#             if ix >= 0 and ix < Nx:
#                 # Nearest neighbour interpolation
#                 result_[i] = data_[ix]
#             else:
#                 # Out of range
#                 result_[i] = 0.0
#     
#     # Done
#     result.shape = samplesx.shape
#     return result
# 
# 
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def interp2_i16(data, samplesx, samplesy, order, spline_type=0.0):
#     """ Interpolation of 2D base data of i16bit floats
#     """    
#     
#     # Create result array with the shape of the samples
#     result = np.empty((samplesx.size,), dtype=FLOATi16)
#     
#     # Flatten and type the samples and result
#     cdef np.ndarray[FLOATi16_T, ndim=2] data_ = data
#     cdef np.ndarray[FLOATi16_T, ndim=1] result_ = result
#     cdef np.ndarray[SAMPLE_T, ndim=1] samplesx_ = samplesx.ravel()
#     cdef np.ndarray[SAMPLE_T, ndim=1] samplesy_ = samplesy.ravel()
#     
#     # Create and init cubic interpolator
#     cdef CoefLut lut
#     cdef AccurateCoef coeffx, coeffy
#     cdef double splineId = 0.0
#     cdef double *ccx, *ccy
#     if order > 1:
#         lut = CoefLut.get_lut(spline_type)
#         splineId = lut.spline_type_to_id(spline_type)
#         coeffx = AccurateCoef(lut)
#         coeffy = AccurateCoef(lut)
#     
#     # Prepare sample location variables
#     cdef double dx, tx, tx_, dy, ty, ty_
#     cdef int ix, iy
#     cdef int cx, cy
#     cdef int cx1, cx2, cy1, cy2
#     cdef double valFactor
#     
#     # Prepare indices and bounds, etc
#     cdef int i
#     cdef int Ni = samplesx.size
#     cdef int Ny = data.shape[0]
#     cdef int Nx = data.shape[1]
#     cdef double val
#     
#     
#     if order == 3:
#        
#         # with nogil: does not make it faster
#         # Iterate over all samples
#         for i in range(0, Ni):
#             
#             # Get integer sample location and t-factor
#             dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
#             dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
#             
#             if (    ix >= 1 and ix < Nx-2 and 
#                     iy >= 1 and iy < Ny-2       ):
#                 # Cubic interpolation
#                 ccx = coeffx.get_coef(tx)
#                 ccy = coeffy.get_coef(ty)
#                 val = 0.0
#                 for cy in range(4):
#                     for cx in range(4):
#                         val += data_[iy+cy-1,ix+cx-1] * ccy[cy] * ccx[cx]
#                 result_[i] = val
#             
#             elif (  dx>=-0.5 and dx<=Nx-0.5 and 
#                     dy>=-0.5 and dy<=Ny-0.5     ):
#                 # Edge effects
#                 
#                 # Get coefficients
#                 ccx = coeffx.get_coef(tx)
#                 ccy = coeffy.get_coef(ty)
#                 
#                 # Correct stuff: calculate offset (max 2)
#                 cx1, cx2 = 0, 4
#                 cy1, cy2 = 0, 4
#                 #
#                 if ix<1: cx1+=1-ix;
#                 if ix>Nx-3: cx2+=(Nx-3)-ix;
#                 #
#                 if iy<1: cy1+=1-iy;
#                 if iy>Ny-3: cy2+=(Ny-3)-iy;
#                 
#                 # Correct coefficients, so that the sum is one
#                 val = 0.0
#                 for cx in range(cx1, cx2):  val += ccx[cx]
#                 val = 1.0/val
#                 for cx in range(cx1, cx2):  ccx[cx] *= val
#                 #
#                 val = 0.0
#                 for cy in range(cy1, cy2):  val += ccy[cy]
#                 val = 1.0/val
#                 for cy in range(cy1, cy2):  ccy[cy] *= val
#                 
#                 # Combine elements
#                 # No need to pre-calculate indices: the C compiler is well
#                 # capable of making these optimizations.
#                 val = 0.0
#                 for cy in range(cy1, cy2):
#                     for cx in range(cx1, cx2):
#                         val += data_[iy+cy-1,ix+cx-1] * ccy[cy] * ccx[cx]
#                 result_[i] = val
#             
#             else:
#                 # Out of range
#                 result_[i] = 0.0
#     
#     elif order == 2:
#         
#         # Iterate over all samples
#         for i in range(0, Ni):
#             
#             # Get integer sample location and t-factor
#             dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
#             dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
#             
#             if (    ix >= 0 and ix < Nx-1 and
#                     iy >= 0 and iy < Ny-1     ):
#                 # Quasi-linear interpolation
#                 tx_ = -2*tx**3 + 3*tx**2
#                 ty_ = -2*ty**3 + 3*ty**2
#                 val =  data_[iy,  ix  ] * (1.0-ty_) * (1.0-tx_)
#                 val += data_[iy,  ix+1] * (1.0-ty_) *      tx_
#                 val += data_[iy+1,ix  ] *      ty_  * (1.0-tx_)
#                 val += data_[iy+1,ix+1] *      ty_  *      tx_
#                 result_[i] = val
#             elif (  dx>=-0.5 and dx<=Nx-0.5 and 
#                     dy>=-0.5 and dy<=Ny-0.5     ):
#                 # Edge effects
#                 if ix<0: tx+=ix; ix=0; 
#                 if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
#                 #
#                 if iy<0: ty+=iy; iy=0; 
#                 if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
#                 # Quasi-linear interpolation (edges)
#                 tx_ = -2*tx**3 + 3*tx**2
#                 ty_ = -2*ty**3 + 3*ty**2
#                 val =  data_[iy,  ix  ] * (1.0-ty_) * (1.0-tx_)
#                 val += data_[iy,  ix+1] * (1.0-ty_) *      tx_
#                 val += data_[iy+1,ix  ] *      ty_  * (1.0-tx_)
#                 val += data_[iy+1,ix+1] *      ty_  *      tx_
#                 result_[i] = val
#             else:
#                 # Out of range
#                 result_[i] = 0.0
#     
#     elif order == 1:
#         
#         # Iterate over all samples
#         for i in range(0, Ni):
#             
#             # Get integer sample location and t-factor
#             dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
#             dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
#             
#             if (    ix >= 0 and ix < Nx-1 and
#                     iy >= 0 and iy < Ny-1     ):
#                 # Linear interpolation
#                 val =  data_[iy,  ix  ] * (1.0-ty) * (1.0-tx)
#                 val += data_[iy,  ix+1] * (1.0-ty) *      tx
#                 val += data_[iy+1,ix  ] *      ty  * (1.0-tx)
#                 val += data_[iy+1,ix+1] *      ty  *      tx
#                 result_[i] = val
#             elif (  dx>=-0.5 and dx<=Nx-0.5 and 
#                     dy>=-0.5 and dy<=Ny-0.5     ):
#                 # Edge effects
#                 if ix<0: tx+=ix; ix=0; 
#                 if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
#                 #
#                 if iy<0: ty+=iy; iy=0; 
#                 if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
#                 # Linear interpolation (edges)
#                 val =  data_[iy,  ix  ] * (1.0-ty) * (1.0-tx)
#                 val += data_[iy,  ix+1] * (1.0-ty) *      tx
#                 val += data_[iy+1,ix  ] *      ty  * (1.0-tx)
#                 val += data_[iy+1,ix+1] *      ty  *      tx
#                 result_[i] = val
#             else:
#                 # Out of range
#                 result_[i] = 0.0
#     
#     else:
#         
#         # Iterate over all samples
#         for i in range(0, Ni):
#             
#             # Get integer sample location
#             dx = samplesx_[i]; ix = floor(dx+0.5)
#             dy = samplesy_[i]; iy = floor(dy+0.5)
#             
#             if (    ix >= 0 and ix < Nx and
#                     iy >= 0 and iy < Ny     ):
#                 # Nearest neighbour interpolation
#                 result_[i] = data_[iy,ix]
#             else:
#                 # Out of range
#                 result_[i] = 0.0
#     
#     # Done
#     result.shape = samplesx.shape
#     return result
# 
# 
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def interp3_i16(data, samplesx, samplesy, samplesz, order, spline_type=0.0):
#     """ Interpolation of 3D base data of i16bit floats
#     """    
#     
#     # Create result array with the shape of the samples
#     result = np.empty((samplesx.size,), dtype=FLOATi16)
#     
#     # Flatten and type the samples and result
#     cdef np.ndarray[FLOATi16_T, ndim=3] data_ = data
#     cdef np.ndarray[FLOATi16_T, ndim=1] result_ = result
#     cdef np.ndarray[SAMPLE_T, ndim=1] samplesx_ = samplesx.ravel()
#     cdef np.ndarray[SAMPLE_T, ndim=1] samplesy_ = samplesy.ravel()
#     cdef np.ndarray[SAMPLE_T, ndim=1] samplesz_ = samplesz.ravel()
#     
#     # Create and init cubic interpolator
#     cdef CoefLut lut
#     cdef AccurateCoef coeffx, coeffy, coeffz
#     cdef double splineId = 0.0
#     cdef double *ccx, *ccy, *ccz
#     if order > 1:
#         lut = CoefLut.get_lut(spline_type)
#         splineId = lut.spline_type_to_id(spline_type)
#         coeffx = AccurateCoef(lut)
#         coeffy = AccurateCoef(lut)
#         coeffz = AccurateCoef(lut)
#     
#     # Prepare sample location variables
#     cdef double dx, tx, tx_, dy, ty, ty_, dz, tz, tz_
#     cdef int ix, iy, iz
#     cdef int cx, cy, cz
#     cdef int cx1, cx2, cy1, cy2, cz1, cz2
#     cdef double valFactor
#     
#     # Prepare indices and bounds, etc
#     cdef int i
#     cdef int Ni = samplesx.size
#     cdef int Nz = data.shape[0]
#     cdef int Ny = data.shape[1]
#     cdef int Nx = data.shape[2]
#     cdef double val
#     
#     
#     if order == 3:
#         
#         # Iterate over all samples
#         for i in range(0, Ni):
#             
#             # Get integer sample location and t-factor
#             dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
#             dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
#             dz = samplesz_[i]; iz = floor(dz); tz = dz-iz
#             
#             if (    ix >= 1 and ix < Nx-2 and 
#                     iy >= 1 and iy < Ny-2 and
#                     iz >= 1 and iz < Nz-2       ):
#                 # Cubic interpolation
#                 ccx = coeffx.get_coef(tx)
#                 ccy = coeffy.get_coef(ty)
#                 ccz = coeffz.get_coef(tz)
#                 val = 0.0
#                 for cz in range(4):
#                     for cy in range(4):
#                         for cx in range(4):
#                             val += data_[iz+cz-1,iy+cy-1,ix+cx-1] * (
#                                             ccz[cz] * ccy[cy] * ccx[cx] )
#                 result_[i] = val
#             
#             elif (  dx>=-0.5 and dx<=Nx-0.5 and 
#                     dy>=-0.5 and dy<=Ny-0.5 and
#                     dz>=-0.5 and dz<=Nz-0.5     ):
#                 # Edge effects
#                 
#                 # Get coefficients
#                 ccx = coeffx.get_coef(tx)
#                 ccy = coeffy.get_coef(ty)
#                 ccz = coeffz.get_coef(tz)
#                 
#                 # Correct stuff: calculate offset (max 2)
#                 cx1, cx2 = 0, 4
#                 cy1, cy2 = 0, 4
#                 cz1, cz2 = 0, 4
#                 #
#                 if ix<1: cx1+=1-ix;
#                 if ix>Nx-3: cx2+=(Nx-3)-ix;
#                 #
#                 if iy<1: cy1+=1-iy;
#                 if iy>Ny-3: cy2+=(Ny-3)-iy;
#                 #
#                 if iz<1: cz1+=1-iz;
#                 if iz>Nz-3: cz2+=(Nz-3)-iz;
#                 
#                 # Correct coefficients, so that the sum is one
#                 val = 0.0
#                 for cx in range(cx1, cx2):  val += ccx[cx]
#                 val = 1.0/val
#                 for cx in range(cx1, cx2):  ccx[cx] *= val
#                 #
#                 val = 0.0
#                 for cy in range(cy1, cy2):  val += ccy[cy]
#                 val = 1.0/val
#                 for cy in range(cy1, cy2):  ccy[cy] *= val
#                 #
#                 val = 0.0
#                 for cz in range(cz1, cz2):  val += ccz[cz]
#                 val = 1.0/val
#                 for cz in range(cz1, cz2):  ccz[cz] *= val
#                 
#                 # Combine elements 
#                 # No need to pre-calculate indices: the C compiler is well
#                 # capable of making these optimizations.
#                 val = 0.0
#                 for cz in range(cz1, cz2):
#                     for cy in range(cy1, cy2):
#                         for cx in range(cx1, cx2):
#                             val += data_[iz+cz-1,iy+cy-1,ix+cx-1] * ccz[cz] * ccy[cy] * ccx[cx]
#                 result_[i] = val
#             
#             else:
#                 # Out of range
#                 result_[i] = 0.0
#     
#     elif order == 2:
#         
#         # Iterate over all samples
#         for i in range(0, Ni):
#             
#             # Get integer sample location and t-factor
#             dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
#             dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
#             dz = samplesz_[i]; iz = floor(dz); tz = dz-iz
#             
#             if (    ix >= 0 and ix < Nx-1 and
#                     iy >= 0 and iy < Ny-1 and
#                     iz >= 0 and iz < Nz-1       ):
#                 # Quasi-linear interpolation
#                 tx_ = -2*tx**3 + 3*tx**2
#                 ty_ = -2*ty**3 + 3*ty**2
#                 tz_ = -2*tz**3 + 3*tz**2
#                 #
#                 val =  data_[iz  ,iy,  ix  ] * (1.0-tz_) * (1.0-ty_) *(1.0-tx_)
#                 val += data_[iz  ,iy,  ix+1] * (1.0-tz_) * (1.0-ty_) *     tx_
#                 val += data_[iz  ,iy+1,ix  ] * (1.0-tz_) *      ty_  *(1.0-tx_)
#                 val += data_[iz  ,iy+1,ix+1] * (1.0-tz_) *      ty_  *     tx_
#                 #
#                 val += data_[iz+1,iy,  ix  ] *      tz_  * (1.0-ty_) *(1.0-tx_)
#                 val += data_[iz+1,iy,  ix+1] *      tz_  * (1.0-ty_) *     tx_
#                 val += data_[iz+1,iy+1,ix  ] *      tz_  *      ty_  *(1.0-tx_)
#                 val += data_[iz+1,iy+1,ix+1] *      tz_  *      ty_  *     tx_
#                 result_[i] = val
#             
#             elif (  dx>=-0.5 and dx<=Nx-0.5 and 
#                     dy>=-0.5 and dy<=Ny-0.5 and
#                     dz>=-0.5 and dz<=Nz-0.5    ):
#                 # Edge effects
#                 if ix<0: tx+=ix; ix=0; 
#                 if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
#                 #
#                 if iy<0: ty+=iy; iy=0; 
#                 if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
#                 #
#                 if iz<0: tz+=iz; iz=0; 
#                 if iz>Nz-2: tz+=iz-(Nz-2); iz=Nz-2; 
#                 # Quasi-linear interpolation (edges)
#                 tx_ = -2*tx**3 + 3*tx**2
#                 ty_ = -2*ty**3 + 3*ty**2
#                 tz_ = -2*tz**3 + 3*tz**2
#                 #
#                 val =  data_[iz  ,iy,  ix  ] * (1.0-tz_) * (1.0-ty_) *(1.0-tx_)
#                 val += data_[iz  ,iy,  ix+1] * (1.0-tz_) * (1.0-ty_) *     tx_
#                 val += data_[iz  ,iy+1,ix  ] * (1.0-tz_) *      ty_  *(1.0-tx_)
#                 val += data_[iz  ,iy+1,ix+1] * (1.0-tz_) *      ty_  *     tx_
#                 #
#                 val += data_[iz+1,iy,  ix  ] *      tz_  * (1.0-ty_) *(1.0-tx_)
#                 val += data_[iz+1,iy,  ix+1] *      tz_  * (1.0-ty_) *     tx_
#                 val += data_[iz+1,iy+1,ix  ] *      tz_  *      ty_  *(1.0-tx_)
#                 val += data_[iz+1,iy+1,ix+1] *      tz_  *      ty_  *     tx_
#                 result_[i] = val
#             
#             else:
#                 # Out of range
#                 result_[i] = 0.0
#     
#     elif order == 1:
#         
#         # Iterate over all samples
#         for i in range(0, Ni):
#             
#             # Get integer sample location and t-factor
#             dx = samplesx_[i]; ix = floor(dx); tx = dx-ix
#             dy = samplesy_[i]; iy = floor(dy); ty = dy-iy
#             dz = samplesz_[i]; iz = floor(dz); tz = dz-iz
#             
#             if (    ix >= 0 and ix < Nx-1 and
#                     iy >= 0 and iy < Ny-1 and
#                     iz >= 0 and iz < Nz-1       ):
#                 # Linear interpolation
#                 val =  data_[iz  ,iy,  ix  ] * (1.0-tz) * (1.0-ty) * (1.0-tx)
#                 val += data_[iz  ,iy,  ix+1] * (1.0-tz) * (1.0-ty) *      tx
#                 val += data_[iz  ,iy+1,ix  ] * (1.0-tz) *      ty  * (1.0-tx)
#                 val += data_[iz  ,iy+1,ix+1] * (1.0-tz) *      ty  *      tx
#                 #
#                 val += data_[iz+1,iy,  ix  ] *      tz  * (1.0-ty) * (1.0-tx)
#                 val += data_[iz+1,iy,  ix+1] *      tz  * (1.0-ty) *      tx
#                 val += data_[iz+1,iy+1,ix  ] *      tz  *      ty  * (1.0-tx)
#                 val += data_[iz+1,iy+1,ix+1] *      tz  *      ty  *      tx
#                 result_[i] = val
#             elif (  dx>=-0.5 and dx<=Nx-0.5 and 
#                     dy>=-0.5 and dy<=Ny-0.5 and
#                     dz>=-0.5 and dz<=Nz-0.5    ):
#                 # Edge effects
#                 if ix<0: tx+=ix; ix=0; 
#                 if ix>Nx-2: tx+=ix-(Nx-2); ix=Nx-2; 
#                 #
#                 if iy<0: ty+=iy; iy=0; 
#                 if iy>Ny-2: ty+=iy-(Ny-2); iy=Ny-2; 
#                 #
#                 if iz<0: tz+=iz; iz=0; 
#                 if iz>Nz-2: tz+=iz-(Nz-2); iz=Nz-2; 
#                 # Linear interpolation (edges)
#                 val =  data_[iz  ,iy,  ix  ] * (1.0-tz) * (1.0-ty) * (1.0-tx)
#                 val += data_[iz  ,iy,  ix+1] * (1.0-tz) * (1.0-ty) *      tx
#                 val += data_[iz  ,iy+1,ix  ] * (1.0-tz) *      ty  * (1.0-tx)
#                 val += data_[iz  ,iy+1,ix+1] * (1.0-tz) *      ty  *      tx
#                 #
#                 val += data_[iz+1,iy,  ix  ] *      tz  * (1.0-ty) * (1.0-tx)
#                 val += data_[iz+1,iy,  ix+1] *      tz  * (1.0-ty) *      tx
#                 val += data_[iz+1,iy+1,ix  ] *      tz  *      ty  * (1.0-tx)
#                 val += data_[iz+1,iy+1,ix+1] *      tz  *      ty  *      tx
#                 result_[i] = val
#             else:
#                 # Out of range
#                 result_[i] = 0.0
#     
#     else:
#         
#         # Iterate over all samples
#         for i in range(0, Ni):
#             
#             # Get integer sample location
#             dx = samplesx_[i]; ix = floor(dx+0.5)
#             dy = samplesy_[i]; iy = floor(dy+0.5)
#             dz = samplesz_[i]; iz = floor(dz+0.5)
#             
#             if (    ix >= 0 and ix < Nx and
#                     iy >= 0 and iy < Ny and
#                     iz >= 0 and iz < Nz     ):
#                 # Nearest neighbour interpolation
#                 result_[i] = data_[iz,iy,ix]
#             else:
#                 # Out of range
#                 result_[i] = 0.0
#     
#     # Done
#     result.shape = samplesx.shape
#     return result
# 
# 
# 
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def project1_i16(data, deformx):
#     """ Forward deformation of 1D base data of i16bit floats.
#     """
#     
#     # Create result array with the shape of the samples
#     result1 = np.zeros(data.shape, dtype=FLOATi16)
#     
#     # Create coefficient array to be able to test coverage 
#     coeff1 = np.zeros(data.shape, dtype=FLOATi16)
#     
#     # Typecast, activate result1 as the result
#     cdef np.ndarray[FLOATi16_T, ndim=1] result1_ = result1
#     cdef np.ndarray[FLOATi16_T, ndim=1] coeff1_ = coeff1
#     cdef np.ndarray[FLOATi16_T, ndim=1] data_ = data
#     cdef np.ndarray[SAMPLE_T, ndim=1] deformx_ = deformx
#     
#     # Prepare sample location variables
#     cdef int iz1, iy1, ix1 # integer pixel locations in source
#     cdef int iz2, iy2, ix2 # integer pixel locations in dest
#     cdef int iz3, iy3, ix3 # integer sub-locations in source
#     cdef double z1, y1, x1 # full pixel locations in source
#     cdef double z2, y2, x2 # full pixel locations in dest
#     
#     # For the bounding box
#     cdef double z2_min, z2_max, y2_min, y2_max, x2_min, x2_max
#     cdef int  iz2_min, iz2_max, iy2_min, iy2_max, ix2_min, ix2_max
#     
#     # More ...
#     cdef double wx, wy, wz, w  # Weights
#     cdef double rangeMax  # max range to determine kernel size
#     cdef double val, c # For storing the temporary values
#         
#     # Get bounds
#     cdef int Nz, Ny, Nx
#     Nx = data.shape[0]
#     
#     
#     for ix1 in range(0, Nx):
#         
#         # Calculate location to map to
#         x2 = deformx_[ix1]
#         
#         # Select where the surrounding pixels map to.
#         # This defines the region that we should fill in. This region
#         # is overestimated as it is assumed rectangular in the destination,
#         # which is not true in general.
#         x2_min = x2
#         x2_max = x2        
#         for ix3 in range(-1,2):
#             if ix3==0:
#                 continue
#             if ix1+ix3 < 0:
#                 x2_min = x2 - 1000000.0 # Go minimal
#                 continue
#             if ix1+ix3 >= Nx:
#                 x2_max = x2 + 1000000.0 # Go maximal
#                 continue
#             # x border
#             val = deformx_[ix1+ix3]
#             x2_min = dmin(x2_min, val)
#             x2_max = dmax(x2_max, val)
#         
#         
#         # Limit to bounds and make integer
#         x2_min = dmax(0, dmin(Nx-1, x2_min ))
#         x2_max = dmax(0, dmin(Nx-1, x2_max ))
#         #
#         ix2_min = imax(0, floor(x2_min) )   # use max/min again to be sure
#         ix2_max = imin(Nx-1, ceil(x2_max) )
#         
#         # Calculate max range to determine kernel size
#         rangeMax = 0.1
#         rangeMax = dmax( rangeMax, dmax(dabs(x2_min-x2), dabs(x2_max-x2)) )
#         rangeMax = 1.0 / rangeMax # pre-divide
#         
#         # Sample value
#         val = data_[ix1]
#         
#         # Splat value in destination
#         for ix2 in range(ix2_min, ix2_max+1):
#             
#             # Calculate weights and make sure theyre > 0
#             wx = 1.0 - rangeMax * dabs( <double>ix2 - x2)
#             w = dmax(0.0, wx)
#             
#             # Assign values
#             result1_[ix2  ] += val * w
#             coeff1_[ ix2  ] +=       w
#     
#     
#     # Divide by the coeffeicients
#     for ix2 in range(Nx):
#         
#         c = coeff1_[ix2]
#         if c>0:
#             result1_[ix2] = result1_[ix2] / c
#     
#     # Done
#     return result1_
# 
# 
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def project2_i16(data, deformx, deformy):
#     """ Forward deformation of 2D base data of i16bit floats.
#     """
#     
#     # Create result array with the shape of the samples
#     result1 = np.zeros(data.shape, dtype=FLOATi16)
#     
#     # Create coefficient array to be able to test coverage 
#     coeff1 = np.zeros(data.shape, dtype=FLOATi16)
#     
#     # Typecast, activate result1 as the result
#     cdef np.ndarray[FLOATi16_T, ndim=2] result1_ = result1
#     cdef np.ndarray[FLOATi16_T, ndim=2] coeff1_ = coeff1
#     cdef np.ndarray[FLOATi16_T, ndim=2] data_ = data
#     cdef np.ndarray[SAMPLE_T, ndim=2] deformx_ = deformx
#     cdef np.ndarray[SAMPLE_T, ndim=2] deformy_ = deformy
#     
#     # Prepare sample location variables
#     cdef int iz1, iy1, ix1 # integer pixel locations in source
#     cdef int iz2, iy2, ix2 # integer pixel locations in dest
#     cdef int iz3, iy3, ix3 # integer sub-locations in source
#     cdef double z1, y1, x1 # full pixel locations in source
#     cdef double z2, y2, x2 # full pixel locations in dest
#     
#     # For the bounding box
#     cdef double z2_min, z2_max, y2_min, y2_max, x2_min, x2_max
#     cdef int  iz2_min, iz2_max, iy2_min, iy2_max, ix2_min, ix2_max
#     
#     # More ...
#     cdef double wx, wy, wz, w  # Weights
#     cdef double rangeMax  # max range to determine kernel size
#     cdef double val, c # For storing the temporary values
#         
#     # Get bounds
#     cdef int Nz, Ny, Nx
#     Ny = data.shape[0]
#     Nx = data.shape[1]
#     
#     cdef double rangeMax_sum = 0.0
#     
#     for iy1 in range(0, Ny):
#         for ix1 in range(0, Nx):
#             
#             # Calculate location to map to
#             y2 = deformy_[iy1,ix1]
#             x2 = deformx_[iy1,ix1]
#             
#             # Select where the surrounding pixels map to.
#             # This defines the region that we should fill in. This region
#             # is overestimated as it is assumed rectangular in the destination,
#             # which is not true in general.
#             x2_min = x2
#             x2_max = x2
#             y2_min = y2
#             y2_max = y2
#             for iy3 in range(-1,2):
#                 for ix3 in range(-1,2):
#                     if iy3*ix3==0:
#                         continue
#                     if iy1+iy3 < 0:
#                         y2_min = y2 - 1000000.0 # Go minimal
#                         continue
#                     if ix1+ix3 < 0:
#                         x2_min = x2 - 1000000.0 # Go minimal
#                         continue
#                     if iy1+iy3 >= Ny:
#                         y2_max = y2 + 1000000.0 # Go maximal
#                         continue
#                     if ix1+ix3 >= Nx:
#                         x2_max = x2 + 1000000.0 # Go maximal
#                         continue
#                     # x border
#                     val = deformx_[iy1+iy3,ix1+ix3]
#                     x2_min = dmin(x2_min, val)
#                     x2_max = dmax(x2_max, val)
#                     # y border
#                     val = deformy_[iy1+iy3,ix1+ix3]
#                     y2_min = dmin(y2_min, val)
#                     y2_max = dmax(y2_max, val)
#             
#             # Limit to bounds and make integer
#             x2_min = dmax(0, dmin(Nx-1, x2_min ))
#             x2_max = dmax(0, dmin(Nx-1, x2_max ))
#             y2_min = dmax(0, dmin(Ny-1, y2_min ))
#             y2_max = dmax(0, dmin(Ny-1, y2_max ))
#             #
#             ix2_min = imax(0, floor(x2_min) )   # use max/min again to be sure
#             ix2_max = imin(Nx-1, ceil(x2_max) )
#             iy2_min = imax(0, floor(y2_min) )
#             iy2_max = imin(Ny-1, ceil(y2_max) )
#             
#             # Calculate max range to determine kernel size
#             rangeMax = 0.1
#             rangeMax = dmax( rangeMax, dmax(dabs(y2_min-y2), dabs(y2_max-y2)) )
#             rangeMax = dmax( rangeMax, dmax(dabs(x2_min-x2), dabs(x2_max-x2)) )
#             
# #             rangeMax_sum += rangeMax
#             rangeMax = 1.0 / rangeMax # pre-divide
#             
#             # Sample value
#             val = data_[iy1,ix1]
#             
#             # Splat value in destination
#             for iy2 in range(iy2_min, iy2_max+1):
#                 for ix2 in range(ix2_min, ix2_max+1):
#                     
#                     # Calculate weights and make sure theyre > 0
#                     wy = 1.0 - rangeMax * dabs( <double>iy2 - y2)
#                     wx = 1.0 - rangeMax * dabs( <double>ix2 - x2)
#                     w = dmax(0.0, wy) * dmax(0.0, wx)
#                     
#                     # Assign values
#                     result1_[iy2  ,ix2  ] += val * w
#                     coeff1_[ iy2  ,ix2  ] +=       w
#     
#     
# #     print rangeMax_sum / (Nx*Ny)
#     
#     # Divide by the coeffeicients
#     for iy2 in range(Ny):
#         for ix2 in range(Nx):
#             
#             c = coeff1_[iy2,ix2]
#             if c>0:
#                 result1_[iy2,ix2] /= c
#     
#     # Done
#     return result1_
# 
# 
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def project3_i16(data, deformx, deformy, deformz):
#     """ Forward deformation of 3D base data of i16bit floats.
#     """
#     
#     # Create result array with the shape of the samples
#     result1 = np.zeros(data.shape, dtype=FLOATi16)
#     
#     # Create coefficient array to be able to test coverage 
#     coeff1 = np.zeros(data.shape, dtype=FLOATi16)
#     
#     # Typecast, activate result1 as the result
#     cdef np.ndarray[FLOATi16_T, ndim=3] result1_ = result1
#     cdef np.ndarray[FLOATi16_T, ndim=3] coeff1_ = coeff1
#     cdef np.ndarray[FLOATi16_T, ndim=3] data_ = data
#     cdef np.ndarray[SAMPLE_T, ndim=3] deformx_ = deformx
#     cdef np.ndarray[SAMPLE_T, ndim=3] deformy_ = deformy
#     cdef np.ndarray[SAMPLE_T, ndim=3] deformz_ = deformz
#     
#     # Prepare sample location variables
#     cdef int iz1, iy1, ix1 # integer pixel locations in source
#     cdef int iz2, iy2, ix2 # integer pixel locations in dest
#     cdef int iz3, iy3, ix3 # integer sub-locations in source
#     cdef double z1, y1, x1 # full pixel locations in source
#     cdef double z2, y2, x2 # full pixel locations in dest
#     
#     # For the bounding box
#     cdef double z2_min, z2_max, y2_min, y2_max, x2_min, x2_max
#     cdef int  iz2_min, iz2_max, iy2_min, iy2_max, ix2_min, ix2_max
#     
#     # More ...
#     cdef double wx, wy, wz, w  # Weights
#     cdef double rangeMax  # max range to determine kernel size
#     cdef double val, c # For storing the temporary values
#         
#     # Get bounds
#     cdef int Nz, Ny, Nx
#     Nz = data.shape[0]
#     Ny = data.shape[1]
#     Nx = data.shape[2]
#     
#     
#     for iz1 in range(0, Nz):
#         for iy1 in range(0, Ny):
#             for ix1 in range(0, Nx):
#                 
#                 # Calculate location to map to
#                 z2 = deformz_[iz1,iy1,ix1]
#                 y2 = deformy_[iz1,iy1,ix1]
#                 x2 = deformx_[iz1,iy1,ix1]
#                 
#                 # Select where the surrounding pixels map to.
#                 # This defines the region that we should fill in. This region
#                 # is overestimated as it is assumed rectangular in the destination,
#                 # which is not true in general.
#                 x2_min = x2
#                 x2_max = x2
#                 y2_min = y2
#                 y2_max = y2
#                 z2_min = z2
#                 z2_max = z2
#                 for iz3 in range(-1,2):
#                     for iy3 in range(-1,2):
#                         for ix3 in range(-1,2):
#                             if iz3*iy3*ix3==0:
#                                 continue
#                             if iz1+iz3 < 0:
#                                 z2_min = z2 - 1000000.0 # Go minimal
#                                 continue
#                             if iy1+iy3 < 0:
#                                 y2_min = y2 - 1000000.0 # Go minimal
#                                 continue
#                             if ix1+ix3 < 0:
#                                 x2_min = x2 - 1000000.0 # Go minimal
#                                 continue
#                             if iz1+iz3 >= Nz:
#                                 z2_max = z2 + 1000000.0 # Go maximal
#                                 continue
#                             if iy1+iy3 >= Ny:
#                                 y2_max = y2 + 1000000.0 # Go maximal
#                                 continue
#                             if ix1+ix3 >= Nx:
#                                 x2_max = x2 + 1000000.0 # Go maximal
#                                 continue
#                             # x border
#                             val = deformx_[iz1+iz3,iy1+iy3,ix1+ix3]
#                             x2_min = dmin(x2_min, val)
#                             x2_max = dmax(x2_max, val)
#                             # y border
#                             val = deformy_[iz1+iz3,iy1+iy3,ix1+ix3]
#                             y2_min = dmin(y2_min, val)
#                             y2_max = dmax(y2_max, val)
#                             # z border
#                             val = deformz_[iz1+iz3,iy1+iy3,ix1+ix3]
#                             z2_min = dmin(z2_min, val)
#                             z2_max = dmax(z2_max, val)
#                 
#                 # Limit to bounds and make integer
#                 x2_min = dmax(0, dmin(Nx-1, x2_min ))
#                 x2_max = dmax(0, dmin(Nx-1, x2_max ))
#                 y2_min = dmax(0, dmin(Ny-1, y2_min ))
#                 y2_max = dmax(0, dmin(Ny-1, y2_max ))
#                 z2_min = dmax(0, dmin(Nz-1, z2_min ))
#                 z2_max = dmax(0, dmin(Nz-1, z2_max ))
#                 #
#                 ix2_min = imax(0, floor(x2_min) )   # use max/min again to be sure
#                 ix2_max = imin(Nx-1, ceil(x2_max) )
#                 iy2_min = imax(0, floor(y2_min) )
#                 iy2_max = imin(Ny-1, ceil(y2_max) )
#                 iz2_min = imax(0, floor(z2_min) )
#                 iz2_max = imin(Nz-1, ceil(z2_max) )
#                 
#                 # Calculate max range to determine kernel size
#                 rangeMax = 0.1
#                 rangeMax = dmax( rangeMax, dmax(dabs(z2_min-z2), dabs(z2_max-z2)) )
#                 rangeMax = dmax( rangeMax, dmax(dabs(y2_min-y2), dabs(y2_max-y2)) )
#                 rangeMax = dmax( rangeMax, dmax(dabs(x2_min-x2), dabs(x2_max-x2)) )
#                 rangeMax = 1.0 / rangeMax # pre-divide
#                 
#                 # Sample value
#                 val = data_[iz1,iy1,ix1]
#                 
#                 # Splat value in destination
#                 for iz2 in range(iz2_min, iz2_max+1):
#                     for iy2 in range(iy2_min, iy2_max+1):
#                         for ix2 in range(ix2_min, ix2_max+1):
#                             
#                             # Calculate weights and make sure theyre > 0
#                             wz = 1.0 - rangeMax * dabs( <double>iz2 - z2)
#                             wy = 1.0 - rangeMax * dabs( <double>iy2 - y2)
#                             wx = 1.0 - rangeMax * dabs( <double>ix2 - x2)
#                             w = dmax(0.0, wz) * dmax(0.0, wy) * dmax(0.0, wx)
#                             
#                             # Assign values
#                             result1_[iz2, iy2, ix2] += val * w
#                             coeff1_[ iz2, iy2, ix2] +=       w
#     
#     
#     # Divide by the coeffeicients
#     for iz2 in range(Nz):
#         for iy2 in range(Ny):
#             for ix2 in range(Nx):
#                 
#                 c = coeff1_[iz2,iy2,ix2]
#                 if c>0:
#                     result1_[iz2,iy2,ix2] = result1_[iz2,iy2,ix2] / c
#     
#     # Done
#     return result1_
