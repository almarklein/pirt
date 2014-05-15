""" This is the definition file for interpolation_.pyx.
"""

cimport numpy as np

# determine datatypes for heap
ctypedef np.float32_t FLOAT32_T
ctypedef np.float64_t FLOAT64_T
ctypedef np.int16_t FLOATi16_T
ctypedef np.float32_t SAMPLE_T


cdef class CoefLut:
    
    # Attributes
    cdef double* _LUT
    cdef int N
    
    # Methods
    cdef void calculate_lut(self, spline_type, int N)
    cdef double* get_coef(self, double t) nogil
    cdef double* get_coef_from_index(self, int i)    
    cdef double spline_type_to_id(self, spline_type)
    
    cdef cubicsplinecoef(self, double splineId, double t, double* out)
    cdef cubicsplinecoef_basis(self, double t, double *out)
    cdef cubicsplinecoef_hermite(self, double t, double *out)
    cdef cubicsplinecoef_cardinal(self, double t, double *out, double tension)
    cdef cubicsplinecoef_catmullRom(self, double t, double *out)
    cdef cubicsplinecoef_lanczos(self, double t, double *out)
    cdef cubicsplinecoef_lagrange(self, double t, double *out)
    cdef cubicsplinecoef_linear(self, double t, double *out)    
    cdef cubicsplinecoef_quadratic(self, double t, double *out)    
    #
#     cdef cubicsplinecoef_cardinal_edge(self, double t, double *out)
#     cdef quadraticsplinecoef_left(self, double t, double *out)
#     cdef quadraticsplinecoef_right(self, double t, double *out)
    

cdef class AccurateCoef:
    
    # Attributes
    cdef double* _out
    cdef CoefLut _lut
    
    # Methods
    cdef double* get_coef(self, double t) nogil
