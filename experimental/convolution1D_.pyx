# Cython specific imports
import numpy as np
cimport numpy as np
import cython 

# Type defs, we support float32 and float64
FLOAT32 = np.float32
FLOAT64 = np.float64

# Define datatypes 
ctypedef np.float32_t FLOAT32_T
ctypedef np.float64_t FLOAT64_T

# Floor operator (deal with negative numbers)
cdef inline int floor(double a): return <int>a if a>=0.0 else (<int>a)-1

fft = np.fft.fft
ifft = np.fft.ifft

@cython.boundscheck(False)
@cython.wraparound(False)
cdef convolve_32(    np.ndarray[FLOAT32_T, ndim=1] data, 
                    np.ndarray[FLOAT32_T, ndim=1] result, 
                    np.ndarray[FLOAT32_T, ndim=1] kernel,
                    int m1, int m2, int n ):

    # Define parameters
    cdef int i, ii, j
    cdef double val
    
    for i in range(n):
        
        # Init val
        val = 0.0
        ii = i + m1 + m1 # or m2?
        
        # Collect val
        for j in range(m1+m2+1):
            val += data[ii-j] * kernel[j]
        
        # Write val
        result[i] = val
    
#     for i in range(m1, m1+n):
#         
#         # Init val
#         val = 0.0
#         
#         # Collect val
#         for j in range(-m1, m2+1):
#             val += data[i+j] * kernel[m1-j]
#         
#         # Write val
#         result[i] = val


@cython.boundscheck(False)
@cython.wraparound(False)
def convolve2_32(data, kernel, axis=0, edge_mode='constant'):
    
    # todo: check input, make kernel an array
    
    # Define variables
    cdef int i, j, ii, jlimit
    cdef int m1, m2, n
    cdef int x, y, z
    cdef int nz, ny, nx
    cdef double val
    
    # Initialize result
    result = np.empty_like(data)
    
    # Get margins
    m1 = int((kernel.size-1) /2)
    m2 = kernel.size - m1 -1

    # Determine result array sizes
    shape = data.shape
    r_shape = [s+m1+m2 for s in shape]
    ny = shape[0]
    nx = shape[1]
    
    # Typedef data array, result array and kernel
    cdef np.ndarray[FLOAT32_T, ndim=2] data_ = data
    cdef np.ndarray[FLOAT32_T, ndim=2] result_ = result
    cdef np.ndarray[FLOAT32_T, ndim=1] kernel_ = kernel
    
    # Create variables to hold 1D temp arrays
    cdef np.ndarray[FLOAT32_T, ndim=1] tmp1 = np.empty(max(r_shape), dtype=FLOAT32)
    cdef np.ndarray[FLOAT32_T, ndim=1] tmp2 = np.empty(max(r_shape), dtype=FLOAT32)
    
    
    if axis == 0:
        
        # Run along y
        n = shape[0]
        
        for x in range(nx):
            
            # Prepare temp array
            for i in range(n):
                tmp1[i+m1] = data_[i,x]
            for i in range(m1):
                tmp1[i] = 0.0
            for i in range(m1+n, m1+m2+n):
                tmp1[i] = 0.0
            
            # Fill array
            for i in range(n):
                
                # Init val
                val = 0.0
                ii = i + m1 + m1 # or m2?
                
                # Collect val
                for j in range( m1+m2+1):
                    val += tmp1[ii-j] * kernel_[j]
                
                # Write val
                #tmp2[i] = val
                result_[i,x] = val
            
            # I think the loop over i and j can also be put in 
            # a separate function
            #convolve_32(tmp1, tmp2, kernel_, m1, m2, n)
            
            # Write back (either here or right when setting val)
            #for i in range(n):
            #    result_[i,x] = tmp2[i]

    
    elif axis == 1:
        
        # Run along x
        n = shape[1]
        for i in range(ny):
            
            # Prepare temp array
            tmp1[m1:m1+n] = data[i,:]
            tmp1[:m1] = 0.0
            tmp1[m1+n:m1+m2+n] = 0.0
            
            # Fill array
            convolve_32(tmp1, tmp2, kernel, m1, m2, n)
            
            # Write back
            result[i,:] = tmp2[m1:m1+n]
    
    else:
        raise ValueError("Invalid axis.")
    
    # Done
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def fftConvolve2_32(data, kernel, axis=0, edge_mode='constant'):
    
    # todo: check input, make kernel an array
    
    # Define variables
    cdef int i, j, ii
    cdef int m1, m2, n
    cdef int x, y, z
    cdef int nz, ny, nx
    cdef double val
    
    # Initialize result
    result = np.empty_like(data)
    
    # Get margins
    m1 = int((kernel.size-1) /2)
    m2 = kernel.size - m1 -1

    # Determine result array sizes
    shape = data.shape
    r_shape = [s+m1+m2 for s in shape]
    ny = shape[0]
    nx = shape[1]
    
    # Typedef data array, result array and kernel
    cdef np.ndarray[FLOAT32_T, ndim=2] data_ = data
    cdef np.ndarray[FLOAT32_T, ndim=2] result_ = result
    cdef np.ndarray[FLOAT32_T, ndim=1] kernel_ = kernel
    
    # Create variables to hold 1D temp arrays
    cdef np.ndarray[FLOAT32_T, ndim=1] tmp1 = np.empty(max(r_shape), dtype=FLOAT32)
    cdef np.ndarray[FLOAT32_T, ndim=1] tmp2 = np.empty(max(r_shape), dtype=FLOAT32)
    
    
    if axis == 0:
        
        # Run along y
        n = shape[0]
        
        for x in range(nx):
            
            # Prepare temp array
            tmp1[m1:m1+n] = data_[:,x]
            tmp1[:m1] = 0.0
            tmp1[m1+n:m1+m2+n] = 0.0
            
            #tmp1 = data_[:,x]
            #tmp2 = result_[:,x]
            
            T1 = fft(tmp1)
            K = fft(tmp2, n+m1+m2)
            T2 = T1 * K
            t2 = np.abs(ifft(T2))
            
            # Write back
            result_[:,i] = t2[m1:m1+n]
            #result_[:,i] = tmp2[m1:m1+n]
            #result_[:,i] = tmp2[:n]
    
    elif axis == 1:
        
        # Run along x
        n = shape[1]
        for i in range(ny):
            
            # Prepare temp array
            tmp1[m1:m1+n] = data[i,:]
            tmp1[:m1] = 0.0
            tmp1[m1+n:m1+m2+n] = 0.0
            
            # Fill array
            convolve_32(tmp1, tmp2, kernel, m1, m2, n)
            
            # Write back
            result[i,:] = tmp2[m1:m1+n]
    
    else:
        raise ValueError("Invalid axis.")
    
    # Done
    return result
        
    
