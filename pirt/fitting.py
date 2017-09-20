import numpy as np
import scipy.linalg

from pirt import PointSet


if True:
    # Precalculate a matrix for fitting a quadratic polynom to three points
    # to find the subpixel maximum value in an 1D image.

    # a x x + b x + c
    _precalculated_A1 = np.matrix(" 1 -1 1; 0 0 1; 1 1 1")  # quadratic 3 points 1D
    _precalculated_Ai1 = scipy.linalg.pinv(_precalculated_A1)

    # Precalculate a matrix for fitting a quadratic polynom to five points
    # to find the subpixel maximum value in an 2D image. Note that this is the
    # same as fitting two times a 1D polynom.

    # a x x + b y y + c x + d y + e = value
    tmp  = '0  1  0 -1  1 ;' # x = 0, y = -1    
    tmp += '1  0 -1  0  1 ;' # x = -1, y = 0
    tmp += '0  0  0  0  1 ;' # x = 0, y = 0
    tmp += '1  0  1  0  1 ;' # x = 1, y = 0    
    tmp += '0  1  0  1  1 ' # x = 0, y = 1    
    _precalculated_A2 = np.matrix(tmp)
    _precalculated_Ai2 = scipy.linalg.pinv(_precalculated_A2)
    del tmp


def fit_lq1(pp):
    """ fit_lq1(points) -> t_max, [a,b,c]
    
    Fit quadratic polynom to three points in 1D. If more than three 
    points are given, the result is the least squares solution.
    
    points can be a 2D ndarray object, resulting in a general
    solution. If only 3 values are given (as list of numpy array),
    the point locations are assumed at t=(-1,0,1).
    
    """
    if isinstance(pp, np.ndarray) and pp.ndim == 2:
        # Arbitraty position of values. Apply general approach
        
        # Prepare A
        A = PointSet(3)
        for x in pp[:,0]:
            A.append(x**2, x, 1)
        Ai = scipy.linalg.pinv(np.matrix(A))
        # Prepare B
        B = np.matrix(pp[:,1])
        # Solve
        X = Ai * B.T
        # Find extreme
        P = X.A.ravel().tolist()    
        x = -0.5 * P[1]/P[0]   
        # Done
        return x, P
    
    else:
        # Values defined at fixed positions (-1,0,1)
        
        # Make suitable form multiplication
        B = np.matrix(pp[0:3]).transpose()
        # Solve    
        X = _precalculated_Ai1 * B    
        # Find extreme
        P = X.A.ravel().tolist()    
        x = -0.5 * P[1]/P[0]   
        # done
        return x, P


def fit_lq2(patch, sample=False):
    """ fit_lq2(patch) --> x_max, y_max
    
    Quadratic (least squares) 2d fitting and subsequent finding of real max. 
    
    Patch is a matrix of the 9 pixels around the extreme.
    Uses the centre and its 4 direct neighours, as specified 
    in patch. fit_lq2(patch, True) will also return a 
    300x300 image illustrating the surface in the 3x3 area.
    
    """
    
    # Get measurements at the points in A    
    patch = patch.ravel()
    I = [1,3,4,5,7] # to correspond with the order of the eqs above...
    B = np.matrix(patch[I]).transpose()
    
    # Calculate X, which is [a,b,c,d,e]
    X = _precalculated_Ai2 * B
    # Make array, such that ravel works, and convert to list
    P = np.array(X).ravel().tolist() 
    # Make tuple, so we can unpack!
    a,b,c,d,e = tuple(P)
    
    # Differentiating the equation to x:
    x = -0.5*c/a
    y = -0.5*d/b
    
    # Produce a sample of the fitted surface
    if sample:
        vals = np.zeros((300,300))
        for ix in range(300):
            for iy in range(300):
                xx = ix/100.0-1.5
                yy = iy/100.0-1.5
                vals[iy,ix] = a*xx*xx + b*yy*yy + c*xx + d*yy + e        
        #print a*x*x + b*y*y + c*x + d*y + e
        return x,y, vals 
    else:
        return x,y
