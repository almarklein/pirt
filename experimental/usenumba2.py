"""
Pirt's interp function is 4x faster than skimage warp(). 
"""

import sys
from time import perf_counter
from skimage.transform import warp
import visvis as vv
import numpy as np
import numba
from numba import cuda
import imageio

try:
    import pirt.interp.interpolation_
except ImportError:
    pirt = None
    print('no pirt')

## Numba implementation


class CoefLut:
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
    
    _LUTS = {}  # cache of luts
    
    def __init__(self):
        self._LUT = None
    
    def calculate_lut(self, spline_type, N):
        """ calculate_lut(spline_type, N)
        
        Calculate the look-up table for the specified spline type
        with N entries.
        
        """
        
        # The actial length is 1 larger, so also t=1.0 is in the table
        N1 = N + 2 # and an extra bit for if we use linear interp.
        
        # Allocate array (first clear)
        self._LUT = np.zeros(N1 * 4, np.float64)
        self.N = N
        
        # Prepare
        step = 1.0 / N
        t = 0.0
        splineId = self.spline_type_to_id(spline_type)
        
        for i in range(N1):
            t += step
            out = self._LUT[i * 4:]
            # For each possible t, calculate the coefficients
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
    
    def spline_type_to_id(self, spline_type):
        """ spline_type_to_id(spline_type)
        
        Method to map a spline name to an integer ID. This is used
        so that the LUT can be created relatively fast without having to
        repeat the loop for each spline type.
        
        spline_type can also be a number between -1 and 1, representing
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
        elif spline_type.lower() in ['lin', 'linear']:
            return 98.0        
        elif spline_type.lower() in ['quad', 'quadratic']:
            return 99.0
        else:
            raise ValueError('Unknown spline type: ' + str(spline_type))
    
    def cubicsplinecoef_catmullRom(self, t, out):
        # See the doc for the catmull-rom spline, this is how the two splines
        # are combined by simply adding (and dividing by two) 
        out[0] = - 0.5*t**3 + t**2 - 0.5*t        
        out[1] =   1.5*t**3 - 2.5*t**2 + 1
        out[2] = - 1.5*t**3 + 2*t**2 + 0.5*t
        out[3] =   0.5*t**3 - 0.5*t**2
    
    def cubicsplinecoef_cardinal(self, t, out, tension):
        tau = 0.5 * (1 - tension)
        out[0] = - tau * (   t**3 - 2*t**2 + t )
        out[3] =   tau * (   t**3 -   t**2     )
        out[1] =           2*t**3 - 3*t**2 + 1  - out[3]
        out[2] = -         2*t**3 + 3*t**2      - out[0]
    
    def cubicsplinecoef_basis(self, t, out):
        out[0] = (1-t)**3                     /6.0
        out[1] = ( 3*t**3 - 6*t**2 +       4) /6.0
        out[2] = (-3*t**3 + 3*t**2 + 3*t + 1) /6.0
        out[3] = (  t)**3                     /6.0
    
    def cubicsplinecoef_hermite(self, t, out):
        out[0] =   2*t**3 - 3*t**2 + 1
        out[1] =     t**3 - 2*t**2 + t
        out[2] = - 2*t**3 + 3*t**2
        out[3] =     t**3 -   t**2
    
    def cubicsplinecoef_lagrange(self, t, out):
        k = -1.0  
        out[0] =               (t  )/(k  ) * (t-1)/(k-1) * (t-2)/(k-2)
        k= 0  
        out[1] = (t+1)/(k+1) *               (t-1)/(k-1) * (t-2)/(k-2)
        k= 1  
        out[2] = (t+1)/(k+1) * (t  )/(k  ) *               (t-2)/(k-2)
        k= 2  
        out[3] = (t+1)/(k+1) * (t  )/(k  ) * (t-1)/(k-1)
    
    def cubicsplinecoef_lanczos(self, t, out):
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
    
    def cubicsplinecoef_linear(self, t, out):
        out[0] = 0.0
        out[1] = (1.0-t)
        out[2] = t
        out[3] = 0.0
    
    def cubicsplinecoef_quadratic(self, t, out):
        # This corresponds to adding the two quadratic polynoms,
        # thus keeping genuine quadratic interpolation. However,
        # it has the same support as a cubic spline, so why use this?
        out[0] = 0.25*t**2 - 0.25*t
        out[1] = -0.25*t**2 - 0.75*t + 1
        out[2] = -0.25*t**2 + 1.25*t
        out[3] = 0.25*t**2 - 0.25*t
    
    @numba.jit
    def get_coef(self, t):
        """ get_coef(t)
        
        Get the coefficients for given value of t. This simply obtains
        the nearest coefficients in the table. For a more accurate result,
        use the AccurateCoef class.
        
        """
        i1 = int( t * self.N + 0.5) * 4
        return self._LUT[i1:i1 + 4]
        # a = self._LUT[i1 + 0]
        # b = self._LUT[i1 + 1]
        # c = self._LUT[i1 + 2]
        # d = self._LUT[i1 + 3]
        # return a, b, c, d
    
    def get_coef_linear(self, t):
        
        t = t * self.N
        i1 = int(t)
        i2 = i1 + 1
        t2 = t - i1
        t1 = 1.0 - t2
        
        i1 *= 4
        i2 *= 4
        
        LUT = self._LUT
        
        a = self._LUT[i1 + 0] * t1 + self._LUT[i2 + 0] * t2
        b = self._LUT[i1 + 1] * t1 + self._LUT[i2 + 1] * t2
        c = self._LUT[i1 + 2] * t1 + self._LUT[i2 + 2] * t2
        d = self._LUT[i1 + 3] * t1 + self._LUT[i2 + 3] * t2
        
        return a, b, c ,d
    
    def get_coef_from_index(self, i):
        """ get_coef_from_index(i)
        
        Get the spline coefficients using the index in the table.
        
        """
        return self._LUT[4 * i: 4 * i + 4]
    
    
    @classmethod
    def get_lut(cls, spline_type, N=32768):
        """ get_lut(spline_type, N=32768)
        
        Classmethod to get a lut of the given spline type and with
        the given amount of elements (default 2**15).
        
        This method uses a global buffer to store previously 
        calculated LUT's; if the requested LUT was created 
        earlier, it does not have to be re-calculated.
        
        """
        
        # Get id
        lut = CoefLut()
        key = lut.spline_type_to_id(spline_type), N
        
        # Create lut if not existing yet
        if key not in cls._LUTS.keys():
            lut.calculate_lut(spline_type, N)
            cls._LUTS[key] = lut
        
        # Return lut
        return cls._LUTS[key] 



def numba_interp(image, coords, order=1):
    
    
    assert order == 1 or order == 3
    coeffx = coeffy = CoefLut.get_lut('cardinal')
    result = np.empty_like(image)
    image_ = cuda.to_device(image)
    result_ = cuda.to_device(result.ravel())
    samplesx_ = cuda.to_device(coords[0].ravel())
    samplesy_ = cuda.to_device(coords[1].ravel())
    
    threadsperblock = 32
    blockspergrid = (samplesx_.size + (threadsperblock - 1)) # threadperblock

    _numba_interp[blockspergrid, threadsperblock](image_, result_, samplesx_, samplesy_, order, coeffx.N, cuda.to_device(coeffx._LUT.astype('float64')))
    result_.copy_to_host(result.ravel())  # only this array is copied back
    return result


#@cuda.jit('float64[:](float64[:], float64, int64)', device=True, inline=True)
@numba.jit((numba.float64[:], numba.float32, numba.int32))
def get_coef(lut, t, n):
    i1 = int( t * n + 0.5) * 4
    return lut[i1:i1 + 4]


@numba.jit
def get_coef_linear(lut, t, n):
    
    t = t * n
    i1 = int(t)
    i2 = i1 + 1
    t2 = t - i1
    t1 = 1.0 - t2
    
    i1 *= 4
    i2 *= 4
    
    a = lut[i1 + 0] * t1 + lut[i2 + 0] * t2
    b = lut[i1 + 1] * t1 + lut[i2 + 1] * t2
    c = lut[i1 + 2] * t1 + lut[i2 + 2] * t2
    d = lut[i1 + 3] * t1 + lut[i2 + 3] * t2
    
    return a, b, c ,d


# nopython: prevent using Python C API calls. Ensures fast code, but has limitations
# nogil: release gil in block that do not use the Python interpreter
# cuda.jit requires cudatoolkit to be installed
# target='gpu'

@cuda.jit#('void(float32[:,:], float32[:], float32[:], float32[:], int32, int32, float64[:])')
#@numba.jit(target='gpu')
#@numba.jit(nopython=True, nogil=True)
def _numba_interp(image, result_, samplesx_, samplesy_, order, lutn, lut):
    
    data_ = image#.ravel()
    
    #  todo: local_data = cuda.local.array()
    coefsx = cuda.local.array((4, ), numba.float64)
    coefsy = cuda.local.array((4, ), numba.float64)
    Ni = samplesx_.size
    Ny = image.shape[0]
    Nx = image.shape[1]
    
    
    if order == 3:
        
        # Iterate over all samples
        #for i in range(0, Ni):
        i = cuda.grid(1)
        if i < Ni:
            
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = int(dx); tx = dx-ix
            dy = samplesy_[i]; iy = int(dy); ty = dy-iy
            
            if (    ix >= 1 and ix < Nx-2 and 
                    iy >= 1 and iy < Ny-2       ):
                # Cubic interpolation
                # ccx = get_coef(lut, lutn, tx)
                # ccy = get_coef(lut, lutn, ty)
                i1 = int( tx * lutn + 0.5) * 4; ccx = lut[i1:i1 + 4]
                i1 = int( ty * lutn + 0.5) * 4; ccy = lut[i1:i1 + 4]
                val = 0.0
                for cy in range(4):
                    for cx in range(4):
                        val += data_[iy+cy-1,ix+cx-1] * ccy[cy] * ccx[cx]
                result_[i] = val
            
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5     ):
                # Edge effects
                
                # Get coefficients
                # ccx = get_coef(lut, lutn, tx)#.copy()
                # ccy = get_coef(lut, lutn, ty)#.copy()
                # todo: cannot allocate new mem in cuda (or need NRT), pass 4-element stub array as input
                i1 = int( tx * lutn + 0.5) * 4; ccx = lut[i1:i1 + 4]#.copy()
                i1 = int( ty * lutn + 0.5) * 4; ccy = lut[i1:i1 + 4]#.copy()
                for i in range(4):
                    coefsx[i] = ccx[i]
                    coefsy[i] = ccy[i]
                
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
                for cx in range(cx1, cx2):  val += coefsx[cx]
                val = 1.0/val
                for cx in range(cx1, cx2):  coefsx[cx] *= val
                #
                val = 0.0
                for cy in range(cy1, cy2):  val += coefsy[cy]
                val = 1.0/val
                for cy in range(cy1, cy2):  coefsy[cy] *= val
                
                # Combine elements
                # No need to pre-calculate indices: the C compiler is well
                # capable of making these optimizations.
                val = 0.0
                for cy in range(cy1, cy2):
                    for cx in range(cx1, cx2):
                        val += data_[iy+cy-1,ix+cx-1] * coefsy[cy] * coefsx[cx]
                result_[i] = val
            
            else:
                # Out of range
                result_[i] = 0.0
    
    elif order == 1:
        # Iterate over all samples
        #for i in range(0, Ni):
        i = cuda.grid(1)
        if i < Ni:
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = int(dx); tx = dx-ix
            dy = samplesy_[i]; iy = int(dy); ty = dy-iy
            
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



@numba.jit(nopython=True, nogil=True)
def _numba_interp_cpu(image, result_, samplesx_, samplesy_, order, lutn, lut):
    
    data_ = image#.ravel()
    
    
    Ni = samplesx_.size
    Ny = image.shape[0]
    Nx = image.shape[1]
    
    
    if order == 3:
        
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = int(dx); tx = dx-ix
            dy = samplesy_[i]; iy = int(dy); ty = dy-iy
            
            if (    ix >= 1 and ix < Nx-2 and 
                    iy >= 1 and iy < Ny-2       ):
                # Cubic interpolation
                #ccx = get_coef(lut, lutn, tx)
                #ccy = get_coef(lut, lutn, ty)
                i1 = int( tx * lutn + 0.5) * 4; ccx = lut[i1:i1 + 4]
                i1 = int( ty * lutn + 0.5) * 4; ccy = lut[i1:i1 + 4]
                val = 0.0
                for cy in range(4):
                    for cx in range(4):
                        val += data_[iy+cy-1,ix+cx-1] * ccy[cy] * ccx[cx]
                result_[i] = val
            
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5     ):
                # Edge effects
                
                # Get coefficients
                #ccx = get_coef(lut, lutn, tx).copy()
                #ccy = get_coef(lut, lutn, ty).copy()
                # todo: cannot allocate new mem in cuda (or need NRT), pass 4-element stub array as input
                i1 = int( tx * lutn + 0.5) * 4; ccx = lut[i1:i1 + 4]#.copy()
                i1 = int( ty * lutn + 0.5) * 4; ccy = lut[i1:i1 + 4]#.copy()
                
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
    
    elif order == 1:
        # Iterate over all samples
        for i in range(0, Ni):
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = int(dx); tx = dx-ix
            dy = samplesy_[i]; iy = int(dy); ty = dy-iy
            
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

##

im1 = imageio.imread('imageio:chelsea.png')[:300, :300, 2].astype('float32') / 255
# im1 = np.row_stack([im1] * 8)
# im1 = np.column_stack([im1] * 8)
im2 = np.zeros_like(im1)
im3 = np.zeros_like(im1)
im4 = np.zeros_like(im1)

ny, nx = im1.shape
coords1 = np.zeros((2, ny, nx), np.float32)
coords2 = np.zeros((ny, nx), np.float32), np.zeros((ny, nx), np.float32)

@numba.jit
def apply_coords(coords, coords1, coords2):
    for y in range(coords1.shape[0]):
        for x in range(coords1.shape[1]):
            coords[0, y, x] = y + 10 * np.sin(x*0.01)
            coords[1, y, x] = x + 10 * np.sin(y*0.1)
            coords2[y, x] = y + 10 * np.sin(x*0.01)
            coords1[y, x] = x + 10 * np.sin(y*0.1)

apply_coords(coords1, coords2[0], coords2[1])

N = 100
order = 3

def timeit(title, func, *args, **kwargs):
    # Run once, allow warmup
    res = func(*args, **kwargs)
    # Prepare timer
    t0 = perf_counter()
    te = t0 + 0.5
    count = 0
    # Run
    while perf_counter() < te:
        func(*args, **kwargs)
        count += 1
    # Process
    tr = perf_counter() - t0
    if tr < 1:
        print(title, 'took %1.1f ms (%i loops)' % (1000 * tr / count, count))
    else:
        print(title, 'took %1.3f s (%i loops)' % (tr / count, count))
    return res

if not pirt:
    im2 = timeit('warp', warp, im1, coords1, order=order)

if pirt:
    im3 = timeit('iterp', pirt.interp.interp, im1, coords2, order=order)

if sys.version_info > (3, 5):
    im4 = timeit('numba', numba_interp, im1, coords2, order)


vv.figure(1); vv.clf()
vv.subplot(221); vv.imshow(im1)
vv.subplot(222); vv.imshow(im2)
vv.subplot(223); vv.imshow(im3)
vv.subplot(224); vv.imshow(im4)

vv.use().Run()
