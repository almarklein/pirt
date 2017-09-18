# Atempt at Cuda implementation, but so far it is slower than the normal one.

import numba
from numba import cuda


@cuda.jit((numba.float64, ), device=True, inline=True)
def cuda_floor(i):
    if i >= 0:
        return int(i)
    else:
        return int(i) - 1


@cuda.jit
def warp2_cuda(data_, result_, samplesx_, samplesy_, order, spline_type):
    
    Ni = samplesx_.size
    Ny = data_.shape[0]
    Nx = data_.shape[1]
    
    ccx = cuda.local.array((4, ), numba.float64)
    ccy = cuda.local.array((4, ), numba.float64)
    
    if order == 3:
        
        # Iterate over all samples
        #for i in range(0, Ni):
        i = cuda.grid(1)
        if i < Ni:
            
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = cuda_floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = cuda_floor(dy); ty = dy-iy
            
            if (    ix >= 1 and ix < Nx-2 and 
                    iy >= 1 and iy < Ny-2       ):
                
                # Get coefficients.
                ccx[0] = - 0.5*tx**3 + tx**2 - 0.5*tx        
                ccx[1] =   1.5*tx**3 - 2.5*tx**2 + 1
                ccx[2] = - 1.5*tx**3 + 2*tx**2 + 0.5*tx
                ccx[3] =   0.5*tx**3 - 0.5*tx**2
                ccy[0] = - 0.5*ty**3 + ty**2 - 0.5*ty        
                ccy[1] =   1.5*ty**3 - 2.5*ty**2 + 1
                ccy[2] = - 1.5*ty**3 + 2*ty**2 + 0.5*ty
                ccy[3] =   0.5*ty**3 - 0.5*ty**2
                
                # Apply
                val = 0.0
                for cy in range(4):
                    for cx in range(4):
                        val += data_[iy+cy-1,ix+cx-1] * ccy[cy] * ccx[cx]
                result_[i] = val
            
            elif (  dx>=-0.5 and dx<=Nx-0.5 and 
                    dy>=-0.5 and dy<=Ny-0.5     ):
                # Edge effects
                
                # Get coefficients. Slower, but only needed at edges.
                ccx[0] = - 0.5*tx**3 + tx**2 - 0.5*tx        
                ccx[1] =   1.5*tx**3 - 2.5*tx**2 + 1
                ccx[2] = - 1.5*tx**3 + 2*tx**2 + 0.5*tx
                ccx[3] =   0.5*tx**3 - 0.5*tx**2
                ccy[0] = - 0.5*ty**3 + ty**2 - 0.5*ty        
                ccy[1] =   1.5*ty**3 - 2.5*ty**2 + 1
                ccy[2] = - 1.5*ty**3 + 2*ty**2 + 0.5*ty
                ccy[3] =   0.5*ty**3 - 0.5*ty**2
                
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
        #for i in range(0, Ni):
        i = cuda.grid(1)
        if i < Ni:
            
            # Get integer sample location and t-factor
            dx = samplesx_[i]; ix = cuda_floor(dx); tx = dx-ix
            dy = samplesy_[i]; iy = cuda_floor(dy); ty = dy-iy
            
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
        #for i in range(0, Ni):
        i = cuda.grid(1)
        if i < Ni:
            
            # Get integer sample location
            dx = samplesx_[i]; ix = cuda_floor(dx+0.5)
            dy = samplesy_[i]; iy = cuda_floor(dy+0.5)
            
            if (    ix >= 0 and ix < Nx and
                    iy >= 0 and iy < Ny     ):
                # Nearest neighbour interpolation
                result_[i] = data_[iy,ix]
            else:
                # Out of range
                result_[i] = 0.0
