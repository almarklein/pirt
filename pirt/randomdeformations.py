""" Module randomDeformations

Provides a mean to produce random deformations.

Defines a function create_random_deformation() and a class to 
keep a set of random deformations: RandomDeformations.

By setting the seed of the random number generator, the (pseudo) random
deformations can be repeated.

"""

import numpy as np
import pirt


# Due to the from_field_multiscale thingy on the end to make the deform 
# injective this function becomes really slow.
def create_random_deformation_gaussian(im, amplitude=1, min_sigma=10, nblobs=50, seed=None):
    """ create_random_deformation(im, amplitude=1, min_sigma=10, nblobs=50, seed=None)
    
    Create a random deformation using Gaussian blobs or different scales.
    Returns a DeformationField instance.
    
    See also the class RandomDeformations.
    
    Parameters
    ----------
    im : numpy array
        The image to create a deformation field for.
    amplitude : scalar
        The relative amplitude of the deformations.
    min_sigma : scalar
        The smallest sigma to create Gaussian blobs for. The largest sigma
        is a quarter of the maximum shape element of the image.
    nblobs : integer
        The amount of Gaussian blobs to compose the deformation with.
    seed : int or None
        Seed for the random numbers to draw. If you want to repeat the 
        same deformations, apply the same seed multiple times.
    
    """
    
    # Seed generator
    np.random.seed(seed)
    
    fields = []
    for dimension in range(len(im.shape)):
        
        # Make field that preserves sampling if it is an Aarray, and is
        # expressed using float32 dtype.
        field = pirt.Aarray(im.shape, fill=0.0, dtype='float32')
        if hasattr(im, 'sampling'):
            field.sampling = im.sampling
        
        for iter in range(nblobs):
        
            # Get randomly sampled variables
            if True:
                # Get sigma
                max_sigma = max(field.shape)/4
                t = 2**np.random.uniform(0,1) # between 1 and 2
                sigma = (t-1) * (max_sigma-min_sigma) + min_sigma
                
                # Get amplitude
                amp = np.random.uniform(-1,1) * sigma**0.5 * amplitude
                
                # Get position
                pos = []
                for d in range(field.ndim):
                    tmp = np.random.uniform(0,field.shape[d])
                    pos.append(int(tmp))
            
            # Create patch
            patch = pirt.gaussfun.gaussiankernel2(sigma,0,0)
            patch = amp * patch / patch.max()
            
            # Get tail
            tail = int(np.ceil( patch.shape[0]/2 ))
            
            # Put the patch in
            if True:
                
                # Get upper right and lower left
                pos1, pos2 = [], []
                for d in range(field.ndim):
                    pos1.append( pos[d] - tail )
                    pos2.append( pos[d] + tail )
                
                # Get patch indices
                pos3, pos4 = [], []
                for d in range(field.ndim):
                    pos3.append( 0 )
                    pos4.append( tail*2 )
                
                # Correct indices
                for d in range(field.ndim):
                    if pos1[d] < 0:
                        pos3[d] = -pos1[d]
                        pos1[d] = 0
                    if pos2[d] >= field.shape[d]:
                        pos4[d] = field.shape[d] - pos2[d] - 2
                        pos2[d] = field.shape[d] - 1
                
                # Build slice objects
                slices_field = []
                slices_patch = []
                for d in range(field.ndim):
                    slices_field.append( slice(pos1[d],pos2[d]+1) )
                    slices_patch.append( slice(pos3[d],pos4[d]+1) )
                
                # Put patch in 
                field[tuple(slices_field)] += patch[tuple(slices_patch)]
        
        # Store field
        fields.append(field)
    
    # Make sure the deform is injectrive, and has frozenedges if required
    gridsampling = min_sigma / 2.0
    deform = pirt.DeformationFieldBackward.from_field_multiscale(fields, gridsampling,
                                            injective=True, frozenedge=True)
    # Apply a bit of Gaussian diffusion
    fields = []
    for field in deform:
        fields.append( pirt.diffuse(field, 1.0) )
    
    # Done
    return pirt.DeformationFieldBackward(*fields)


def create_random_deformation(im, amplitude=20, scale=50, n=50, 
                                frozenedge=True, mapping='backward', seed=None):
    """ create_random_deformation(im, amplitude=20, scale=50, n=50, 
                                frozenedge=True, mapping='backward', seed=None)
    
    Create a random deformation by creating two random sets of 
    deformation vectors which are then converted to an injective
    deformation field using Lee and Choi's method.
    
    See also the class RandomDeformations.
    
    Parameters
    ----------
    im : numpy array or pirt.FieldDescription
        The image to create a deformation field for, or anything tha can be
        converted to a FieldDesctription instance.
    amplitude : scalar
        The relative amplitude of the deformations. The deformation vectors
        are randomly chosen with a maximum norm of this amplitude.
    scale : scalar
        The smallest resolution of the B-spline grid to regularize the
        deformation. Default 50.
    n : integer
        The amount of vectors to generate. Default 50.
    frozenedge : bool
        Whether the edges remain fixed or not (default True).
    mapping : {'forward', 'backward'}
        Whether the generated deformation uses forward or backward mapping.
        Default backward.
    seed : int or None
        Seed for the random numbers to draw. If you want to repeat the 
        same deformations, apply the same seed multiple times.
    
    """
    
    # Seed generator
    np.random.seed(seed)
    
    # Get field description
    fd = pirt.FD(im)
    
    # Init pointsets
    pp1 = pirt.PointSet(fd.ndim)
    pp2 = pirt.PointSet(fd.ndim)
    
    # Generate random points
    for iter in range(n):
        
        # Generate random deformation vector
        p1, p2 = [],[]
        for i in range(fd.ndim):
            d = fd.ndim - i - 1
            
            # Get range and amplitude: in world coords!
            ran = fd.shape[d] * fd.sampling[d]
            amp = float(amplitude)
            
            # Generate numbers
            loc1 = np.random.uniform(0, ran)
            loc2 = np.random.uniform( max(0, loc1-amp), min(ran, loc1+amp))
            p1.append(loc1)
            p2.append(loc2)
        
        # Store
        pp1.append(*p1)    
        pp2.append(*p2)    
    
    # Get deformationfield class
    if mapping.lower() == 'forward':
        DeformationField = pirt.DeformationFieldForward
    elif mapping.lower() == 'backward':
        DeformationField = pirt.DeformationFieldBackward
    else:
        raise ValueError('Invalid mapping.')
    
    # Create field
    # mmm, it might not really matter whether we regularize the deform
    # using a forward or backward deform, does it?
    gridsampling = scale / 1.0
    deform = DeformationField.from_points_multiscale(fd, gridsampling,
                            pp1, pp2, injective=True, frozenedge=frozenedge)
    
    # Apply a bit of Gaussian diffusion
    fields = []
    for field in deform:
        fields.append( pirt.diffuse(field, 1.0) )
    
    # Done
    return DeformationField(*fields)
    


class RandomDeformations:
    """ RandomDeformations(im, amplitude=20, scale=50, n=50, 
                                frozenedge=True, mapping='backward', seed=None)
    
    Represents a collection of random deformations. This can be used
    in experiments to test multiple methods on the same set of random
    deformations.
    
    It creates (and stores) random deformations on the fly when requested.
    
    The seed given to create_random_deformation is (seed + 100*index).
    It can be used to produce a random, yet repeatable set of deformations.
    For the sake of science, let's use a seed when doing experiments that
    are going to be published in a scientific medium, so they
    can be reproduced by others. For the sake of simplicity, let's agree to
    use the arbitrary seeds listed here: 1234 for training sets, 5678 for 
    test sets.
    
    See also the function create_random_deformation()
    
    Example
    -------
    rd = RandomDeformations(im)
    deform1 = rd[0]
    deform2 = rd[1]
    # etc.
    
    """
    
    def __init__(self, im, amplitude=20, scale=50, n=50, 
                                frozenedge=True, mapping='backward', seed=None):
        # Store image description
        self._fd = pirt.FD(im)
        # Store other parameters
        self._amplitude = float(amplitude)
        self._scale = float(scale)
        self._n = int(n)
        #
        self._frozenedge = bool(frozenedge)
        self._mapping = mapping
        self._seed = seed
        
        # Store dict with deforms
        self._deforms = {}
    
    
    def __getitem__(self, index):
        """ Get the deformation at the specified index. If it does not yet
        exist, will create a new deformation using arguments specified 
        during initialization. 
        """
        
        # Check
        if not isinstance(index, int) or index<0:
            raise ValueError('Can only index using (non-negative) integers.')
        
        # Create item if not already there
        if index not in self._deforms.keys():
            seed = None
            if self._seed is not None:
                seed = self._seed + index*100
            deform = create_random_deformation( self._fd, self._amplitude,
                                                self._scale, self._n,
                                                self._frozenedge, self._mapping,
                                                seed )
            self._deforms[index] = deform
        
        # Return
        return self._deforms[index]
    
    
    def get(self, index, *args, **kwargs):
        """ get(index, *args, **kwargs)
        
        Get the deformation at the specified index. If it does not yet
        exist, will create a new deformation using the specified arguments.
        
        Note that this will not take the seed into account.
        
        """
        
        # Create item if not already there
        if index not in self._deforms.keys():
            deform = create_random_deformation(self._fd, *args, **kwargs)
            self._deforms[index] = deform
        
        # Return
        return self._deforms[index]
