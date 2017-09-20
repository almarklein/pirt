import numpy as np

from .. import Aarray
from .. import interp
from ..splinegrid import GridInterface, FD
from ..splinegrid import calculate_multiscale_sampling

from ._deformbase import Deformation


class DeformationField(Deformation):
    """ DeformationField(*fields)
    
    A deformation field represents a deformation using an array for
    each dimension, thus specifying the deformation at each pixel/voxel. 
    The fields should be in z-y-x order. The deformation is represented
    in world units (not pixels, unless pixel units are used).
    
    Can be initialized with:
      * DeformationField(field_z, field_y, field_x)
      * DeformationField(image) # Null deformation
      * DeformationField(3) # Null deformation specifying only the ndims
    
    This class has functionality to reshape the fields. This can be usefull
    during registration if using a scale space pyramid.
    
    """
    
    def __init__(self, *fields):
        
        if len(fields)==1 and isinstance(fields[0], (list, tuple)):
            fields = fields[0]
        
        if not fields:
            raise ValueError('No arguments given to DeformationField().')
        
        if len(fields)==1 and isinstance(fields[0], int):
            # Null deformation
            ndim = fields[0]
            self._field_shape = tuple([1 for i in range(ndim)])
            self._field_sampling = tuple([1.0 for i in range(ndim)])
            self._fields = []
        
        elif len(fields)==1 and isinstance(fields[0], FD):
            # Null deformation with known dimensions
            self._field_shape = fields[0].shape
            self._field_sampling = fields[0].sampling
            self._fields = []
        
        else:
            # The fields itself are given
            
            # Check
            if not self._check_fields_same_shape(fields):
                raise ValueError('Fields must all have the same shape.')
            if len(fields) != fields[0].ndim:
                raise ValueError('There must be a field for each dimension.')
            
            # Make Aarray if they are not
            fields2 = []
            for field in fields:
                if not hasattr(field, 'sampling'):
                    field = Aarray(field)
                fields2.append(field)
            
            # Store dimensions and fields
            self._field_shape = fields2[0].shape
            self._field_sampling = fields2[0].sampling
            self._fields = tuple(fields2)
    
    
    def __repr__(self):
        mapping = ['backward', 'forward'][self.forward_mapping]
        if self.is_identity:
            return '<DeformationField (%s) with %iD identity deformation>' % (
                mapping, self.ndim)
        else:
            shapestr = 'x'.join([('%i'%s) for s in self.field_shape])
            samplingstr = 'x'.join([('%1.2f'%s) for s in self.field_sampling])
            return '<DeformationField (%s) with shape %s and sampling %s>' % (
                        mapping, shapestr, samplingstr)
    
    
    def resize_field(self, new_shape):
        """ resize_field(new_shape)
        
        Create a new DeformationField instance, where the underlying field is
        resized. 
        
        The parameter new_shape can be anything that can be converted 
        to a FieldDescription instance.
        
        If the field is already of the correct size, returns self.
        
        """
        
        if isinstance(new_shape, DeformationIdentity):
            return self
        elif self.is_identity:
            return self.__class__( FD(self) )
        else:
            fd1 = FD(self)
            fd2 = FD(new_shape)
            if (fd1.shape == fd2.shape) and self._sampling_equal(fd1, fd2):
                return self
            else:
                return self._resize_field(fd2)
    
    
    def _sampling_equal(self, fd1, fd2):
        sam_errors = [abs(s1-s2) for s1,s2 in zip(fd1.sampling, fd2.sampling)]
        if max(sam_errors) > 0.0001 * min(fd1.sampling): # 0.01% error
            return False
        else:
            return True
    
    
    def _resize_field(self, fd):
        """ _resize_field(fd)
        
        Create a new DeformationField instance, where the underlying field is
        reshaped. Requires a FieldDescription instance.
        
        """
        
        # Interpolate (upscale/downscale the other)
        fields = []
        for field1 in self:
            field2 = interp.resize(field1, fd.shape, 3, 'C', prefilter=False, extra=False)
            fields.append( field2 )
        
        # Verify
        if not self._sampling_equal(field2, fd):
            if not fd.defined_sampling: #sum([(s==1) for s in fd.sampling]) == self.ndim:
                pass # Sampling probably not given
            else:
                raise ValueError('Given reference field sampling does not match.')
        
        # Return
        return self.__class__(fields)

    
    ## Sequence stuff
    
    def __len__(self):
        return len(self._fields)
    
    
    def __getitem__(self, item):
        if isinstance(item, int):
            if item>=0 and item<len(self._fields):
                return self._fields[item]
            else:
                raise IndexError("Field index out of range.")
        else:
            raise IndexError("DeformationField only supports integer indices.")
    
    
    def __iter__(self):
        return tuple(self._fields).__iter__()
    
    
    ## Helper methods
    
    def _check_fields_same_shape(self, fields):
        """ _check_fields_same_shape(shape)
        
        Check whether the given fields all have the same shape.
        
        """
        
        # Get shape of first field
        shape = fields[0].shape
        
        # Check if matches with all other fields
        for field in fields:
            if field.shape != shape:
                return False
        else:
            return True
    
    
    def _check_which_shape_is_larger(self, shape):
        """ _check_which_shape_is_larger(self, shape)
        
        Test if shapes are equal, smaller, or larger:
          *  0: shapes are equal;
          *  1: the shape of this deformation field is larger
          * -1: the given shape is larger
        
        """
        
        larger_smaller = None
        
        for d in range(self.ndim):
            
            # Get larger or smaller
            if self.field_shape[d] > shape[d]:
                tmp = 1
            elif self.field_shape[d] < shape[d]:
                tmp = -1
            else:
                tmp = 0
            
            # Check
            if larger_smaller is None:
                larger_smaller = tmp
            elif larger_smaller != tmp:
                raise ValueError('The shapes of these arrays do not match.')
        
        # Done
        return larger_smaller

    
    ## Multiscale composition from points or a field
    
    @classmethod        
    def from_field_multiscale(cls, field, sampling, weights=None, 
                            injective=True, frozenedge=True, fd=None):
        """ from_field_multiscale(field, sampling, weights=None, 
                                  injective=True, frozenedge=True, fd=None)
        
        Create a DeformationGrid from the given deformation field 
        (z-y-x order). 
        
        Uses B-spline grids in a multi-scale approach to regularize the 
        sparse known deformation. This produces a smooth field (minimal
        bending energy), similar to thin plate splines.
        
        The optional weights array can be used to individually weight the
        field elements. Where the weight is zero, the values are not 
        evaluated. The speed can therefore be significantly improved if 
        there are relatively few nonzero elements.
        
        Parameters
        ----------
        field : list of numpy arrays
            These arrays describe the deformation field (one per dimension).
        sampling : scalar
            The smallest sampling of the B-spline grid that is used to create
            the field.
        weights : numpy array
            This array can be used to weigh the contributions of the 
            individual elements.
        injective : bool or number
            Whether to prevent the grid from folding. An injective B-spline
            grid is diffeomorphic. When a number between 0 and 1 is given, 
            the unfold constraint can be tightened to obtain smoother
            deformations.
        frozenedge : bool
            Whether the edges should be frozen. This can help the registration
            process. Also, when used in conjunction with injective, a truly
            diffeomorphic deformation is obtained: every input pixel maps
            to a point within the image boundaries.
        fd : field
            Field description to describe the shape and sampling of the
            underlying field to be deformed.
        
        Notes
        -----
        The algorithmic is based on:
        Lee S, Wolberg G, Chwa K-yong, Shin SY. "Image Metamorphosis with
        Scattered Feature Constraints". IEEE TRANSACTIONS ON VISUALIZATION
        AND COMPUTER GRAPHICS. 1996;2:337--354.
        
        The injective constraint desctribed in this paper is not applied
        by this method, but by the DeformationGrid, since it is method 
        specifically for deformations.
        
        """
        
        if fd is None:
            fd = field[0]
        
        def setR(gridAdd, residu):
            gridAdd._set_using_field(residu, weights, injective, frozenedge)
        
        def getR(gridRef=None):
            if gridRef is None:
                return field
            else:
                # if frozenedge:
                #     interp.fix_samples_edges(defField)
                residu = []
                for d in range(gridRef.ndim):
                    residu.append(field[d] - gridRef[d])
                    #residu.append(field[d] - gridRef[d].get_field())
                return residu
        
        return cls._multiscale(setR, getR, fd, sampling)
    
    
    @classmethod    
    def from_points_multiscale(cls, image, sampling, pp1, pp2, 
                               injective=True, frozenedge=True):
        """ from_points_multiscale(image, sampling, pp1, pp2, 
                                   injective=True, frozenedge=True)
        
        Obtains the deformation field described by the two given sets
        of corresponding points. The deformation describes moving the
        points pp1 to points pp2. Note that backwards interpolation is 
        used, so technically, the image is re-interpolated by sampling
        at the points in pp2 from pixels specified by the points in pp1.
        
        Uses B-spline grids in a multi-scale approach to regularize the 
        sparse known deformation. This produces a smooth field (minimal
        bending energy), similar to thin plate splines.
        
        Parameters
        ----------
        image : numpy array or shape
            The image (of any dimension) to which the deformation applies.
        sampling : scalar
            The sampling of the smallest grid to describe the deform.
        pp1 : PointSet, 2D ndarray
            The base points.
        pp2 : PointSet, 2D ndarray
            The target points.
        injective : bool or number
            Whether to prevent the grid from folding. An injective B-spline
            grid is diffeomorphic. When a number between 0 and 1 is given, 
            the unfold constraint can be tightened to obtain smoother
            deformations.
        frozenedge : bool
            Whether the edges should be frozen. This can help the registration
            process. Also, when used in conjunction with injective, a truly
            diffeomorphic deformation is obtained: every input pixel maps
            to a point within the image boundaries.
        
        Notes
        -----
        The algorithmic is based on:
        Lee S, Wolberg G, Chwa K-yong, Shin SY. "Image Metamorphosis with
        Scattered Feature Constraints". IEEE TRANSACTIONS ON VISUALIZATION
        AND COMPUTER GRAPHICS. 1996;2:337--354.
        
        The injective constraint desctribed in this paper is not applied
        by this method, but by the DeformationGrid, since it is method 
        specifically for deformations.
        
        """
        assert isinstance(pp1, np.ndarray) and pp1.ndim == 2
        assert isinstance(pp2, np.ndarray) and pp2.ndim == 2
        
        # Obtain reference points and vectors
        if cls._forward_mapping:  # using the prop on cls would get the prop, not the value!
            pp = pp1
            dd = pp2 - pp1
        else:
            pp = pp2
            dd = pp1 - pp2
        
        # Init residu
        _residu = dd.copy()
        
        def setResidu(gridAdd, residu):
            gridAdd._set_using_points(pp, residu, injective, frozenedge)
        
        def getResidu(defField=None):
            if defField is None:
                return _residu
            else:
                for d in range(defField.ndim):
                    i = defField.ndim - d - 1
                    tmp = defField.get_field_in_points(pp, d)
                    _residu[:,i] = dd[:,i] - tmp
                return _residu
        
        # Use multiscale method
        grid = cls._multiscale(setResidu, getResidu, image, sampling)
        return grid
    
    
    @classmethod
    def _multiscale(cls, setResidu, getResidu, field, sampling):
        """ _multiscale(setResidu, getResidu, field, sampling)
        
        Method for generating a deformation field using a multiscale 
        B-spline grid approach. from_field_multiscale()
        and from_points_multiscale() use this classmethod by each supplying 
        appropriate setResidu and getResidu functions.
        
        """
        
        # Get sampling
        tmp = GridInterface(field, 1)
        sMin, sMax = calculate_multiscale_sampling(tmp, sampling)
        s, sRef = sMax, sMin*0.9
        
        # Init our running deformation field (identity deform at first)
        defField = cls(FD(field))
        
        # Init residu
        residu = getResidu()
        
        # defField: working field
        # gridAdd: grid to add to working-field
        
        if cls._forward_mapping:  # using the prop on cls would get the prop, not the value!
            DeformationGrid = DeformationGridForward
        else:
            DeformationGrid = DeformationGridBackward
        
        while s > sRef:
            
            # Init loop
            error = 9999999999999999999
            oldError = error**2
            deltaError = oldError-error
            alpha = 0.9
            K = 2.046392675
            th = alpha * ((1/K) * s)**2
            
            # Create grid by combining refined grid of previous pass and
            # the gridAdd. 
            iter, maxIter= 0, 16
            while deltaError > th and iter<maxIter:
                iter += 1
                
                # Create addGrid using the residual values
                gridAdd = DeformationGrid(field, s)        
                setResidu(gridAdd, residu)
                
                # Compose deformation field and gridAdd
                defField.resize_field(gridAdd)
                defField = defField.compose(gridAdd)
                #defField = defField.add(gridAdd)
                
                # Calculate error
                residu = getResidu(defField)
                if isinstance(residu, (list, tuple)):
                    error = residu[0]**2
                    for i in range(1,len(residu)):
                        error += residu[i]**2
                elif hasattr(residu, '_is_Pointset'):
                    print('Warning, vv.Pointset is used ...')
                    error = residu[:,0]**2
                    for i in range(1, residu.ndim):
                        error += residu[:,i]**2
                elif residu.ndim == 2:
                    error = residu[:,0]**2
                    for i in range(1, residu.shape[1]):
                        error += residu[:,i]**2
                else:
                    error = residu**2
                error = float(error.max())
                deltaError = oldError - error
                oldError = error
            
            #print iter, 'multiscale iters'
            
            # Prepare for next iter
            s /= 2.0
            
            if s > sRef:
                # Get current values in the field and calculate residual
                # Use refGrid, as it does not *exactly* represent the
                # same field as grid.
                residu = getResidu(defField)
        
        # Done
        return defField
    
    ## Testing
    
    def test_jacobian(self, show=True):
        """ test_jacobian(show=True)
        
        Test the determinand of the field's Jacobian. It should be all 
        positive for the field to be diffeomorphic.
        
        Returns the number of pixels where the Jacobian <= 0. If show==True,
        will show a figure with these pixels indicated.
        
        """
        
        if self.ndim == 2:
            
            # Calculate gradients
            gradYY, gradYX = np.gradient(np.asarray(self[0]))
            gradXY, gradXX = np.gradient(np.asarray(self[1]))
            
            # Calculate determinants
            det = (gradYY+1)*(gradXX+1) - gradXY*gradYX
            
            if show:
                import visvis as vv
                vv.figure(); vv.imshow(det<=0)
            
            # Done
            return (det<=0).sum()
        
        else:
            raise ValueError('Cannot yet check if 3D field is diffeomorphic.')


# Import at the end to work around recursion
from ._subs import (DeformationIdentity,
                    DeformationGridForward, DeformationGridBackward,
                    #DeformationFieldForward, DeformationFieldBackward
                    )
