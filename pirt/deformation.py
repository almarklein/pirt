""" Module pirt.deformation

This modules implements two classes to represent deformations:

  * DeformationGrid: represents a deformation in world coordinates using
    a spline grid.
  * DeformationField: represents a deformation in world coordinates using
    an array for each dimension; it describes the deformation for each
    pixel/voxel.

But these are actually base classes, one should use
DeformationFieldBackward, DeformationFieldForward,
DeformationGridBackward and DeformationGridForward.

"""

import time  # noqa

import numpy as np
import visvis as vv

import pirt
from pirt.utils import Pointset, Aarray
from pirt import interp
from pirt.splinegrid import GridInterface, GridContainer, SplineGrid, FD
from pirt.splinegrid import _calculate_multiscale_sampling
from pirt import _splinegrid


class Deformation(object):
    """ Deformation
    
    This class is an abstract base class for deformation grids and 
    deformation fields. 
    
    A deformation maps one image (1D, 2D or 3D) to another. A deformation
    can be represented using a B-spline grid, or using a field (i.e. array).
    A deformation is either forward mapping or backward mapping.
    
    """
    
    _forward_mapping = None # Must be set to True or False in subclasses
    
    
    ## Base properties
    
    @property
    def forward_mapping(self):
        """ Returns True if this deformation is forward mapping. 
        """
        assert self._forward_mapping is not None, ("You're using an abstract "
            "deformation class (but must be forward or backward)")
        return self._forward_mapping
    
    @property
    def is_identity(self):
        """ Returns True if this is an identity deform, or null deform; 
        representing no deformation at all. 
        """
        if isinstance(self, DeformationGrid):
            return self[0]._knots is None
        elif isinstance(self, DeformationField):
            return not bool(self._fields)
        elif isinstance(self, DeformationIdentity):
            return True
        else:
            raise ValueError('Unknown deformation class.')
    
    @property
    def ndim(self):
        """ The number of dimensions of the deformation. 
        """
        return len(self._field_shape)
    
    @property
    def field_shape(self):
        """ The shape of the deformation field.
        """
        return tuple(self._field_shape)
    
    @property
    def field_sampling(self):
        """ For each dim, the sampling (distance between pixels/voxels) 
        of the field (all 1's if isotropic).
        """
        return tuple(self._field_sampling)
    
    
    ## Modifying and combining deformations
    
    def __add__(self, other):
        return self.add(other)
    
    def __mul__(self, other):
        if isinstance(other, Deformation):
            # Compose that creates a new field
            return other.compose(self)
        else:
            # Scale
            factor = other
            return self.scale(factor)
    
    def copy(self):
        """ copy()
        
        Copy this deformation instance (deep copy).
        
        """
        if self.is_identity:
            if isinstance(self, DeformationIdentity):
                return DeformationIdentity()
            else:
                return self.__class__(FD(self))
        else:
            return self.scale(1.0) # efficient copy
    
    
    def scale(self, factor):
        """ scale(factor)
        
        Scale the deformation (in-place) with the given factor. Note that the result
        is diffeomorphic only if the original is diffeomorphic and the 
        factor is between -1 and 1.
        
        """
        if isinstance(self, DeformationGrid):
            # Create new grid
            fd = FD(self)
            if self.forward_mapping:
                newDeform = DeformationGridForward(fd, self.grid_sampling)
            else:
                newDeform = DeformationGridBackward(fd, self.grid_sampling)
            # Scale knots
            for d in range(self.ndim):
                if factor == 1.0:
                    newDeform[d]._knots = self[d]._knots.copy()
                else:
                    newDeform[d]._knots = self[d]._knots * float(factor)
            return newDeform
        
        elif isinstance(self, DeformationField):
            # Scale fields
            fields = []
            for d in range(self.ndim):
                if factor == 1.0:
                    fields.append( self[d].copy() )
                else:
                    fields.append( self[d] * factor )
            # Create new field
            if self.forward_mapping:
                newDeform = DeformationFieldForward(*fields)
            else:
                newDeform = DeformationFieldBackward(*fields)
            return newDeform
        else:
            raise ValueError('Unknown deformation class.')
    
    
    def add(def1, def2):  # noqa - flake8 wants first arg to be self
        """ add(other)
        
        Combine two deformations by addition. 
        
        Returns a DeformationGrid instance if both deforms are grids. 
        Otherwise returns deformation field. The mapping (forward/backward)
        is taken from the left deformation.
        
        Notes
        -----
        Note that the resulting deformation is not necesarily diffeomorphic 
        even if the individual deformations are.
        
        Although diffeomorphisms can in general not be averaged, the 
        constraint of Choi used in this framework enables us to do so 
        (add the individual deformations and scale with 1/n).
        
        This function can also be invoked using the plus operator.
        
        """
        
        # Check input class
        if not isinstance(def1, Deformation):
            raise ValueError('Can only combine Deformations (left one is not).')
        if not isinstance(def2, Deformation):
            raise ValueError('Can only combine Deformations (right one is not).')
        
        if def1.is_identity:
            # That's easy
            return def2.copy()
            
        elif def2.is_identity:
            # That's easy
            return def1.copy()
        
        else:
            # Apply normally
                        
            # Check shape match
            if def1.field_shape != def2.field_shape:
                raise ValueError('Can only combine deforms with same field shape.')
            
            # Check mapping match
            if def1.forward_mapping != def2.forward_mapping:
                raise ValueError('Can only combine deforms with the same mapping')
            
            ii = isinstance
            if ii(def1, DeformationGrid) and ii(def2, DeformationGrid):                 
                # Keep it a grid
                
                # Check sampling
                if def1.grid_sampling != def2.grid_sampling:
                    raise ValueError('Cannot add: sampling of grids does not match.')
                # Create new grid
                fd = FD(def1)
                if def1.forward_mapping:
                    newDeform = DeformationGridForward(fd, def1.grid_sampling)
                else:
                    newDeform = DeformationGridBackward(fd, def1.grid_sampling)
                # Add knots
                for d in range(def1.ndim):
                    newDeform[d]._knots = def1[d]._knots + def2[d]._knots
                return newDeform
            
            else:
                # Make it a field
                
                # Add fields
                fields = []
                for d in range(def1.ndim):
                    fields.append( def1.get_field(d) + def2.get_field(d) )
                # Create new field
                if def1.forward_mapping:
                    newDeform = DeformationFieldForward(*fields)
                else:
                    newDeform = DeformationFieldBackward(*fields)
                return newDeform
    
    
    def compose(def1, def2):  # noqa - flake8 wants first arg to be self
        """ compose(other):
        
        Combine two deformations by composition. The left is the "static"
        deformation, and the right is the "delta" deformation.
        
        Always returns a DeformationField instance. The mapping 
        (forward/backward) of the result is taken from the left deformation.
        
        Notes
        -----
        Let "h = f.compose(g)" and "o" the mathematical composition operator.
        Then mathematically "h(x) = g(f(x))" or "h = g o f".
        
        Practically, new deformation vectors are created by sampling in one
        deformation, at the locations specified by the vectors of the other.
        
        For forward mapping we sample in g at the locations of f. For backward 
        mapping we sample in f at the locations of g.
        
        Since we cannot impose the field values for a B-spline grid without
        resorting to some multi-scale approach (which would use composition
        and is therefore infinitely recursive), the result is always a
        deformation field.
        
        If the deformation to sample in (f for forward mapping, g for backward)
        is a B-spline grid, the composition does not introduce any errors;
        sampling in a field introduces interpolation errors. Since the static
        deformation f is often a DeformationField, forward mapping is preferred
        with regard to the accuracy of composition.
        
        """
        
        # Check input class
        if not isinstance(def1, Deformation):
            raise ValueError('Can only combine Deformations (left one is not).')
        if not isinstance(def2, Deformation):
            raise ValueError('Can only combine Deformations (right one is not).')
        
        if def1.is_identity:
            # That's easy
            return def2.copy().as_deformation_field()
        elif def2.is_identity:
            # That's easy
            return def1.copy().as_deformation_field()
        
        else:
            # Apply normally
            
            # Check shape match
            if def1.field_shape != def2.field_shape:
                raise ValueError('Can only combine deforms with same field shape.')
            
            # Check mapping match
            if def1.forward_mapping != def2.forward_mapping:
                raise ValueError('Can only combine deforms with the same mapping')
            
            if def1.forward_mapping:
                fields = def1._compose_forward(def2)
                return DeformationFieldForward(*fields)
            else:
                fields = def1._compose_backward(def2)
                return DeformationFieldBackward(*fields)
    
    
    def _compose_forward(def1, def2):  # noqa - flake8 wants first arg to be self
        # Sample in def2 at the locations pointed to by def1
        # Sample in the other at the locations pointed to by this field
        
        # Make sure the first is a deformation field
        def1 = def1.as_deformation_field()
        
        # Get sample positions in pixel coordinates
        sampleLocations = def1.get_deformation_locations()
        
        fields = []
        if isinstance(def2, DeformationGrid):
            # Composition with a grid has zero error
            for d in range(def1.ndim):
                field1 = def1[d]
                grid2 = def2[d]
                field = _splinegrid.get_field_at(grid2, sampleLocations)
                field = pirt.Aarray(field1+field, def1.field_sampling)
                fields.append(field)
        elif isinstance(def2, DeformationField):
            # Composition with a field introduces interpolation artifacts
            for d in range(def1.ndim):
                field1 = def1[d]
                field2 = def2[d]
                # linear or cubic. linear faster and I've not seen cubic doing better
                field = pirt.interp.warp(field2, sampleLocations, 'linear')
                field = pirt.Aarray(field1+field, def1.field_sampling)
                fields.append(field)
        else:
            raise ValueError('Unknown deformation class.')
        
        # Done
        return fields
    
    def _compose_backward(def1, def2):  # noqa - flake8 wants first arg to be self
        # Sample in def1 at the locations pointed to by the def2
        return def2._compose_forward(def1)
    
    
    ## Getting and applying 
    
    
    def get_deformation_locations(self):
        """ get_deformation_locations()
        
        Get a tuple of arrays (x,y,z order) that represent sample locations
        to apply the deformation. The locations are absolute and expressed
        in pixel coordinates. As such, they can be fed directly to interp()
        or project().
        
        """
        
        # Get as deformation field
        deform = self.as_deformation_field()
        
        # Reverse fields
        deltas = [s for s in reversed(deform)]
        
        # Make absolute and return
        return pirt.interp.make_samples_absolute(deltas) 
    
    
    def get_field(self, d):
        """ get_field(d)
        
        Get the field corresponding to the given dimension.
        
        """
        if isinstance(self, DeformationGrid):
            return self[d].get_field()
        elif isinstance(self, DeformationField):
            return self[d]
        else:
            raise ValueError('Unknown deformation class.')
    
    
    def get_field_in_points(self, pp, d, interpolation=1):
        """ get_field_in_points(pp, d, interpolation=1)
        
        Obtain the field for dimension d in the specied points. 
        The interpolation value is used only if this is a deformation
        field.
        
        The points pp should be a Pointset (x-y-z order).
        
        """
        if isinstance(self, DeformationGrid):
            return self[d].get_field_in_points(pp)
        elif isinstance(self, DeformationField):
            data = self[d]
            samples = []
            samling_xyz = [s for s in reversed(self.field_sampling)]
            for i in range(self.ndim):
                s = pp[:,i] / samling_xyz[i]
                samples.append(s)
            return pirt.interp.interp(data, samples, order=interpolation)
        else:
            raise ValueError('Unknown deformation class.')
    
    
    def apply_deformation(self, data, interpolation=3):
        """ apply_deformation(data, interpolation=3)
        
        Apply the deformation to the given data. Returns the deformed data.
        
        Parameters
        ----------
        data : numpy array 
            The data to deform
        interpolation : {0,1,3}
            The interpolation order (if backward mapping is used). 
        
        """
        
        # Null deformation
        if self.is_identity:
            return data
        
        # Make sure it is a deformation field
        deform = self.as_deformation_field()
        
        # Need upsampling?
        deform = deform.resize_field(data)
        
        # Reverse (from z-y-x to x-y-z)
        samples = [s for s in reversed(deform)]
        
        # Deform!
        # t0 = time.time()
        if self.forward_mapping:
            result = pirt.interp.deform_forward(data, samples)
            # print 'forward deformation took %1.3f seconds' % (time.time()-t0)
        else:
            result = pirt.interp.deform_backward(data, samples, interpolation)
            # print 'backward deformation took %1.3f seconds' % (time.time()-t0)
        
        # Make Aarray and return
        result = Aarray(result, deform.field_sampling)
        return result
    
    
    def show(self, axes=None, axesAdjust=True):
        """ show(axes=None, axesAdjust=True)
        
        Illustrates 2D deformations.
        
        It does so by creating an image of a grid and deforming it.
        The image is displayed in the given (or current) axes. 
        Returns the texture object of the grid image.
        
        Requires visvis.
        
        """
        import visvis as vv
        
        # Test dimensions
        if self.ndim != 2:
            raise RuntimeError('Show only works for 2D data.')
        
        # Create image
        shape, sampling = self.field_shape, self.field_sampling
        im = Aarray(shape, sampling, fill=0.0, dtype=np.float32)
        #
        step = 10
        mayorStep = step*5
        #
        im[1::step,:] = 1
        im[:,1::step] = 1
        im[::mayorStep,:] = 1.2
        im[1::mayorStep,:] = 1.2
        im[2::mayorStep,:] = 1.2
        im[:,::mayorStep] = 1.2
        im[:,1::mayorStep] = 1.2
        im[:,2::mayorStep] = 1.2
        
        # Deform
        im2 = self.apply_deformation(im)
        
        # Draw
        t = vv.imshow(im2, axes=axes, axesAdjust=axesAdjust)
        
        # Done
        return t
    
    
    ## Converting
    
    
    def inverse(self):
        """ inverse()
        
        Get the inverse deformation. This is only valid if the 
        current deformation is diffeomorphic. The result is always 
        a DeformationField instance.
        
        """
        
        # Identity deforms are their own inverse
        if self.is_identity:
            return self
        
        # Get deform as a deformation field
        deform = self.as_deformation_field()
        
        # Get samples
        samples = [s for s in reversed(deform)]
        
        # Get fields
        fields = []
        for field in deform:
            fields.append( interp.deform_forward(-field, samples) )
        
        # Done
        if self.forward_mapping:
            return DeformationFieldForward(*fields)
        else:
            return DeformationFieldBackward(*fields)
    
    
    def as_deformation_field(self):
        """ as_deformation_field()
        
        Obtain a deformation fields instance that represents the same
        deformation. If this is already a deformation field, returns self.
        
        """
        
        if isinstance(self, DeformationField):
            return self
        elif isinstance(self, DeformationGrid):
            # Obtain deformation in each dimension. Keep in world coordinates.
            deforms = [g.get_field() for g in self]
            # Make instance
            if self.forward_mapping:
                return DeformationFieldForward(deforms)
            else:
                return DeformationFieldBackward(deforms)
        else:
            raise ValueError('Unknown deformation class.')
    
    
    def as_other(self, other):
        """ as_other(other)
        
        Returns the deformation as a forward or backward mapping, 
        so it matches the other deformations.
        
        """
        if other.forward_mapping:
            return self.as_forward()
        else:
            return self.as_backward()
    
    
    def as_forward(self):
        """ as_forward()
        
        Returns the same deformation as a forward mapping. Returns
        the original if already in forward mapping. 
        
        """
        if self.forward_mapping:
            # Quick
            return self
        else:
            # Slow
            fields = [field for field in self.inverse()]
            return DeformationFieldForward(*fields)
    
    
    def as_forward_inverse(self):
        """ as_forward_inverse()
        
        Returns the inverse deformation as a forward mapping. Returns
        the inverse of the original if already in forward mapping. If
        in backward mapping, the data is the same, but wrapped in a
        Deformation{Field/Grid}Backward instance.
        
        Note: backward and forward mappings with the same data are
        each-others reverse.
        
        """
        if self.forward_mapping:
            # Slow
            return self.inverse()
        else:
            # Quick: same data wrapped in forward class
            if isinstance(self, DeformationGrid):
                fd = FD(self)
                deform = DeformationGridForward(fd, self.grid_sampling)
                for d in deform.ndim:
                    deform[d]._knots = self[d]._knots
                return deform
            elif isinstance(self, DeformationField):
                fields = [field for field in self]
                return DeformationFieldForward(*fields)
    
    
    def as_backward(self):
        """ as_backward()
        
        Returns the same deformation as a backward mapping. Returns
        the original if already in backward mapping. 
        
        """
        if not self.forward_mapping:
            # Quick
            return self
        else:
            # Slow
            fields = [field for field in self.inverse()]
            return DeformationFieldBackward(*fields)
    
    
    def as_backward_inverse(self):
        """ as_forward_inverse()
        
        Returns the inverse deformation as a forward mapping. Returns
        the inverse of the original if already in forward mapping. If
        in backward mapping, the data is the same, but wrapped in a
        DeformationFieldBackward instance.
        
        Note: backward and forward mappings with the same data are
        each-others reverse.
        
        """
        if not self.forward_mapping:
            # Slow
            return self.inverse()
        else:
             # Quick: same data wrapped in backward class
            if isinstance(self, DeformationGrid):
                fd = FD(self)
                deform = DeformationGridBackward(fd, self.grid_sampling)
                for d in deform.ndim:
                    deform[d]._knots = self[d]._knots
                return deform
            elif isinstance(self, DeformationField):
                fields = [field for field in self]
                return DeformationFieldBackward(*fields)



class DeformationGrid(Deformation, GridContainer):
    """ DeformationGrid(image, sampling=5)
   
    A deformation grid represents a deformation using a spline grid. 
    
    The 'grids' property returns a tuple of SplineGrid instances (one for 
    each dimension). These sub-grids can also obtained by indexing and
    are in z,y,x order.
    
    Parameters
    ----------
    image : shape-tuple, numpy-array, Aarray, or FieldDescription
        A description of the field that this grid applies to.
        The image itself is not stored, only the field's shape and 
        sampling are of interest.
    sampling : number
        The spacing of the knots in the field. (For anisotropic fields,
        the spacing is expressed in world units.)
    
    Usage
    -----
    After normal instantiation, the grid describes a field with all zeros.
    Use the From* classmethods to obtain a grid which represents the given
    values.
    
    Limitations
    -----------
    The grid can in principle be of arbitrary dimension, but this 
    implementation currently only supports 1D, 2D and 3D.
    
    """
    
    def __init__(self, *args, **kwargs):
        GridContainer.__init__(self, *args, **kwargs)
        
        # Create sub grids
        for d in range(self.ndim):
            grid = SplineGrid(*args, **kwargs)
            grid._thisDim = d
            self._grids.append(grid)
    
    
    def show(self, axes=None, axesAdjust=True, showGrid=True):
        """ show(axes=None, axesAdjust=True, showGrid=True)
        
        For 2D grids, illustrates the deformation and the knots of the grid.
        A grid image is made that is deformed and displayed in the given 
        (or current) axes. By default the positions of the underlying knots 
        are also shown using markers.
        Returns the texture object of the grid image.
        
        Requires visvis.
        
        """
        
        import visvis as vv
        
        # Show grid using base method
        t = Deformation.show(self, axes, axesAdjust)
        
        
        if showGrid:
            
            # Get points for all knots
            pp = Pointset(2)
            for gy in range(self.grid_shape[0]):
                for gx in range(self.grid_shape[1]):
                    x = (gx-1) * self.grid_sampling
                    y = (gy-1) * self.grid_sampling
                    pp.append(x, y)
        
            # Draw
            vv.plot(pp, ms='.', mc='g', ls='', axes=axes, axesAdjust=0)
        
        # Done
        return t
    
    
    ## Classmethods
    
    @classmethod        
    def from_field(cls, field, sampling, weights=None, 
                   injective=True, frozenedge=True, fd=None):
        """ from_field(field, sampling, weights=None, injective=True,
                       frozenedge=True, fd=None)
        
        Create a DeformationGrid from the given deformation field
        (z-y-x order). Also see from_field_multiscale()
        
        The optional weights array can be used to individually weight the
        field elements. Where the weight is zero, the values are not 
        evaluated. The speed can therefore be significantly improved if 
        there are relatively few nonzero elements.
        
        Parameters
        ----------
        field : list of numpy arrays
            These arrays describe the deformation field (one per dimension).
        sampling : scalar
            The sampling of the returned grid
        weights : numpy array
            This array can be used to weigh the contributions of the 
            individual elements.
        injective : bool
            Whether to prevent the grid from folding. This also penetalizes
            large relative deformations. An injective B-spline grid is
            diffeomorphic.
        frozenedge : bool
            Whether the edges should be frozen. This can help the registration
            process. Also, when used in conjunction with injective, a truly
            diffeomorphic deformation is obtained: every input pixel maps
            to a point within the image boundaries.
        fd : field
            Field description to describe the shape and sampling of the
            underlying field to be deformed.
        """
        if fd is None:
            fd = field[0]
        grid = cls(fd, sampling)
        grid._set_using_field(field, weights, injective, frozenedge)
        return grid
    
    
    @classmethod        
    def from_field_multiscale(cls, field, sampling, weights=None, fd=None):
        """ from_field_multiscale(field, sampling, weights=None, fd=None)
        
        Create a DeformationGrid from the given deformation field 
        (z-y-x order). Applies from_field_multiscale() for each
        of its subgrids.
        
        Important notice
        ----------------
        Note that this is not the best way to make a deformation, as it does
        not take aspects specific to deformations into account, such as
        injectivity, and uses addition to combine the sub-deformations instead
        of composition.
        
        See DeformationField.from_points_multiscale() for a sound alternative.
        
        Parameters
        ----------
        field : list of numpy arrays
            These arrays describe the deformation field (one per dimension).
        sampling : scalar
            The sampling of the returned grid
        weights : numpy array
            This array can be used to weigh the contributions of the 
            individual elements.
        fd : field
            Field description to describe the shape and sampling of the
            underlying field to be deformed.
        """
        if fd is None:
            fd = field[0]
        
        # Get sampling
        tmp = GridInterface(fd, 1)
        sMin, sMax = _calculate_multiscale_sampling(tmp, sampling)
        
        # Make grid
        grid = cls(fd, sMin)
        
        # Fill grids
        for d in range(grid.ndim):
            # i = grid.ndim - d - 1
            tmp = SplineGrid.from_field_multiscale(Aarray(field[d], fd.sampling),
                                                   sampling, weights)
            grid._grids[d] = tmp
        
        # Done
        return grid
    
    
    @classmethod    
    def from_points(cls, image, sampling, pp1, pp2,
                    injective=True, frozenedge=True):
        """ from_points(image, sampling, pp1, pp2,
                        injective=True, frozenedge=True)
        
        Obtains the deformation field described by the two given sets
        of corresponding points. The deformation describes moving the
        points pp1 to points pp2. Note that backwards interpolation is 
        used, so technically, the image is re-interpolated by sampling
        at the points in pp2 from pixels specified by the points in pp1.
        
        Parameters
        ----------
        image : numpy array or shape
            The image (of any dimension) to which the deformation applies.
        sampling : scalar
            The sampling of the returned grid.
        pp1 : Pointset
            The base points.
        pp2 : Pointset
            The target points.
        injective : bool
            Whether to prevent the grid from folding. This also penetalizes
            large relative deformations. An injective B-spline grid is
            diffeomorphic.
        frozenedge : bool
            Whether the edges should be frozen. This can help the registration
            process. Also, when used in conjunction with injective, a truly
            diffeomorphic deformation is obtained: every input pixel maps
            to a point within the image boundaries.
        
        """
        
        # Obtain reference points and vectors
        if cls._forward_mapping:  # using the prop on cls would get the prop, not the value!
            pp = pp1
            dd = pp2 - pp1
        else:
            pp = pp2
            dd = pp1 - pp2
        
        # Go
        grid = cls(image, sampling)
        grid._set_using_points(pp, dd, injective, frozenedge)
        return grid
    
    
    @classmethod    
    def from_points_multiscale(cls, image, sampling, pp1, pp2):
        """ from_points_multiscale(image, sampling, pp1, pp2)
        
        Obtains the deformation field described by the two given sets
        of corresponding points. The deformation describes moving the
        points pp1 to points pp2. Applies from_points_multiscale() for each
        of its subgrids.
        
        See DeformationField.from_points_multiscale() for a sound alternative.
        
        Important notice
        ----------------
        Note that this is not the best way to make a deformation, as it does
        not take aspects specific to deformations into account, such as
        injectivity, and uses addition to combine the sub-deformations instead
        of composition.
        
        Parameters
        ----------
        image : numpy array or shape
            The image (of any dimension) to which the deformation applies.
        sampling : scalar
            The sampling of the returned grid.
        pp1 : Pointset
            The base points.
        pp2 : Pointset
            The target points.
        
        """
        
        # Obtain reference points and vectors
        if cls._forward_mapping:  # using the prop on cls would get the prop, not the value!
            pp = pp1
            dd = pp2 - pp1
        else:
            pp = pp2
            dd = pp1 - pp2
        
        # Get sampling
        tmp = GridInterface(image, 1)
        sMin, sMax = _calculate_multiscale_sampling(tmp, sampling)
        
        # Make grid
        grid = cls(image, sMin)
        
        # Fill grids
        for d in range(grid.ndim):
            i = grid.ndim - d - 1
            tmp = SplineGrid.from_points_multiscale(image, sampling, pp, dd[:,i])
            grid._grids[d] = tmp
        
        # Done
        return grid
    
    
    ## Private methods to help getting/setting the grid
    
    
    def _set_using_field(self, deforms, weights=None, injective=True, frozenedge=True):
        """ _set_using_field(deforms, weights=None, injective=True, frozenedge=True)
        
        Sets the deformation by specifying deformation fields for
        each dimension (z-y-x order). Optionally, an array with
        weights can be given to weight each deformation unit.
        
        """
        
        # Check number of deforms given
        errMsg = 'Deforms must be a list/tuple/DeformationField with a deform for each dim.'
        if not isinstance(deforms, (tuple, list, DeformationField)):
            raise ValueError(errMsg)
        elif len(deforms) != self.ndim:
            raise ValueError(errMsg)
        
        # Apply using SplineGrid's method
        for d in range(self.ndim):
            #i = self.ndim - d - 1
            self[d]._set_using_field(deforms[d], weights)
        
        # Diffeomorphic constraints
        if injective:
            self._unfold(injective)
        if frozenedge:
            self._freeze_edges()
    
    
    def _set_using_points(self, pp, dd, injective=True, frozenedge=True):
        """ _set_using_points(pp, dd, injective=True, frozenedge=True)
        
        Deform the lattices according to points pp with the deformations 
        defined in dd.
        
        pp is the position to apply the deformation, dd is the relative
        position to sample the pixels from.
        
        By default performs folding and shearing prevention to obtain 
        a grid that is injective (i.e. can be inverted).
        
        """
        
        # Apply using SplineGrid's method
        for d in range(self.ndim):
            i = self.ndim - d - 1
            self[d]._set_using_points(pp, dd[:,i])
        
        # Diffeomorphic constraints
        if injective:
            self._unfold(injective)
        if frozenedge:
            self._freeze_edges()
    
    
    def _unfold(self, factor):
        """ _unfold(factor)
        
        Prevent folds in the grid, by putting a limit to the values
        that the knots may have.
        
        The factor determines how smooth the deformation should be. Zero
        is no deformation, 1 is *just* no folding. Better use a value of
        0.9 at the highest, to account for numerical errors.
        
        Based on:
        Choi, Yongchoel, and Seungyong Lee. 2000. "Injectivity conditions of
        2d and 3d uniform cubic b-spline functions". GRAPHICAL MODELS 62: 2000.
        
        But we apply a smooth mapping rather than simply truncating the values.
        Give a negative factor to use the truncated method
        
        """
        
        # Check factor
        mode = 2
        if factor is False:
            mode = 0
            return # Do not unfold
        elif factor is True:
            factor = 0.9 # Default
        elif factor < 0:
            mode = 1
            factor = - factor
        
        # Get K factor
        if self.ndim == 1:
            K = 2.0 # not sure about this
        if self.ndim == 2:
            K = 2.046392675
        elif self.ndim == 3:
            K = 2.479472335
        else:
            return
        
        # Calculate limit, see Choi et al. 2000, Lee at al. 1996
        limit = (1.0/K) * self.grid_sampling * factor
        
        for d in range(self.ndim):
            
            # Get knots 
            knots = self[d].knots.ravel()
            
            if mode == 1:
                # Hard limit (as proposed by Lee et al.)
                I, = np.where(np.abs(knots)>limit)
                knots[I] = limit * np.sign(knots[I])
            elif mode == 2:
                # Apply smooth limit, see example in scripts directory
                f = np.exp(-np.abs(knots)/limit)
                knots[:] = limit * (f-1) * -np.sign(knots)
            else:
                pass
    
    def _freeze_edges(self):
        """ _freeze_edges()
        
        Freeze the outer knots of the grid such that the deformation is
        zero at the edges of the image.
        
        This sets three rows of knots to zero at the top/left of the
        grid, and four rows at the bottom/right. This is because at the
        left there is one knot extending beyond the image, while at the
        right there are two.
        
        """
        
   
        def get_t_factor(grid, d):
            field_edge = (grid.field_shape[d]-1) * grid.field_sampling[d] 
            grid_edge = (grid.grid_shape[d]-4) * grid.grid_sampling
            return 1.0 - (field_edge - grid_edge) / grid.grid_sampling
        
        for d in range(len(self)):
            grid = self[d]
            
            # Check if grid is large enough
            if grid._knots.shape[d] < 6:
                grid._knots[:] = 0
                continue
            
            if d==0:
                # top/left
                # k1, k2 = knots[0], knots[1]
                grid._knots[0] = 0
                grid._knots[1] = - 0.25*grid._knots[2]
                
                # Get t factor and coefficients
                t = get_t_factor(grid, d)
                c1, c2, c3, c4 = interp.get_cubic_spline_coefs(t, 'B')
                
                # bottom/right
                grid._knots[-3] = (1-t)*grid._knots[-3]
                grid._knots[-1] = 0
                k3, k4 = grid._knots[-3], grid._knots[-4]
                grid._knots[-2] = -(k3*c3 + k4*c4)/c2
            
            elif d==1:
                # top/left
                grid._knots[:,0] = 0
                grid._knots[:,1] = - 0.25*grid._knots[:,2]
                
                # Get t factor and coefficients
                t = get_t_factor(grid, d)
                c1, c2, c3, c4 = interp.get_cubic_spline_coefs(t, 'B')
                
                # bottom/right
                grid._knots[:,-3] = (1-t)*grid._knots[:,-3]
                grid._knots[:,-1] = 0
                k3, k4 = grid._knots[:,-3], grid._knots[:,-4]
                grid._knots[:,-2] = -(k3*c3 + k4*c4)/c2
            
            elif d==2:
                # top/left
                grid._knots[:,:,0] = 0
                grid._knots[:,:,1] = - 0.25*grid._knots[:,:,2]
                
                # Get t factor and coefficients
                t = get_t_factor(grid, d)
                c1, c2, c3, c4 = interp.get_cubic_spline_coefs(t, 'B')
                
                # bottom/right
                grid._knots[:,:,-3] = (1-t)*grid._knots[:,:,-3]
                grid._knots[:,:,-1] = 0
                k3, k4 = grid._knots[:,:,-3], grid._knots[:,:,-4]
                grid._knots[:,:,-2] = -(k3*c3 + k4*c4)/c2



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
                if not isinstance(field, Aarray):
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
            field2 = pirt.interp.resize(field1, fd.shape, 3, 'C', prefilter=False, extra=False)
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
                #     pirt.interp.fix_samples_edges(defField)
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
        pp1 : Pointset
            The base points.
        pp2 : Pointset
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
        sMin, sMax = _calculate_multiscale_sampling(tmp, sampling)
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
                elif isinstance(residu, vv.Pointset):
                    # print('Warning, vv.Pointset is used ...')
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


class DeformationIdentity(Deformation):
    """ Abstract identity deformation. It is not a grid nor a field, nor
    is it forward or backward mapping.
    
    It is nothing more than a simple tool to initialize a deformation with.
    """
    pass


class DeformationGridForward(DeformationGrid):
    """ A deformation grid representing a forward mapping; to create the 
    deformed image, the pixels are mapped to their new locations. 
    """
    _forward_mapping = True


class DeformationGridBackward(DeformationGrid):
    """ A deformation grid representing a backward mapping; the field 
    represents where the pixels in the deformed image should be sampled to
    in the original image.
    """
    _forward_mapping = False


class DeformationFieldForward(DeformationField):
    """ A deformation field representing a forward mapping; to create the 
    deformed image, the pixels are mapped to their new locations. 
    """
    _forward_mapping = True


class DeformationFieldBackward(DeformationField):
    """ A deformation field representing a backward mapping; the field 
    represents where the pixels in the deformed image should be sampled to
    in the original image.
    """
    _forward_mapping = False
