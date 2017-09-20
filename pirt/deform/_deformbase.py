import time  # noqa

import numpy as np

from .. import Aarray
from .. import interp
from ..splinegrid import FD


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
                field = grid2.get_field_in_samples(sampleLocations)
                field = Aarray(field1+field, def1.field_sampling)
                fields.append(field)
        elif isinstance(def2, DeformationField):
            # Composition with a field introduces interpolation artifacts
            for d in range(def1.ndim):
                field1 = def1[d]
                field2 = def2[d]
                # linear or cubic. linear faster and I've not seen cubic doing better
                field = interp.warp(field2, sampleLocations, 'linear')
                field = Aarray(field1+field, def1.field_sampling)
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
        return interp.make_samples_absolute(deltas) 
    
    
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
        
        The points pp should be a point set (x-y-z order).
        
        """
        assert isinstance(pp, np.ndarray) and pp.ndim == 2
        if isinstance(self, DeformationGrid):
            return self[d].get_field_in_points(pp)
        elif isinstance(self, DeformationField):
            data = self[d]
            samples = []
            samling_xyz = [s for s in reversed(self.field_sampling)]
            for i in range(self.ndim):
                s = pp[:,i] / samling_xyz[i]
                samples.append(s)
            return interp.interp(data, samples, order=interpolation)
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
            result = interp.deform_forward(data, samples)
            # print 'forward deformation took %1.3f seconds' % (time.time()-t0)
        else:
            result = interp.deform_backward(data, samples, interpolation)
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


# Import at the end to work around recursion
from ._deformgrid import DeformationGrid
from ._deformfield import DeformationField
from ._subs import (DeformationIdentity,
                    DeformationGridForward, DeformationGridBackward,
                    DeformationFieldForward, DeformationFieldBackward)
