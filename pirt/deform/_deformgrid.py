import numpy as np

from .. import PointSet, Aarray
from .. import interp
from ..splinegrid import GridInterface, GridContainer, SplineGrid
from ..splinegrid import calculate_multiscale_sampling

from ._deformbase import Deformation


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
            pp = PointSet(2)
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
        sMin, sMax = calculate_multiscale_sampling(tmp, sampling)
        
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
        pp1 : PointSet, 2D ndarray
            The base points.
        pp2 : PointSet, 2D ndarray
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
        assert isinstance(pp1, np.ndarray) and pp1.ndim == 2
        assert isinstance(pp2, np.ndarray) and pp2.ndim == 2
        
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
        pp1 : PointSet, 2D ndarray
            The base points.
        pp2 : PointSet, 2D ndarray
            The target points.
        
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
        
        # Get sampling
        tmp = GridInterface(image, 1)
        sMin, sMax = calculate_multiscale_sampling(tmp, sampling)
        
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
        assert isinstance(pp, np.ndarray) and pp.ndim == 2
        
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


# Import at the end to work around recursion
from ._deformfield import DeformationField
