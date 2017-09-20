import numpy as np

from .. import PointSet, Aarray
from . import _splinegridfuncs


## Helper classes and functions


class FieldDescription:
    """ FieldDescription(*args)
    
    Describes the dimensions of a field (i.e. Aarray). It stores
    the following properties: shape, sampling, origin

    This class can for example be used to instantiate a new grid 
    without requiring the actual field.
    
    This class can be instantiated with a shape and sampling tuple, or with
    any object that describes a field in a way that we know of 
    (e.g. SplineGrid and DeformationField instances).
    
    Examples
    --------
      * FieldDescription(shape, sampling=None, origin=None)
      * FieldDescription(grid)
      * FieldDescription(Aarray)
      * FieldDescription(np.ndarray) # assumes unit sampling and zero origin
    
    """
    
    def __init__(self, shape, sampling=None, origin=None):
        
        if hasattr(shape, 'shape') and isinstance(shape.shape, (list, tuple)):
            # Field given
            field = shape
            shape, sampling, origin = field.shape, None, None
            if hasattr(field, 'sampling'):
                sampling = field.sampling
            if hasattr(field, 'origin'):
                origin = field.origin
        
        elif (  hasattr(shape, 'field_shape') and 
                isinstance(shape.field_shape, (list, tuple))    ):
            # Grid or deformation field
            field = shape
            shape, sampling, origin = field.field_shape, None, None
            if hasattr(field, 'field_sampling'):
                sampling = field.field_sampling
            if hasattr(field, 'field_origin'):
                origin = field.field_origin
        
        # Init
        self._defined_samping = False
        self._defined_origin = False
        
        # Check and set shape
        if isinstance(shape, (list, tuple)):
            self._shape = tuple([int(s) for s in shape])
        else:
            raise TypeError('Invalid argument for FieldDescription')
        
        # Check and set sampling
        if isinstance(sampling, (list, tuple)):
            self._defined_samping = True
            self._sampling =  tuple([float(s) for s in sampling])
        elif sampling is None:
            self._sampling = tuple([1.0 for i in self.shape])
        else:
            raise TypeError('Invalid sampling for FieldDescription')
        
        # Check and set origin
        if isinstance(origin, (list, tuple)):
            self._defined_origin = True
            self._origin =  tuple([float(s) for s in origin])
        elif origin is None:
            self._origin = tuple([0.0 for i in self.shape])
        else:
            raise TypeError('Invalid origin for FieldDescription')
    
    
    @property
    def ndim(self):
        """ The number of dimensions of the field. 
        """
        return len(self._shape)
    
    @property
    def shape(self):
        """ The shape of the field.
        """
        return self._shape
    
    @property
    def sampling(self):
        """ The sampling between the pixels of the field.
        """
        return self._sampling
    
    @property
    def origin(self):
        """ The origin (spatial offset) of the field.
        """
        return self._origin
    
    @property
    def defined_sampling(self):
        """ Whether the sampling was explicitly defined.
        """
        return self._defined_samping
    
    @property
    def defined_origin(self):
        """ Whether the origin was explicitly defined.
        """
        return self._defined_origin

# Crate shot-named alias
FD = FieldDescription


def calculate_multiscale_sampling(grid, sampling):
    """ calculate_multiscale_sampling(grid, sampling)
    Calculate the minimal and maximal sampling from user input.
    """
    
    if isinstance(sampling, (float, int)):
        # Find maximal sampling for this grid
        
        # Set min and init max
        sMin = sMax = sampling
        
        # Determine minimal sampling
        tmp = [sh*sa for sh,sa in zip(grid.field_shape, grid.field_sampling)]
        minimalSampling = max(tmp)
        
        # Calculate max
        while sMax < minimalSampling:
            sMax *= 2
    
    elif isinstance(sampling, (list,tuple)) and len(sampling)==2:
        # Two values given
        
        # Set min and init max
        sMin = sMax = min(sampling)
        
        # Increase sMax with powers of two untill we reach the given value
        sMax_ = max(sampling)
        while sMax < sMax_:
            sMax *= 2
        
        # Select closest level
        if abs(sMax_ - sMax / 2) < abs(sMax_ - sMax):
            sMax /= 2
    
    else:  # nocov
        raise ValueError('For multiscale, sampling must have two values.')
    
    
    # Done
    return sMin, sMax


## The grid classes


class GridInterface:
    """ GridInterface(field, sampling=5)
    
    Abstract class to define the interface of a spline grid.
    Implemented by the Grid and GridContainer classes.
    
    This class provides some generic methods and properties for grids
    in general. Most importantly, it handles initialization of the
    desctiption of the grid (dimensionality, shape and sampling of the 
    underlying field, and the shape and sampling of the grid itself).
    
    Parameters
    ----------
    field : shape-tuple, numpy-array, Aarray, or FieldDescription
        A description of the field that this grid applies to.
        The field itself is not stored, only the field's shape and 
        sampling are of interest.
    sampling : number
        The spacing of the knots in the field. (For anisotropic fields,
        the spacing is expressed in world units.)
    
    """
    
    def __init__(self, field, sampling):
        
        # Sets:
        # * _field_shape
        # * _field_sampling
        # * _grid_shape
        # * _grid_sampling
        
        # Check field
        if isinstance(field, FieldDescription):
            self._field_shape = field.shape
            self._field_sampling = field.sampling
        elif hasattr(field, 'sampling'):
            self._field_shape = field.shape
            self._field_sampling = field.sampling
        elif isinstance(field, np.ndarray):
            self._field_shape = field.shape
            self._field_sampling = [1.0 for i in field.shape]
        elif isinstance(field, (list, tuple)):
            for el in field:
                if not isinstance(el, int):
                    raise ValueError('Shape must be a list/tuple of integers.')
            self._field_shape = [el for el in field]
            self._field_sampling = [1.0 for i in field]
        else:
            raise ValueError('field should be a numpy array, Aarray, or shape-tuple.')
        
        # Set spacing between knots in world units
        self._grid_sampling = sampling
        
        # Calculate number of rows and cols in the current lattice
        # +3 because we pad the field (we always pad extra!)
        # +1 because the first row AND last row contains a knot
        grid_sampling_in_pixels = self.grid_sampling_in_pixels
        self._grid_shape = []
        for d in range(self.ndim):
            max_field_index = self._field_shape[d]-1
            tmp = int(max_field_index / grid_sampling_in_pixels[d]) + 1 + 3
            self._grid_shape.append(tmp)
    
    ## Dimensions of field and grid
    
    @property
    def ndim(self):
        """ The number of dimensions of this grid. 
        """
        return len(self._field_shape)
    
    @property
    def field_shape(self):
        """ The shape of the underlying field. (i.e. the size in each dim.)
        """
        return tuple(self._field_shape)
    
    @property
    def field_sampling(self):
        """ For each dim, the sampling of the field, i.e. the distance
        (in world units) between pixels/voxels (all 1's if isotropic).
        """
        return tuple(self._field_sampling)
    
    @property
    def grid_shape(self):
        """ The shape of the grid. (i.e. the size in each dim.)
        """
        return tuple(self._grid_shape)
    
    @property
    def grid_sampling(self):
        """ A *scalar* indicating the spacing (in world units) between the knots.
        """
        return self._grid_sampling
    
    @property
    def grid_sampling_in_pixels(self):
        """ For each dim, the spacing (in sub-pixels) between the knots.
        A dimension that has a low field sampling will have a high grid
        sampling in pixels (since the pixels are small, more fit between
        two knots).
        """
        return tuple([self._grid_sampling/float(i) for i in self._field_sampling])
    
    
    ## Methods to obtain derived grids
    
    def copy(self):
        """ copy()
        
        Return a deep copy of the grid.
        
        """
        
        # Get new grid
        fd = FieldDescription(self)
        newGrid = self.__class__(fd, self.grid_sampling)
        
        if isinstance(self, GridContainer):
            # Copy all subgrids
            newGrid._map('copy', self)
        elif isinstance(self, SplineGrid):
            # Copy knots array
            newGrid._knots = self.knots.copy() # note the copy
        else:  # nocov
            raise ValueError('Cannot copy: unknown grid class.')
        
        # Done
        return newGrid
    
    
    def refine(self):
        """ refine()
        
        Refine the grid, returning a new grid instance (of the same type)
        that represents the same field, but with half the grid_sampling.
        
        """
        
        # Create new grid
        newSampling = self.grid_sampling / 2.0
        fieldDes = FieldDescription(self)
        newGrid = self.__class__(fieldDes, newSampling)
        
        if isinstance(self, GridContainer):
            # Refine each of the subgrids and put in newGrid
            newGrid._map('refine', self)
        
        elif isinstance(self, SplineGrid):
            
            # Get knots array for the new grid
            newKnots = self._refine(self.knots)
            
            # Apply calculated knots, crop if necessary
            NS = newGrid.grid_shape
            if NS != newKnots.shape:
                pass
                #print('shape mismatch:', NS, newKnots.shape)
            #
            if self.ndim == 1:
                newGrid._knots = newKnots[:NS[0]]
            elif self.ndim == 2:
                newGrid._knots = newKnots[:NS[0], :NS[1]]
            elif self.ndim == 3:
                newGrid._knots = newKnots[:NS[0], :NS[1], :NS[2]]
        
        else:  # nocov
            raise ValueError('Cannot refine: unknown grid class.')
        
        # Done
        return newGrid
    
    
    def add(self, other_grid):
        """ add(other_grid)
        
        Create a new grid by adding this grid and the given grid.
        
        """
        
        # Check
        if not (self.field_shape == other_grid.field_shape and 
                self.field_sampling == other_grid.field_sampling):  # nocov
            raise ValueError('Can only add grids that have the same shape and sampling.') 
        
        # Create empty grid with same shape as the other grid.
        fd = FieldDescription(self)
        newGrid = self.__class__(fd, self.grid_sampling)
        
        if isinstance(self, GridContainer):
            # Add each of the subgrids and put in newGrid
            newGrid._map('add', self, other_grid)
        elif isinstance(self, SplineGrid):
            # Add knots arrays
            newGrid._knots = self.knots + other_grid.knots
        else:  # nocov
            raise ValueError('Cannot add: unknown grid class.')
        
        # Done
        return newGrid
    
    
#     # todo: result_grid?
#     def compose(self, other_grid, result_grid=None):
#         """ compose(other_grid, result_grid=None)
#         
#         Compose a new grid by calculating the field-values (of this grid)
#         at the knots of the given grid, and adding them to the values of 
#         knots of the given grid.
#         
#         This method does not require the two grids to have the same shape
#         or sampling.
#         
#         If result_grid is given, the result is written to it. The resulting
#         grid is alwats returned.
#         
#         If this grid represents a deformation
#         -------------------------------------
#         If both this grid and the given grid are C2 continuous and
#         injective, the result is also C2 continuous and injective (in
#         contrast two adding to grids). See Choi et al. 2000.
#         
#         """
#         
#         # Create empty grid with same shape as the other grid.
#         if result_grid is None:
#             fd = FieldDescription(self)
#             newGrid = self.__class__(fd, self.grid_sampling)
#         else:
#             newGrid = result_grid
#         
#         if isinstance(self, GridContainer):
#             # Compose each of the subgrids and put in newGrid
#             newGrid._map('compose', self, other_grid)
#         elif isinstance(self, SplineGrid):
#             # Fill the new grid
#             _splinegridfuncs.get_field_grid(self, newGrid)
#             # Add delta to given grid
#             newGrid._knots += other_grid.knots
#         else:
#             raise ValueError('Cannot compose: unknown grid class.')
#         
#         # Done
#         return newGrid
    
    
    def resize_field(self, new_shape=None): # must be a keyword argument
        """ resize_field(new_shape)
        
        Create a new grid, where the underlying field is reshaped. The 
        field is still the same; it only has a different shape and sampling.
        
        The parameter new_shape can be anything that can be converted 
        to a FieldDescription instance.
        
        Note that the knots arrays are shallow copied (which makes
        this function very efficient). 
        
        """
        
        # Get description
        fd = FieldDescription(new_shape)
        
        # Create new grid
        newGrid = self.__class__(fd, self.grid_sampling)
        
        if isinstance(self, GridContainer):
            # reshape all subgrids
            newGrid._map('resize_field', self, new_shape=fd)
        elif isinstance(self, SplineGrid):
            # Simply copy the knots array
            newGrid._knots = self.knots
        else:  # nocov
            raise ValueError('Cannot resize_field: unknown grid class.')
        
        # Done
        return newGrid
    
    
    ## Multiscale composition
    
    
    @classmethod
    def _multiscale(cls, setResidu, getResidu, field, sampling):
        """ _multiscale(setResidu, getResidu, field, sampling)
        
        General method for multiscale grid formation. from_field_multiscale()
        and from_points_multiscale() use this classmethod by each supplying 
        appropriate setResidu and getResidu functions.
        
        """
        
        # Set grid class
        GridClass = cls
        
        # Get sampling
        tmp = GridInterface(field, 1)
        sMin, sMax = calculate_multiscale_sampling(tmp, sampling)
        s, sRef = sMax, sMin*0.9
        
        # Init refined grid (starts with highest sampling)
        grid = GridClass(field, s)
        
        # Init residu
        residu = getResidu()
        
        # grid: working grid
        # gridRef: refined grid
        # gridAdd: grid to add to working-grid
        
        while s > sRef:
            
            # Create addGrid using the residual values
            gridAdd = GridClass(field, s)        
            setResidu(gridAdd, residu)
            
            # Create grid by combining refined grid of previous pass and
            # the gridAdd. 
            grid = grid.add(gridAdd)
            
            # Prepare for next iter
            s /= 2.0
            
            if s > sRef: # last round
                
                # Refine grid            
                grid = grid.refine()
                
                # Get current values in the field and calculate residual
                # Use refGrid, as it does not *exactly* represent the
                # same field as grid.
                residu = getResidu(grid)
        
        # Done
        return grid



class GridContainer(GridInterface):
    """ GridContainer(field, sampling=5)
    
    Abstract base class that represents multiple SplineGrid instances.
    Since each SplineGrid instance describes a field of scalar values,
    the GridContainer can be used to describe vectors/tensors. Examples
    are color and 2D/3D deformations.
    
    The implementing class should:
      * instantiate SplineGrid instances and append them to '_grids'
      * implement methods to set the grid accordingly, probably using
        classmethods such as from_points, from_field, etc.
    
    """
    
    def __init__(self, *args, **kwargs):
        GridInterface.__init__(self, *args, **kwargs)
        
        # Init list of sub grids
        self._grids = []
    
    
    def __len__(self):
        return len(self._grids)
    
    def __getitem__(self, item):
        if isinstance(item, int):
            if item>=0 and item<len(self._grids):
                return self._grids[item]
            else:
                raise IndexError("Grid index out of range.")
        else:
            raise IndexError("DeformationGrid only supports integer indices.")
    
    
    def __iter__(self):
        return self.grids.__iter__()
    
    
    @property
    def grids(self):
        """ A tuple of subgrids.
        """
        return tuple(self._grids)
    
    
    def _map(self, method, source, *args, **kwargs):
        """ _map(self, method, source, *args, **kwargs)
        
        Set the knots of the sub-grids by mapping a method on the
        subgrids of a source grid, optionally with additional grid
        arguments.
        
        Examples
        --------
        newGrid._map('copy', sourceGrid)
        newGrid._map('add', sourceGrid1, sourceGrid2)
        
        """
        for d in range(len(self.grids)):
            # Get bound function object
            fun = getattr(source[d], method)
            # Index args
            args2 = [arg[d] for arg in args]
            # Call
            self[d]._knots = fun(*args2, **kwargs).knots
    


class SplineGrid(GridInterface):
    """ SplineGrid(field, sampling=5)
    
    A SplineGrid is a representation of a scalar field in N 
    dimensions. This field is represented in a sparse way using 
    knots, which are distributed in a uniform grid.
    
    The manner in which these knots describe the field depends
    on the underlying spline being used, which is a Cubic 
    B-spline. This spline adopts a shape corresponding to minimum
    bending energy, which makes them the preferred choice for many 
    interpolation tasks. (Earlier versions of Pirt allowed setting the
    spline types, but to make things easier, and because the B-spline is
    the only sensible choice, this option was removed.)
    
    Parameters
    ----------
    field : shape-tuple, numpy-array, Aarray, FieldDescription
        A description of the field that this grid applies to.
        The field itself is not stored, only the field's shape and 
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
        GridInterface.__init__(self, *args, **kwargs)
        
        # Init knots. The knots array is created by the convenience methods.
        # If the knots property is obtained, 
        self._knots = None
        
        # Private variable to indicate what dimension this Grid represents
        # when used in a DeformationGrid
        self._thisDim = 0
    
    
    def show(self, axes=None, axesAdjust=True, showGrid=True):
        """ show(axes=None, axesAdjust=True, showGrid=True)
        
        For 2D grids, shows the field and the knots of the grid.
        The image is displayed in the given (or current) axes. By default
        the positions of the underlying knots are also shown using markers.
        Returns the texture object of the field image.
        
        Requires visvis.
        
        """
        import visvis as vv
        
        # Test dimensions
        if self.ndim != 2:
            raise RuntimeError('Show only works for 2D data.')
        
        # Get field
        field = Aarray(self.get_field(), self.field_sampling)
        
        # Get points for all knots
        pp = PointSet(2)
        for gy in range(self.grid_shape[0]):
            for gx in range(self.grid_shape[0]):
                x = (gx-1)* self.grid_sampling
                y = (gy-1)* self.grid_sampling
                pp.append(x,y)
        
        # Draw
        if showGrid:
            vv.plot(pp, ms='.', mc='g', ls='', axes=axes, axesAdjust=axesAdjust)
            return vv.imshow(field, axes=axes)
        else:
            return vv.plot(pp, ms='.', mc='g', ls='', axes=axes, axesAdjust=axesAdjust)
    
    
    ## Getters
    
    @property
    def knots(self):
        """ A numpy array that represent the values of the knots.
        """
        if self._knots is None:
            self._knots = np.zeros(self._grid_shape, dtype=np.float64)
        return self._knots 
    
    
    def get_field(self):
        """ get_field()
        
        Obtain the full field that this grid represents. 
        
        """
        field = _splinegridfuncs.get_field(self)
        return Aarray(field, self.field_sampling)
    
    
    def get_field_in_points(self, pp):
        """ get_field_in_points(pp)
        
        Obtain the field in the specied points (in world coordinates).
        
        """
        assert isinstance(pp, np.ndarray) and pp.ndim == 2
        
        # Throw away points that are not inside the field
        pp, tmp = self._select_points_inside_field(pp)
        
        # Sample field
        return _splinegridfuncs.get_field_sparse(self, pp)
    
    
    def get_field_in_samples(self, samples):
        """ get_field_in_samples(pp)
        
        Obtain the field in the specied samples (a tuple with pixel
        coordinates, in x-y-z order).
        
        """
        if not isinstance(samples, (tuple, list)):  # nocov
            raise ValueError('Samples must be list or tuple.')
        if len(samples) != self.ndim:  # nocov
            raise ValueError('Samples must contain one element per dimension.')
        
        # Sample field
        return _splinegridfuncs.get_field_at(self, samples)
    
    
    ## Classmethods to get a grid
    
    @classmethod    
    def from_field(cls, field, sampling, weights=None):
        """ from_field(field, sampling, weights=None)
        
        Create a SplineGrid from a given field. Note that the smoothness 
        of the grid and the extent to which the grid follows the given values. 
        Also see from_field_multiscale()
        
        The optional weights array can be used to individually weight the
        field elements. Where the weight is zero, the values are not 
        evaluated. The speed can therefore be significantly improved if 
        there are relatively few nonzero elements.
        
        Parameters
        ----------
        field : numpy array or shape
            The field to represent with this grid.
        sampling : scalar
            The sampling of the returned grid.
        weights : (optional) numpy array
            This array can be used to weigh the contributions of the 
            individual elements.
        
        """
        grid = SplineGrid(field, sampling)
        grid._set_using_field(field, weights)
        return grid
    
    
    @classmethod    
    def from_field_multiscale(cls, field, sampling, weights=None):
        """ from_field_multiscale(field, sampling, weights=None)
        
        Create a SplineGrid from the given field. By performing a 
        multi-scale approach the grid adopts a minimal bending to 
        conform to the given field.
        
        The optional weights array can be used to individually weight the
        field elements. Where the weight is zero, the values are not 
        evaluated. The speed can therefore be significantly improved if 
        there are relatively few nonzero elements.
        
        Parameters
        ----------
        field : numpy array or shape
            The field to represent with this grid.
        sampling : scalar
            The sampling of the returned grid.
        weights : (optional) numpy array
            This array can be used to weigh the contributions of the 
            individual elements.
        
        Notes
        -----
        The algorithmic is based on:
        Lee, Seungyong, George Wolberg, and Sung Yong Shin. 1997. 
        "Scattered Data Interpolation with Multilevel B-splines". 
        IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS 3 (3): 228-244.
        
        """
        
        def setR(gridAdd, residu):
            gridAdd._set_using_field(residu, weights)
        
        def getR(gridRef=None):
            if gridRef is None:
                return field
            else:
                return field - gridRef.get_field()
        
        return cls._multiscale(setR, getR, field, sampling)
    
    
    @classmethod    
    def from_points(cls, field, sampling, pp, values):
        """ from_points(field, sampling, pp, values)
        
        Create a SplineGrid from the values specified at a set of 
        points. Note that the smoothness of the grid and the extent to 
        which the grid follows the given values. Also see 
        from_points_multiscale()
        
        Parameters
        ----------
        field : numpy array or shape
            The image (of any dimension) to which the grid applies.
        sampling : scalar
            The sampling of the returned grid.
        pp : PointSet, 2D ndarray
            The positions (in world coordinates) at which the values are given.
        values : list or numpy array
            The values specified at the given positions.
        
        """
        assert isinstance(pp, np.ndarray) and pp.ndim == 2
        grid = SplineGrid(field, sampling)
        grid._set_using_points(pp, values)
        return grid
    
    
    @classmethod    
    def from_points_multiscale(cls, field, sampling, pp, values):
        """ from_points_multiscale(field, sampling, pp, values)
        
        Create a SplineGrid from the values specified at a set of 
        points. By performing a multi-scale approach the grid adopts a 
        minimal bending to conform to the given values.
        
        Parameters
        ----------
        field : numpy array or shape
            The image (of any dimension) to which the grid applies.
        sampling : scalar
            The sampling of the returned grid.
        pp : PointSet, 2D ndarray
            The positions (in world coordinates) at which the values are given.
        values : list or numpy array
            The values specified at the given positions.
        
        Notes
        -----
        The algorithmic is based on:
        Lee, Seungyong, George Wolberg, and Sung Yong Shin. 1997. 
        "Scattered Data Interpolation with Multilevel B-splines". 
        IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS 3 (3): 228-244.
        
        """
        assert isinstance(pp, np.ndarray) and pp.ndim == 2
        
        def setR(gridAdd, residu):
            gridAdd._set_using_points(pp, residu)
        
        def getR(gridRef=None):
            if gridRef is None:
                return values
            else:
                return values - gridRef.get_field_in_points(pp)
        
        return cls._multiscale(setR, getR, field, sampling)
    
    
    ## Private methods to help getting/setting the grid
    
    
    def _select_points_inside_field(self, pp, values=None):
        """ _select_points_inside_field(self, pp, values=None)
        
        Selects the points which lay inside the field, discarting the
        outliers.
        
        When values is given, returns a tuple (pp, values) with the
        new pointset and values array. If values is not given, sets
        the outlier points in pp to the origin.
        
        """
        assert isinstance(pp, np.ndarray) and pp.ndim == 2
        
        # Keep original for debugging
        ppo = pp  # noqa
        
        # Get reversed field shape
        tmp = [sh*sa for sh,sa in zip(self.field_shape, self.field_sampling)]
        field_shape_R = [i for i in reversed(tmp)]
        
        # Remove points outside the field
        for d in range(self.ndim):
            I, = np.where( (pp[:,d]>=0) & (pp[:,d]<field_shape_R[d]) )
            if len(I) < len(pp):
                print('WARNING: some points were outside the field boundaries.')
                if values is not None:
                    pp = pp[I]
                    values = values[I]
                else:
                    I, = np.where( (pp[:,d]<0) | (pp[:,d]>=field_shape_R[d]) )
                    pp[I,:] = 0.0
        
        # Done
        return pp, values
    
    
    def _set_using_field(self, field, weights=None):
        """ _set_using_field(field, weights=None)
        
        Set the grid using an existing field, optionally with weighting.
        
        """
        
        # If not given, make unit weights
        if weights is None:
            weights = np.ones_like(field)
        
        # Go
        _splinegridfuncs.set_field(self, field, weights)
    
    
    def _set_using_points(self, pp, values):
        """ _set_using_points(pp, values)
        
        Set the grid using sparse data, defined at the points in pp.
        
        """
        assert isinstance(pp, np.ndarray) and pp.ndim == 2
        
        # Make sure values is an array if a list is given
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        
        # Throw away points that are not inside the field
        pp, values = self._select_points_inside_field(pp, values)
        
        # Go
        _splinegridfuncs.set_field_sparse(self, pp, values)
    
    
    def _refine(self, knots):
        """ _refine(knots)
        
        Workhorse method to refine the knots array.
        
        Designed for B-splines. For other splines, this method introduces
        errors; the resulting grid does not exactly represent the original.
        
        Refines the grid to a new grid (newGrid). Let grid have
        (n+3)*(m+3) knots, then newGrid has (2n+3)*(2m+3) knots (sometimes
        the grid needs one row less, this is checked in the Refine() method). 
        In both grids, the second knot lays exactly on the first pixel of the 
        image. Below is an illustration of a few knots:

        ( )   ( )   ( )
            x  x  x  x                      ( ): knots of grid
        ( ) x (x) x (x) ------------         x : knots of newGrid
            x  x  x  x
              |            image
              |

        Lee tells on page 7 of "Lee et al. 1997 - Scattered Data Interpolation
        With Multilevel B-splines" that there are several publications on how
        to create a lattice from another lattice in such a way that they
        describe the same deformation. Our case is relatively simple because we
        have about the same lattice, but with a twice as high accuracy. What we
        do here is based on what Lee says on page 7 of his article. Note that
        he handles the indices a bit different as I do.

        For each knot in the grid we update four knots in new grid. The indexes
        of the knots to update in newGrid are calculated using a simple formula
        that can be derived from the illustration shown below: For each knot in
        grid ( ) we determine the 4 x's (in newGrid) from the 0's.
          0      0       0

          0     (x)  x   0
                 x   x   
          0      0       0 

        We can note a few things. 
          * the loose x's are calculated using 2 neighbours in each dimension.
          * the x's insided ( ) are calculated using 3 neighbours in each dim.
          * (Knots can also be loose in one dim and not loose in another.)
          * the newGrid ALWAYS has its first knot between grid's first and 
            second knot. 
          * newGrid can have a smaller amount of rows and/or cols than you
            would think. According to Lee the newGrid has 2*(lat.rows-3)+3
            rows, but in our case this is not necessarily true. The image does
            not exactly fit an integer amount of knots, we thus need one knot
            extra. But when we go from a course lattice to a finer one, we 
            might need one row/col of knots less.
        
        """
        
        # For each dimension, refine the grid with a factor of two
        for d in range(knots.ndim):
            
            # Obtain reference knots
            if d == 0:
                knots1, knots2  = knots[:-1], knots[1:]
                knots3, knots4  = knots[:-2], knots[2:]
                knots5          = knots[1:-1]
            elif d == 1:
                knots1, knots2  = knots[:,:-1], knots[:,1:]
                knots3, knots4  = knots[:,:-2], knots[:,2:]
                knots5          = knots[:,1:-1]
            elif d == 2:
                knots1, knots2  = knots[:,:,:-1], knots[:,:,1:]
                knots3, knots4  = knots[:,:,:-2], knots[:,:,2:]
                knots5          = knots[:,:,1:-1]
            
            # Calculate edge knots (knots between original knots)
            eknots = 0.5 * (knots1 + knots2)
            
            # Calculate vertex knots (knots on original knots)
            vknots = 0.125 * (knots3 + 6*knots5 + knots4)
            
            # Init new knots array
            shape = [s for s in knots.shape]
            shape[d] = eknots.shape[d] + vknots.shape[d]
            knots = np.zeros(shape, dtype=knots.dtype)
            
            # Set values by interleaving eknots and vknots
            if d == 0:
                knots[::2] = eknots
                knots[1::2] = vknots
            elif d == 1:
                knots[:,::2] = eknots
                knots[:,1::2] = vknots
            elif d == 2:
                knots[:,:,::2] = eknots
                knots[:,:,1::2] = vknots
            
        # Done
        return knots
