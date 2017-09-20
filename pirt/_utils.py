"""
Small utility classes.
"""

import re

import numpy as np


try:  # pragma: no cover
    from collections import OrderedDict as _dict  # noqa
except ImportError:
    _dict = dict

# Vebdored code, copied as module
from .new_pointset import PointSet


def isidentifier(s):
    # http://stackoverflow.com/questions/2544972/
    if not isinstance(s, str):
        return False
    return re.match(r'^\w+$', s, re.UNICODE) and re.match(r'^[0-9]', s) is None


# Copied with changes from Pyzo/zon
class Parameters(_dict):
    """ A dict in which the items can be get/set as attributes.
    """
    
    __reserved_names__ = dir(_dict())  # Also from OrderedDict
    __pure_names__ = dir(dict())
    
    __slots__ = []
    
    def __repr__(self):
        identifier_items = []
        nonidentifier_items = []
        for key, val in self.items():
            if isidentifier(key):
                identifier_items.append('%s=%r' % (key, val))
            else:
                nonidentifier_items.append('(%r, %r)' % (key, val))
        if nonidentifier_items:
            return 'Parameters([%s], %s)' % (', '.join(nonidentifier_items),
                                       ', '.join(identifier_items))
        else:
            return 'Parameters(%s)' % (', '.join(identifier_items))
    
    def __str__(self):
        
        # Get alignment value
        c = 0
        for key in self:
            c = max(c, len(key))
        
        # How many chars left (to print on less than 80 lines)
        charsLeft = 79 - (c+6)
        
        s = '<%i parameters>\n' % len(self)
        for key in self.keys():
            valuestr = repr(self[key])
            if len(valuestr) > charsLeft:
                valuestr = valuestr[:charsLeft-3] + '...'
            s += key.rjust(c+4) + ": %s\n" % (valuestr)
        return s
    
    def __getattribute__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            if key in self:
                return self[key]
            else:
                raise
    
    def __setattr__(self, key, val):
        if key in self.__class__.__reserved_names__:
            # Either let OrderedDict do its work, or disallow
            if key not in self.__class__.__pure_names__:
                return _dict.__setattr__(self, key, val)
            else:
                raise AttributeError('Reserved name, this key can only ' +
                                     'be set via ``d[%r] = X``' % key)
        else:
            # if isinstance(val, dict): val = Dict(val) -> no, makes a copy!
            self[key] = val
    
    def __dir__(self):
        names = [k for k in self.keys() if isidentifier(k)]
        return self.__class__.__reserved_names__ + names


# Vendored class from Visvis

class Aarray(np.ndarray):
    """ Aarray(shape_or_array, sampling=None, origin=None, fill=None,
                dtype='float32', **kwargs)
    
    Anisotropic array; inherits from numpy.ndarray and adds a sampling
    and origin property which gives the sample distance and offset for
    each dimension.
    
    Parameters
    ----------
    shape_or_array : shape-tuple or numpy.ndarray
        Specifies the shape of the produced array. If an array instance is
        given, the returned Aarray is a view of the same data (i.e. no data
        is copied).
    sampling : tuple of ndim elements
        Specifies the sample distance (i.e. spacing between elements) for
        each dimension. Default is all ones.
    origin : tuple of ndim elements
        Specifies the world coordinate at the first element for each dimension.
        Default is all zeros.
    fill : scalar (optional)
        If given, and the first argument is not an existing array,
        fills the array with this given value.
    dtype : any valid numpy data type
        The type of the data
    
    All extra arguments are fed to the constructor of numpy.ndarray.
    
    Implemented properties and methods
    -----------------------------------
      * sampling - The distance between samples as a tuple
      * origin - The origin of the data as a tuple
      * get_start() - Get the origin of the data as a Point instance
      * get_end() - Get the end of the data as a Point instance
      * get_size() - Get the size of the data as a Point instance
      * sample() - Sample the value at the given point
      * point_to_index() - Given a poin, returns the index in the array
      * index_to_point() - Given an index, returns the world coordinate
    
    Slicing
    -------
    This class is aware of slicing. This means that when obtaining a part
    of the data (for exampled 'data[10:20,::2]'), the origin and sampling
    of the resulting array are set appropriately.
    
    When applying mathematical opertaions to the data, or applying
    functions that do not change the shape of the data, the sampling
    and origin are copied to the new array. If a function does change
    the shape of the data, the sampling are set to all zeros and ones
    for the origin and sampling, respectively.
    
    World coordinates vs tuples
    ---------------------------
    World coordinates are expressed as Point instances (except for the
    "origin" property). Indices as well as the "sampling" and "origin"
    attributes are expressed as tuples in z,y,x order.
    
    """
    
    _is_Aarray = True
    
    def __new__(cls, shapeOrArray, sampling=None, origin=None, fill=None,
                dtype='float32', **kwargs):
        
        if isinstance(shapeOrArray, np.ndarray):
            shape = shapeOrArray.shape
            ob = shapeOrArray.view(cls)
            if sampling is None and hasattr(shapeOrArray, 'sampling'):
                sampling = shapeOrArray.sampling
            if origin is None and hasattr(shapeOrArray, 'origin'):
                origin = shapeOrArray.origin
        else:
            shape = shapeOrArray
            ob = np.ndarray.__new__(cls, shape, dtype=dtype, **kwargs)
            if fill is not None:
                ob.fill(fill)
        
        # init sampling and origin
        ob._sampling = tuple( [1.0 for i in ob.shape] )
        ob._origin = tuple( [0.0 for i in ob.shape] )
        
        # set them
        if sampling:
            ob.sampling = sampling
        if origin:
            ob.origin = origin
        
        # return
        return ob
    
    
    def __array_finalize__(self, ob):
        """ So the sampling and origin is maintained when doing
        calculations with the array. """
        #if hasattr(ob, '_sampling') and hasattr(ob, '_origin'):
        if isinstance(ob, Aarray):
            if self.shape == ob.shape:
                # Copy sampling and origin for math operation
                self._sampling = tuple( [i for i in ob._sampling] )
                self._origin = tuple( [i for i in ob._origin] )
            else:
                # Don't bother (__getitem__ will set them after this)
                # Other functions that change the shape cannot be trusted.
                self._sampling = tuple( [1.0 for i in self.shape] )
                self._origin = tuple( [0.0 for i in self.shape] )
        elif isinstance(self, Aarray):
            # This is an Aarray, but we do not know where it came from
            self._sampling = tuple( [1.0 for i in self.shape] )
            self._origin = tuple( [0.0 for i in self.shape] )
    
    
    def __getslice__(self, i, j):
        # Called only when indexing first dimension and without a step
        
        # Call base getitem method
        ob = np.ndarray.__getslice__(self, i, j)
        
        # Perform sampling and origin corrections
        sampling, origin = self._correct_sampling(slice(i,j))
        ob.sampling = sampling
        ob.origin = origin
        
        # Done
        return ob
    
    
    def __getitem__(self, index):
        
        # Call base getitem method
        ob = np.ndarray.__getitem__(self, index)
        
        if isinstance(index, np.ndarray):
            # Masked or arbitrary indices; sampling and origin irrelevant
            ob = np.asarray(ob)
        elif isinstance(ob, Aarray):
            # If not a scalar, perform sampling and origin corrections
            # This means there is only a very small performance penalty
            sampling, origin = self._correct_sampling(index)
            if sampling:
                ob.sampling = sampling
                ob.origin = origin
        
        # Return
        return ob
    
    def __array_wrap__(self, out, context=None):
        """ So that we return a native numpy array (or scalar) when a
        reducting ufunc is applied (such as sum(), std(), etc.)
        """
        if not out.shape:
            return out.dtype.type(out)  # Scalar
        elif out.shape != self.shape:
            return out.view(type=np.ndarray)
        else:
            return out  # Type Aarray

    def _correct_sampling(self, index):
        """ _correct_sampling(index)
        
        Get the new sampling and origin when slicing.
        
        """
        
        # Init origin and sampling
        _origin = self._origin
        _sampling = self._sampling
        origin = []
        sampling = []
        
        # Get index always as a tuple and complete
        index2 = [None]*len(self._sampling)
        if not isinstance(index, (list,tuple)):
            index2[0] = index
        else:
            try:
                for i in range(len(index)):
                    index2[i] = index[i]
            except Exception:
                index2[0] = index
        
        # Process
        for i in range(len(index2)):
            ind = index2[i]
            if isinstance(ind, slice):
                    #print(ind.start, ind.step)
                if ind.start is None:
                    origin.append( _origin[i] )
                else:
                    origin.append( _origin[i] + ind.start*_sampling[i] )
                if ind.step is None:
                    sampling.append(_sampling[i])
                else:
                    sampling.append(_sampling[i]*ind.step)
            elif ind is None:
                origin.append( _origin[i] )
                sampling.append(_sampling[i])
            else:
                pass # singleton dimension that pops out
        
        # Return
        return sampling, origin
    
    
    def _set_sampling(self,sampling):
        if not isinstance(sampling, (list,tuple)):
            raise ValueError("Sampling must be specified as a tuple or list.")
        if len(sampling) != len(self.shape):
            raise ValueError("Sampling given must match shape.")
        for i in sampling:
            if i <= 0:
                raise ValueError("Sampling elements must be larger than zero.")
        # set
        tmp = [float(i) for i in sampling]
        self._sampling = tuple(tmp)
    
    
    def _get_sampling(self):
        l1, l2 = len(self._sampling), len(self.shape)
        if l1 < l2:
            tmp = list(self._sampling)
            tmp.extend( [1 for i in range(l2-l1)] )
            return tuple( tmp )
        elif l1 > l2:
            tmp = [self._sampling[i] for i in range(l2)]
            return tuple(tmp)
        else:
            return self._sampling
    
    sampling = property(_get_sampling, _set_sampling, None,
        "A tuple with the sample distance for each dimension.")
    
    
    def _set_origin(self,origin):
        if not isinstance(origin, (list,tuple)):
            raise ValueError("Origin must be specified as a tuple or list.")
        if len(origin) != len(self.shape):
            raise ValueError("Origin given must match shape.")
        # set
        tmp = [float(i) for i in origin]
        self._origin = tuple(tmp)
    
    
    def _get_origin(self):
        l1, l2 = len(self._origin), len(self.shape)
        if l1 < l2:
            tmp = list(self._origin)
            tmp.extend( [0 for i in range(l2-l1)] )
            return tuple( tmp )
        elif l1 > l2:
            tmp = [self._origin[i] for i in range(l2)]
            return tuple(tmp)
        else:
            return self._origin
    
    origin = property(_get_origin, _set_origin, None,
        "A tuple with the origin for each dimension.")
    
    
    def point_to_index(self, point, non_on_index_error=False):
        """ point_to_index(point, non_on_index_error=False)
        
        Given a point returns the sample index (z,y,x,..) closest
        to the given point. Returns a tuple with as many elements
        as there are dimensions.
        
        If the point is outside the array an IndexError is raised by default,
        and None is returned when non_on_index_error == True.
        
        """
        point = tuple(point.flat)
        
        # check
        if len(point) != len(self.shape):
            raise ValueError("Given point must match the number of dimensions.")
        
        # calculate indices
        ii = []
        for i in range(len(point)):
            s = self.shape[i]
            p = ( point[-(i+1)] - self._origin[i] ) / self._sampling[i]
            p = int(p+0.5)
            if p<0 or p>=s:
                ii = None
                break
            ii.append(p)
        
        # return
        if ii is None and non_on_index_error:
            return None
        elif ii is None:
            raise IndexError("Sample position out of range: %s" % str(point))
        else:
            return tuple(ii)
    
    
    def sample(self, point, default=None):
        """ sample(point, default=None)
        
        Take a sample of the array, given the given point
        in world-coordinates, i.e. transformed using sampling.
        By default raises an IndexError if the point is not inside
        the array, and returns the value of "default" if it is given.
        
        """
        
        tmp = self.point_to_index(point,True)
        if tmp is None:
            if default is None:
                ps = str(point)
                raise IndexError("Sample position out of range: %s" % ps)
            else:
                return default
        return self[tmp]


    def index_to_point(self, *index):
        """ index_to_point(*index)
        
        Given a multidimensional index, get the corresponding point in world
        coordinates.
        
        """
        # check
        if len(index)==1:
            index = index[0]
        if not hasattr(index,'__len__'):
            index = [index]
        if len(index) != len(self.shape):
            raise IndexError("Invalid number of indices.")
        
        # init point as list
        pp = []
        # convert
        for i in range(len(self.shape)):
            ii = index[i]
            if ii<0:
                ii = self.shape[i] - ii
            p = ii * self._sampling[i] + self._origin[i]
            pp.append(p)
        # return
        pp.reverse()
        return PointSet(pp)


    def get_size(self):
        """ get_size()
        
        Get the size (as a vector) of the array expressed in world coordinates.
        
        """
        pp = []
        for i in range(len(self.shape)):
            pp.append( self._sampling[i] * self.shape[i] )
        pp.reverse()
        return PointSet(pp)
    
    
    def get_start(self):
        """ get_start()
        
        Get the origin of the array expressed in world coordinates.
        Differs from the property 'origin' in that this method returns
        a point rather than indices z,y,x.
        
        """
        pp = [i for i in self.origin]
        pp.reverse()
        return PointSet(pp)
    
    
    def get_end(self):
        """ get_end()
        
        Get the end of the array expressed in world coordinates.
        
        """
        pp = []
        for i in range(len(self.shape)):
            pp.append( self._origin[i] + self._sampling[i] * self.shape[i] )
        pp.reverse()
        return PointSet(pp)
