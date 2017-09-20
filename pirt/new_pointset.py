import time
import numpy as np

# Visvis contains a Pointset and Point class, which proofed useful, but had
# downsides, mostly because they do not inherit from ndarray, but wrap it.
# This new version subclasses ndarray so that it can be used much more
# generically. This copy was taken from stentseg. But that should not matter,
# duck typing for the win.

# todo: Awesome! but now where do I put this ...
# todo: rename to avoid name clashes with vv.Pointset?
# PointSet is a bit ugly, but "pointset" is nice because it does not imply plural
# Vector seems more general (points are subset of vectors?) but it pronounces less nicely
class PointSet(np.ndarray):
    """ The PointSet class can be used to represent sets of points or
    vectors, as well as singleton points. The dimensionality of the
    vectors in the pointset can be anything, and the dtype can be any
    of those supported by numpy.
    
    This class inherits from np.ndarray, which makes it very flexible;
    you can threat it as a regular array, and also pass it to functions
    that require a numpy array. The shape of the array is NxD, with N
    the number of points, and D the dimensionality of each point.
    
    This class has a __repr__ that displays a pointset-aware description.
    To see the underlying array, use print, or use pointset[...] to 
    convert to a pure numpy array.
    
    Parameters
    ----------
    input : various
        If input is in integer, it specifies the dimensionality of the array,
        and an empty pointset is created. If input is a list, it specifies
        a point, with which the pointset is initialized. If input is a numpy
        array, the pointset is a view on that array (ndim must be 2).
    dtype : dtype descrsiption
        The data type of the numpy array. If not given, the result will
        be float32.
    
    """
    
    def __new__(cls, input, dtype=np.float32):
        if isinstance(input, int):
            # ndim is given, start empty pointset
            return np.ndarray.__new__(cls, (0,input), dtype=dtype)
        elif isinstance(input, (tuple, list, reversed)):
            # Point is given, initialize with that point
            input = list(input)
            if not all([isinstance(i, (float, int)) for i in input]):
                raise ValueError('Lists given to PointSet must be all scalars.')
            a = np.array(input, dtype=dtype)
            a.shape = 1, len(a)
            return a.view(cls)
        elif isinstance(input, np.ndarray):
            # Array is given, turn into pointset
            if not input.ndim == 2:
                raise ValueError('Arrays given to PointSet must have ndim=2.')
            if input.dtype != np.dtype(dtype):
                input = input.astype(dtype)
            return input.view(cls)
        else:
            # Don't know what to do
            raise ValueError('Invalid type to instantiate PointSet with (%r)' % type(input))
    
    
    def __str__(self):
        """ print() shows elements as normal. """
        return self[...].__str__()


    def __repr__(self):
        """" Return short(one line) string representation of the pointset. """
        if len(self) == 0:
            return "<Empty PointSet (np.ndarray) for points of %i dimensions>" % (
                 self.shape[1], )
        elif len(self) == 1:
            r = ', '.join( ['%1.4g' % i for i in self[0,:]])
            return "<PointSet (np.ndarray) with 1 point: %s>" % r
        else:
            return "<PointSet (np.ndarray) with %i points of %i dimensions>" % (
                len(self), self.shape[1] )
    
    
    @property
    def can_resize(self):
        """ Whether points can be appended to/removed from this pointset.
        This can be False if the array does not own its own data or when
        it is not contiguous. In that case, one should make a copy first.
        """
        try:
            self.resize(self.shape[0]+1, self.shape[1], refcheck=False)
        except Exception:
            return False
        else:
            self.resize(self.shape[0]-1, self.shape[1], refcheck=False)
            return True
    
    
    def __array_wrap__(self, out, context=None):
        """ So that we return a native numpy array (or scalar) when a
        reducting ufunc is applied (such as sum(), std(), etc.)
        """
        if not out.shape:
            return out.dtype.type(out)  # Scalar
        elif out.shape != self.shape:
            return np.asarray(out)
        else:
            return out  # Type Image
    
    
    def ravel(self, *args, **kwargs):
        # Return numpy array on ravel
        return np.ndarray.ravel(self,*args, **kwargs)[...]
    
    
    def __getitem__(self, index):
        """ Get a point or part of the pointset. """
        
        # Single index from numpy scalar
        if isinstance(index, np.ndarray) and index.size==1:
            index = int(index)
        
        if isinstance(index, tuple):
            # Multiple indexes: return as array
            return np.asarray(self)[index]
        elif isinstance(index, slice):
            # Slice: return subset
            return np.ndarray.__getitem__(self, index)
        elif isinstance(index, int):
            # Single index: return point
            a = np.ndarray.__getitem__(self, index)
            a.shape = 1, len(a)
            return a
        else: 
            # Probably some other form of subslicing
            return np.asarray(self)[index]
    
    
    def append(self, *p):
        """ Append a point to this pointset. One can give the elements
        of the points as separate arguments. Alternatively, a tuple or
        numpy array can be given.
        """
        p = self._as_point(*p)
        
        # resize 
        self.resize((self.shape[0]+1, self.shape[1]), refcheck=False)
        
        # append point
        self[-1] = p
    
    
    def extend(self, data):
        """ Extend the point set with more points. The shape[1] of the
        given data must match with that of this array.
        """
        # Turn data into PointSet, which will do some checks for us
        pp = PointSet(data)
        
        # Check dimensions
        if pp.shape[1] != self.shape[1]:
            raise ValueError("Cannot extend pointset, because vector sizes does not match.")
        
        # Store current length
        curlen = len(self)
        
        # Resize
        newlen = curlen + len(pp)
        self.resize((newlen, self.shape[1]), refcheck=False)
        
        # Insert new data
        self[curlen:,:] = pp
    
    
    def insert(self, index, *p):
        """ Insert a point at the given index. 
        """
        
        # check index
        if index < 0:
            index = len(self) + index
        if index<0 or index>len(self):
            raise IndexError("Index to insert point out of range.")
        
        # make sure p is a point
        p = self._as_point(*p)
        
        # Get data from index to end
        tmp = self[index:,:].copy()
        
        # Resize 
        self.resize((self.shape[0]+1, self.shape[1]), refcheck=False)
        
        # Insert point
        tmp = self[index:,:].copy()
        self[index] = p
        self[index+1:,:] = tmp
    
    
    def contains(self, *p):
        """ Check whether the given point is already in this set. 
        """
        
        if not len(self):
            return False
        
        # make sure p is a point
        p = self._as_point(*p)
        
        mask = np.zeros((len(self),),dtype='uint8')
        for i in range(self.shape[1]):
            mask += self[:,i]==p[i]
        if mask.max() >= self.shape[1]:
            return True
        else:
            return False
    
    
    def remove(self, *p, **kwargs):
        """ Remove the given point from the point set. Produces an error
        if such a point is not present. If the keyword argument `all`
        is given and True, all occurances of that point are removed.
        Otherwise only the first occurance is removed.
        """
        
        # Parse kwargs
        all = kwargs.pop('all', False)
        if kwargs:
            raise ValueError('Invalid keyword arguments: %r' % list(kwargs.keys()) )
            
        # make sure p is a point
        p = self._as_point(*p)
        
        # calculate mask
        mask = np.zeros((len(self),),dtype='uint8')
        for i in range(self.shape[1]):
            mask += self[:,i] == p[i]
        
        # find points given
        I, = np.where(mask==self.shape[1])
        
        # produce error if not found
        if len(I) == 0:
            raise ValueError("Given point to remove was not found.")
        
        # remove first point
        if all:
            # remove all points (backwards!)
            for i in reversed(I):
                del self[i]
        else:
            del self[ I[0] ]
    
    
    def __delitem__(self, index):
        """ Remove one or multiple points from the pointset. """
        
        # If tuple, not valid
        if isinstance(index, tuple):
            raise IndexError("Can only remove points using 1D slicing.")
        
        # Get start/stop/step
        if isinstance(index, slice):
            start, stop, step = index.indices(self._len)
        else:
            start, stop, step = index, index+1, 1
        
        # if stepping, do it the slow way
        if step > 1:
            indices = [i for i in range(start,stop, step)]
            for i in reversed(indices):
                del self[i]
            return
        
        # move latter block forward
        tmp = self[stop:]
        self[start:start+len(tmp)] = tmp
        
        # reduce length
        newlen = start + len(tmp)
        self.resize((newlen, self.shape[1]), refcheck=False)
    
    
    def pop(self, index=-1):
        """ Remove and returns a point from the pointset. Removes the last
        by default (which is more efficient than popping from anywhere else).
        """
        
        # check index
        index2 = index
        if index < 0:
            index2 = len(self)+ index
        if index2<0 or index2>len(self):
            raise IndexError("Index to insert point out of range.")
        
        # get point
        p = self[index]
        
        # remove it
        if index == -1:
            # easy
            self.resize((self.shape[0]-1, self.shape[1]), refcheck=False)    
        else:       
            # The hard way
            del self[index]
        
        # return
        return p
    
    
    def _as_point(self, *p):
        """ Return as something that can be applied to a row in the array.
        Check whether the point-dimensions match with this point set.
        """
        
        # the point directly given?
        if len(p)==1 and not isinstance(p[0], (float, int)):
            p = p[0]
        
        if isinstance(p, np.ndarray):
            p = p.ravel()
        elif not isinstance(p, (tuple, list)):
            raise ValueError('Invalid point')
        
        # check whether we can append it
        if len(p) != self.shape[1]:
            tmp = "Given point does not match dimension of pointset."
            raise ValueError(tmp)
        
        # done
        return p

    
    ## Math stuff
    
    def norm(self):
        """  Calculate the norm (length) of the vector. This is the
        same as the distance to the origin, but implemented a bit
        faster.
        """
        
        # we could do something like:
        #   return self.distance(Point(0,0,0))
        # but we don't have to perform checks now, which is faster...
        
        dists = np.zeros( (self.shape[0],), np.float64)
        for i in range(self.shape[1]):
            dists += self[:,i].astype(np.float64)**2
        return np.sqrt(dists)


    def normalize(self):
        """ Return normalized vector (to unit length). 
        """
        
        # calculate factor array
        f = 1.0/self.norm()
        f.shape = f.size,1
        f = f.repeat(self.shape[1],1)
        
        # apply
        return self * f


    def normal(self):
        """ Calculate the normalized normal of a vector. Use
        (p1-p2).normal() to calculate the normal of the line p1-p2.
        Only works on 2D points. For 3D points use cross().
        """
        
        # check dims
        if self.shape[1] != 2:
            raise ValueError("Normal can only be calculated for 2D points.")
        
        # prepare
        a = self.copy()        
        f = 1.0/self.norm()
        
        # swap xy, y goes minus
        tmp = a[:,0].copy()
        a[:,0] = a[:,1] * f
        a[:,1] = -tmp * f
        return a
    
    
    ## Math stuff on two vectors
    
    def _check_and_sort(self, p1, p2, what='something'):
        """ _check_and_sort(p1,p2, what='something')
        Check if the two things (self and a second point/pointset)
        can be used to calculate stuff.
        Returns (p1,p2), if one is a point, p1 is it.
        """
        
        # if one is a singleton pointset, put it in p1
        if len(p2) == 1:
            p2,p1 = p1,p2
        
        # only pointsets of equal length can be used
        if len(p1) != len(p2) and len(p1) > 1 and len(p2) > 1:
            # define errors
            err = "To calculate %s  with two pointsets, " % what
            err += "one must be singleton, or both must have the same size."
            raise ValueError(err)
        
        # check dimensions
        if p1.shape[1] != p2.shape[1]:
            tmp = "To calculate %s between two pointsets, " % what
            err = tmp + "their dimensions must be equal."
            raise ValueError(err)
        
        return p1, p2
    
    
    def distance(self, *p):
        """ Calculate the Euclidian distance between two points or
        pointsets. Use norm() to calculate the length of a vector.
        """
        
        # the point directly given?
        if len(p)==1:
            p = p[0]
        
        # Make p a PointSet
        if not isinstance(p, PointSet):
            p = PointSet(p)
        
        # If one of the pointsets is singleton, put it in p1
        p1, p2 = self._check_and_sort(self, p, 'distance')
        
        # Calculate
        dists = np.zeros( (p2.shape[0],), np.float64)
        for i in range(self.shape[1]):
            tmp = p1[:,i] - p2[:,i]
            dists += tmp.astype(np.float64)**2        
        return np.sqrt(dists)


    def angle(self, *p):
        """ Calculate the angle (in radians) between two vectors. For
        2D uses the arctan2 method so the angle has a sign. For 3D the
        angle is the smallest angles between the two vectors.
        
        If no point is given, the angle is calculated relative to the
        positive x-axis.
        """
        
        # the point directly given?
        if len(p)==1:
            p = p[0]
        elif len(p)==0:
            # use default point
            p = PointSet([0 for i in range(self.shape[1])])
            p[0,0] = 1
        
        # Make p a PointSet
        if not isinstance(p, PointSet):
            p = PointSet(p)
        
        # check. Keep the correct order!
        self._check_and_sort(self, p, 'angle')
        p1, p2 = self, p
        
        if p1.shape[1] == 2:
            # calculate 2D case
            angs1 = np.arctan2( p1[:,1], p1[:,0] )
            angs2 = np.arctan2( p2[:,1], p2[:,0] )
            dangs =  angs1 - angs2
            # make number between -pi and pi
            I = np.where(dangs<-np.pi)
            dangs[I] += 2*np.pi
            I = np.where(dangs>np.pi)
            dangs[I] -= 2*np.pi
            return dangs
            
        elif p1.shape[1] == 3:
            # calculate 3D case
            p1, p2 = p1.normalize(), p2.normalize()
            data = np.inner(p1, p2)  # Not np.dot()!
            # correct for round off errors (or we will get NANs)
            data[data>1.0] = 1.0
            data[data<-1.0] = -1.0
            #return data
            return np.arccos(data)
        else:
            # not possible
            raise ValueError("Can only calculate angle for 2D and 3D vectors.")
    
    
    def angle2(self, *p):
        """ Calculate the angle (in radians) of the vector between 
        two points. 
        
        Say we have p1=(3,4) and p2=(2,1). ``p1.angle(p2)`` returns the
        difference of the angles of the two vectors: ``0.142 = 0.927 - 0.785``
        
        ``p1.angle2(p2)`` returns the angle of the difference vector ``(1,3)``:
        ``p1.angle2(p2) == (p1-p2).angle()``
        
        """
        
        # the point directly given?
        if len(p)==1:
            p = p[0]
        elif len(p)==0:
            raise ValueError("Function angle2() requires another point.")
        
        # Make p a PointSet
        if not isinstance(p, PointSet):
            p = PointSet(p)
        
        # check. Keep the correct order!
        self._check_and_sort(self, p,'angle')
        p1, p2 = self, p
        
        if p1.ndim in [2,3]:
            # subtract and use angle()
            return (p1-p2).angle()    
            #meaning: dangs = np.arctan2( data2[:,1]-data1[:,1], data2[:,0]-data1[:,0] )    
        else:
            # not possible
            raise ValueError("Can only calculate angle for 2D and 3D vectors.")
    
    
    def dot(self, *p):
        """ Calculate the dot product of two pointsets. The dot product
        is the standard inner product of the orthonormal Euclidean
        space. The sizes of the point sets should match, or one point
        set should be singular.
        """
        
        # the point directly given?
        if len(p)==1:
            p = p[0]
            
        # Make p a PointSet
        if not isinstance(p, PointSet):
            p = PointSet(p)     
        
        # check
        p1, p2 = self._check_and_sort(self,p,'dot')
        
        # calculate
        data = np.zeros( (p2.shape[0],), np.float64 )
        for i in range(p1.ndim):
            tmp = p1[:,i] * p2[:,i]
            data += tmp
        return data
    
    
    def cross(self, *p):
        """ Calculate the cross product of two 3D vectors. Given two
        vectors, returns the vector that is orthogonal to both vectors.
        The right hand rule is applied; this vector is the middle
        finger, the argument the index finger, the returned vector
        points in the direction of the thumb.
        """
        
        # the point directly given?
        if len(p)==1:
            p = p[0]
        
        if not self.shape[1] == 3:
            raise ValueError("Cross product only works for 3D vectors!")
        
        # Make p a PointSet
        if not isinstance(p, PointSet):
            p = PointSet(p)    
        
        # check (note that we use the original order for calculation)
        p1, p2 = self._check_and_sort(self, p,'cross')
        
        # calculate
        a, b = self, p
        data = np.zeros( p2.shape, np.float64 )
        data[:,0] = a[:,1]*b[:,2] - a[:,2]*b[:,1]
        data[:,1] = a[:,2]*b[:,0] - a[:,0]*b[:,2]
        data[:,2] = a[:,0]*b[:,1] - a[:,1]*b[:,0]
        
        # return
        return PointSet(data)



if __name__ == '__main__':
    import visvis as vv
    a = PointSet(2, np.float32)
    b = vv.Pointset(2)
    
    pp = np.random.uniform(0, 1, (100,2))
    
    t0 = time.time()
    for i in range(len(pp)):
        a.append(pp[i])
    print('numpy resize: %1.3f s' % (time.time()-t0))
    
    t0 = time.time()
    for i in range(len(pp)):
        b.append(pp[i])
    print('PointSet append: %1.3f s' % (time.time()-t0))

    
    