""" Module experiment.

Provides a means to perform experiments (not limited to registration).

This module provides two classes:
  * Database - A generic way to store data using ssdf.
  * Experiment - A base class that represents an experiment.

"""

# todo: move to pyzolib? or somewhere else?

import os
import hashlib

import numpy as np
from visvis import ssdf


class Database:
    """ Database(fname=None)
    
    Database for storing objects using keys. These keys can be
    strings (restricted similar to Python variable names) or
    ssdf structs. 
    
    When a filename is given, the database is loaded from that file (if
    it exists) and save() will save the database to this file.
    
    Objects to store should be classes natively supported by SSDF, 
    or SSDF-compatible classes (i.e. classes
    that implement the __to_ssdf__ and __from_ssdf__ methods). See
    the documentation of SSDF for more information.
    
    """
    
    def __init__(self, fname=None):
        
        # Store filename
        self._fname = fname
        
        # Load
        if fname and os.path.exists(fname):
            self._db = ssdf.load(fname)
        else:
            self._db = ssdf.new()
        
        # Key of last added entry
        self._lastkey = ''
    
    
    @property
    def db(self):
        """ Get the underlying ssdf struct.
        """
        return self._db
    
    
    def set(self, key, object, save_now=True):
        """ set(key, object, save_now=True)
        
        Add an object to the database, using the given key.
        If save_now is True, will save the database to disk now.
        
        """
        
        # Correct key
        key = self._key2key(key)
        #print 'set', key
        
        # Test object
        if ssdf.isstruct(object):
            pass
        if isinstance(object, (int, float, str, np.ndarray, tuple, list)):
            pass # Default ssdf compatible
        elif ssdf.is_compatible_class(object):
            pass # Compatible
        else:
            raise ValueError('The given object is not ssdf compatible.')
        
        # Store last key
        self._lastkey = key
        
        # Store object in db
        self._db[key] = object
        
        # Save?
        if save_now and self._fname:
            ssdf.save(self._fname, self._db)
    
    
    def get(self, key):
        """ get(key)
        
        Get the stored object using the given key. Returns None if no
        object is stored under the given key.
        
        """
        
        # Correct key
        key = self._key2key(key)
        #print 'get', key
        
        # Get object
        try:
            return self._db[key]
        except KeyError:
            return None
    
    
    def save(self, fname=None):
        """ save(fname=None)
        
        Save the results to disk now. If fname is not given, the last used
        fname is used. If there is no fname available, will raise a runtime
        error.
        
        """
        
        # Use given or stored fname?
        if fname is None:
            fname = self._fname
        else:
            self._fname = fname
        
        # Check if there is an fname available
        if fname is None:
            raise RuntimeError('No known filename to write to.')
        
        # Save
        ssdf.save(fname, self._db)
    
    
    def _key2key(self, key):
        """ _key2key(key)
        
        Get real key from given key.
        
        """
        
        # Hash key?
        if isinstance(key, str):
            key = key.replace(' ', '_')
        else:
            key = self._hasher(key)
        
        return key
    
    
    def _hasher(self, object):
        """ _hasher(object)
        
        Create a hash string from the given object. This object can be 
        a string, dict or ssdf struct.
        
        """ 
        
        # Init
        h = hashlib.sha256()
        
        # Hash
        if isinstance(object, str):
            h.update(object)
        elif ssdf.isstruct(object):
            h.update(ssdf.saves(object))
        elif isinstance(object, dict):
            for key in params:
                s = key + '=' + str(params[key])
                h.update(s)
        else:
            raise ValueError('I do not know how to hash this object.')
        
        # Done        
        return '_' + h.hexdigest()


NAME_SERIES_PARAMS = '_series_params'
NAME_SERIES_RESULTS = '_series_results'

class Experiment:    
    """ Experiment(params, database=None)
    
    Base class to perform experiments. 
    
    This class enables users to do experiments. It uses the concept of
    series to enable sweeping different parameters. It can use a
    database to store and reuse experiment results.
    
    This class promotes the separation between the experiment and its 
    quantitative analysis. The get_result() method, can be used to 
    obtain the raw experiment results, which can then be processed 
    before displaying them.
    
    
    Series
    ------
    When doing an experiment, one deals with statistics. Therefore an 
    experiment should often be performed multiple times. Further, when 
    testing an algorithm, one often wants to see how a specific parameter 
    of the algorithm affects the performance. 
    
    Therefore, one often wants to run a series of experiments. Sometimes 
    one wants to run a series of series of experiments, and sometimes, 
    one even wants to run a series of series of series of experiments ...
    
    This class uses the notion of series. Each series has a number, starting
    from zero. Series 0 simply means that the experiment is performed
    multiple times (while varying a parameter). Series 1 means that series 0 
    is performed multiple times (while varying another parameter). Series 2
    means that series 1 is performed multiple times, etc.
    
    The number of series is limited by the maximum recursion depth of Python.
    
    In pseudocode:
    {{{
    for parameter_changes in series2:
        for parameter_changes in series1:
            for parameter_changes in series0:
                experiment()
    }}} 
    
    
    Parameters
    ----------
    This class uses an ssdf struct to store the experiment parameters. Which 
    parameter is changed in a series, and what values that parameter should
    take in that series, can be set using set_series_params().
    
    
    Buffering
    ---------
    This class can make use of a Database instance. When set, it will store 
    all results produced by the experiment() method. This way, when an
    experiment is repeated with the same parameters, the experiment results
    can be obtained from the database. 
    
    The database's save() method is called at the end of each series 1. 
    Note that in order to use a database, the results produced by experiment()
    must be SSDF compatible, and if the result is a custom class, this class
    needs to be registered to SSDF. 
    
    
    Usage
    -----
    To use this class, subclass it and implement the following methods:
      * experiment() - This method accepts one argument (a params struct) 
        and should return a result object.
      * quantify_results() - To process the results, making them ready for
        presentation (using get_result()).
      * show_results() - Show the results (using quantift_results()).
    
    The quantify_results() and show_results() are not required to run the
    experiments, but function as a guide to process the experiment results.
    
    To run a single experiment with the given parameters, use do_experiment(). 
    To do a series of experiments, use do_experiment_series().
    
    There is an example at the bottom of the file (experiment.py) that defines
    this class.
    
    """
    
    def __init__(self, params, database=None):
        
        # Store parameters
        self._params = params
        
        # Init one result
        self._one_result = None
        
        # Series params are stored as NAME_SERIES_PARAMS_x
        # Series results are stored as NAME_SERIES_RESULTS_x
        
        # Attach database
        self._database = database
        self._database_overwrite = False
        self._save_next_result = False
    
    
    def set_database(self, database= None):
        """ set_database(database)
        
        Set the database to use to buffer experiment results. Call
        without arguments to disable using a database.
        
        Special database attrinutes
        ---------------------------
        The attribute _database_overwrite can be used to overwrite the 
        database (for example when you changed your algoritm).
        
        The attribute _save_next_result can be set to False in experiment() 
        to signal that the result that it produces should not be stored.
        This variable is reset each time before experiment() is called.
        
        """
        self._database = database
    
    
    @property
    def params(self):
        """ Get the original parameters as given during initialization.
        """
        return self._params
    
    
    def _get_list_member(self, name, series_nr, clear=False):
        """ _get_list_member(name, series_nr, clear=False)
        
        Return a list instance that is a member of this class, 
        corresponding with the given name and series_nr. This method
        enables ad hoc creation of members as needed (since we do not
        know how many levels the user wants).
        
        """
        
        # Get full name
        full_name = '%s_%i' % (name, series_nr)
        
        # Create list and return
        if clear:
            self.__dict__[full_name] = tmp = []
        else:
            tmp = self.__dict__[full_name]        
        return tmp
    
    
    def set_series_params(self, series_nr, param, values):
        """ set_series_params(seriesNr, param, values)
        
        Set which parameter to vary for the given series, and what values
        should be iterated.
        
        """
        
        # Test arguments
        if not isinstance(series_nr, int) or series_nr < 0:
            raise ValueError('series_nr must be an integer >= zero.')
        if not isinstance(param, str):
            raise ValueError('param must be a parameter name (as a string).')
        if isinstance(values, np.ndarray):
            values = [val for val in values]
        if not isinstance(values, (list, tuple)):
            raise ValueError('values must be a list or tuple.')
        
        # Get list (resets the parameter list)
        L = self._get_list_member(NAME_SERIES_PARAMS, series_nr, clear=True)
        
        # Set
        if None not in [param, values]:
            L.append(param)
            L.append(values)
    
    
    def get_series_params(self, series_nr):
        """ get_series_params(seriesNr)
        
        Get which parameter is varied for the given series, and what
        values are iterated.
        Returns a tuple (param, values). 
        
        Raises a RuntimeError if no series params have been defined for the
        given series_nr.
        
        """
        
        # Test arguments
        if not isinstance(series_nr, int) or series_nr < 0:
            raise ValueError('series_nr must be an integer >= zero.')
        
        # Get list
        try:
            L = self._get_list_member(NAME_SERIES_PARAMS, series_nr)
        except KeyError:
            raise RuntimeError(
                'Series params have not been set for series %i.' % series_nr)
        
        # Return
        return L[0], L[1]
    
    
    def do_experiment_series(self, series_nr, params=None):
        """ do_experiment_series(series_nr)
        
        Perform a series of experiments. 
        
        Returns a list of results (which may contain lists of results, etc.).
        The results can also be accesed using the get_result() method.
        
        """
        
        # Get params (copy)
        if params is None:
            params = self.params
        params = ssdf.copy(params)
        
        # Get series parameter
        param, values = self.get_series_params(series_nr)
        
        # Init results
        series_results = self._get_list_member(NAME_SERIES_RESULTS, series_nr, True)
        
        for value in values:
            
            # Print info
            tmp = '=' * 2 * (series_nr+1)
            rv = repr(value)
            print('%s SERIES %i: %s = %s %s' % (tmp, series_nr, param, rv, tmp))
            
            # Set value for this iteration
            params[param] = value
            
            # Perform experiment, or another series of experiments
            if series_nr == 0:
                result = self.do_experiment(params)
            else:
                result = self.do_experiment_series(series_nr-1, params)
            
            # Store result
            series_results.append(result)
            
            # Store database if series_nr is 1
            if self._database and self._database._fname and series_nr==1:
                self._database.save()
        
        # Done
        return series_results
    
    
    def do_experiment(self, params=None):
        """ do_experiment()
        
        Perform a single experiment.
        The resulting result is stored returned. The result can also be 
        accesed using the get_result() method.
        
        """
        
        # Get params (copy when calling experiment)
        if params is None:
            params = self.params
        
        # Try getting result from database
        result = None
        if self._database and not self._database_overwrite:
            result = self._database.get(params)
        
        # Get result by running experiment
        if result:            
            print('\b' + ' (result in db)')
        else:
            
            # Perform experiment 
            self._save_next_result = True
            result = self.experiment(ssdf.copy(params))
            
            # Check
            if result is None:
                raise RuntimeError('experiment should not return None.')
            
            # Store in db
            if self._database and self._save_next_result:
                self._database.set(params, result, False)
        
        # Store result and return
        self._one_result = result
        return result
    
    
    def get_result(self, *args):
        """ get_result(series0_index, series1_index, series2_index, ...)
        
        Get the result (as returned by the overloaded experiment()) for the
        given series indices. 
        
        If not arguments are given, returns the last result. If one index
        is given, returns the result corresponding to the last series 0. etc.
        
        None can be given for up to one series index, in which case a list
        of results is returned corresponding to the range of values given
        in set_series_params().
        
        """
        
        
        # The last result
        if len(args) == 0:
            return self._one_result
        
        # The length of the args determines the series_nr
        series_nr = len(args) - 1
        
        # Get result list
        try:
            L0 = self._get_list_member(NAME_SERIES_RESULTS, series_nr)
        except KeyError:
            raise ValueError('More indices supplied than series available.')
        
        # Get amount of None's
        none_count = list(args).count(None)
        
        if none_count == 0:
            # Return single result object
            
            L = L0
            for arg in reversed(args):
                if isinstance(arg, int):
                    L = L[arg]
                else:
                    raise ValueError(
                                'get_experiment_result only accepts integers.')
            result = L
        
        elif none_count == 1:
            # Return list of result objects. Find objects one by one.
            
            # Init
            result = []
            i = -1
            
            while i>-2:
                i += 1
                
                # Find scalar value, while filling in an index where it is None
                L = L0
                for arg in reversed(args):
                    if arg is None:
                        if i >= len(L):
                            i = -9
                            break
                        arg = i
                    if isinstance(arg, int):
                        L = L[arg]
                    else:
                        raise ValueError(
                                'get_experiment_result only accepts integers.')
                else:
                    result.append(L)
        
        else:
            # Not possible
            raise ValueError('Can only use one None value.')
        
        # Return result (which should now be an object as returned by experiment) 
        return result
    
    
    def experiment(self, params):
        """ experiment(params)
        
        This is the method that needs to be implemented in order for
        the experiment to do what you want.
        
        """
        raise NotImplemented("This method needs to be implemented.")
    
    
    def quantify_results(self):
        """ quantify_results()
        
        Implement this method to perform post-processing, to quantify 
        the results. This might not be necessary in all situations.
        
        Hint
        ----
        When implementing this method, use get_result() to obtain results
        and return them in a way that allows easy representation.
        
        """
        raise NotImplemented("This method needs to be implemented.")
    
    
    def show_results(self):
        """ show_results()
        
        Implement this method to show the results.
        
        Hint
        ----
        When implementing this method, use quantify_results() to obtain
        results ready for displaying and then dispay the result as
        appropriate.
        
        """
        raise NotImplemented("This method needs to be implemented.")


if __name__ == '__main__':
    # Test experiment
    
    # Let's say we want to estimate for which value of x some magic process
    # is maximum. We simulate this process using noise. By using a seed,
    # the experiments are repeatable.
    import time
    np.random.seed(1234)
    
    params = ssdf.new()
    params.noise_level = 3
    params.x = 3
    
    class TestExperiment(Experiment):
        """ In this experiment we will use series 0 to repeat the process
        (using different instantiations of the noise). We will use series 1
        to vary the parameter x of which we want to know the optimum value.
        """
        
        def experiment(self, params):
            """ experiment(params)            
            Our magic box. Given params.x, it calculates a value. In a
            real experiment, this calculation may be unknown to us.
            """
            t0 = time.time()
            noise = np.random.normal(0, params.noise_level)
            value = - 1 * params.x**2 + 10 * params.x + noise
            t1 = time.time()
            # Return value and elapsed time to calculate
            return float(value), t1-t0
        
        def quantify_results(self, what):
            """ quantify_results(what)
            Make the results ready for presentation. Basically, we select
            either the value or the time from the raw results, and we 
            average the results.
            """
            
            # Get params for different series
            param_x, values_x = self.get_series_params(1)
            param_iter, values_iter = self.get_series_params(0)
            
            # Set result index
            if what=='value':
                result_index = 0
            elif what=='time':
                result_index = 1
            else:
                raise ValueError('Not sure what you want to quantify.')
            
            # Collect results for different x (by averaging over multiple iters)
            results = []
            for i in range(len(values_x)):
                result = 0
                for j in range(len(values_iter)):
                    raw = self.get_result(j, i)
                    result += raw[result_index]
                results.append( float(result)/len(values_iter) ) # Average
            
            # Done
            return results
        
        def show_results(self, what='value'):
            """ show_results()
            Show the results.
            """
            
            # Get params for x            
            param_x, values_x = self.get_series_params(1) 
            
            # Get results
            results = self.quantify_results(what)
            
            # Plot or show
            try:
                import visvis as vv
                vv.figure(); vv.plot(values_x, results)
                vv.xlabel(param_x); vv.ylabel('values')
            except ImportError:
                print('param', param_x, ': values')
                for x, r in zip(values_x, results):
                    print(x, ':', r)
    
    # Create experiment
    db = Database()
    exp = TestExperiment(params, db)
    # Set series
    exp.set_series_params(0, 'noise_iter', range(100, 110))
    exp.set_series_params(1, 'x', range(10))
    # Run experiment and show results
    exp.do_experiment_series(1)
    exp.show_results()
