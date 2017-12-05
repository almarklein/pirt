--------------------------------------
pirt.reg - The registration algorithms
--------------------------------------

.. automodule:: pirt.reg


* The :class:`pirt.AbstractRegistration` class provides the root of all
  registration classes, implementing a common API.
* The :class:`pirt.BaseRegistration` class provides the base class for all
  registration algorithms implemented in PIRT itself.
* The :class:`pirt.GDGRegistration` class provides the base class for the
  registration algorithms that adopt diffeomorphic constraints.
* The :class:`pirt.NullRegistration` class provides the identity transform.
* The :class:`pirt.OriginalDemonsRegistration` class represents the Demons
  algorithm.
* The :class:`pirt.DiffeomorphicDemonsRegistration` class represents the Demons
  algorithm with a diffeomorphic twist.
* The :class:`pirt.GravityRegistration` class represents the Gravity
  registration algorithm.
* The :class:`pirt.ElastixRegistration` class and its subclasses represents
  registration algorithms provided by Elastix.


Base registration classes
-------------------------


.. autoclass:: pirt.AbstractRegistration
    :members:

.. autoclass:: pirt.BaseRegistration
    :members:

.. autoclass:: pirt.GDGRegistration
    :members:


Special classes
---------------

.. autoclass:: pirt.NullRegistration
    :members:


Demons registration
-------------------

The demon's algorithm is a simple yet often effective algorithm. We've improved
upon the algorithm by making it diffeomorphic, greatly improving the realism
of the produced deformations.

.. autoclass:: pirt.OriginalDemonsRegistration
    :members:

.. autoclass:: pirt.DiffeomorphicDemonsRegistration
    :members:


Gravity registration
--------------------

.. autoclass:: pirt.GravityRegistration
    :members:


Elastix registration
--------------------

.. autoclass:: pirt.ElastixRegistration
    :members:

.. autoclass:: pirt.ElastixRegistration_rigid
    :members:

.. autoclass:: pirt.ElastixRegistration_affine
    :members:

.. autoclass:: pirt.ElastixGroupwiseRegistration
    :members:

