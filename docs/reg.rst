--------------------------------------
pirt.reg - The registration algorithms
--------------------------------------

.. automodule:: pirt.reg


Base registration classes
-------------------------


.. autoclass:: pirt.AbstractRegistration
    :members:

.. autoclass:: pirt.BaseRegistration
    :members:

.. autoclass:: pirt.GDGRegistration
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

.. autoclass:: pirt.ElastixGroupwiseRegistration
    :members:
