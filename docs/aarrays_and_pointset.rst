--------------------------------
Anisotropic arrays and pointsets
--------------------------------

Pirt uses two classes to represent anisotropic data and points. These allow
taking anisotropy into account throughout the registration process (all
the way to the interpolation). The visvis library is "aware" of the Aarray class
(well, it checks arrays for a sampling and origin attribute, as duck-typing goes),
and the PointSet class is really just a numpy nd array. Therefore
(anisotropic) data represented with these classes can be correctly
visualized with Visvis without applying any transformations.


.. autoclass:: pirt.Aarray
    :members:

.. autoclass:: pirt.PointSet
    :members:
