"""
Injectivity conditions for 2D and 3D B-spline grids.

In [1] the base technique for multilevel B-spline grids is given. In [2]
the authors expand on this by introducing sufficient conditions (K2) to
ensure injectivity for 2D grids. This also requires a different approach
for the multi-level grid refinement. In [3] they provide extra conditions
and also give the conditions for the 3D case.

Given:
K2 = 2.046392675
K3 = 2.479472335
A2 = 1.596416285
A3 = 1.654116969

A 2D grid is locally injective all over the domain ...
  * if Kx < 1/K2 and Ky < 1/K2
  * if Kx**2 + Ky**2 < (1/A2)**2

A 2D grid is locally injective all over the domain ...
  * if Kx < 1/K3 and Ky < 1/K3 and Kz < 1/K3
  * if Kx**2 + Ky**2 + Kz**2 < (1/A3)**2

These are *sufficient* conditions. Which means that a grid can also be 
injective if it does not meet this equations. If these conditions are
met, however, injectivity is guaranteed.

A grid is injective if it does not fold. If it maps all the points to a 
unique point. Injectivity therefore implies invertability.


[1]
Lee S, Wolberg G, Shin SY. "Scattered Data Interpolation with Multilevel 
B-splines". IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS.
1997;3(3):228―244. Available at: .

[2]
Lee S, Wolberg G, Chwa K-yong, Shin SY. "Image Metamorphosis with Scattered
Feature Constraints". IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS.
1996;2:337--354. Available at: Bezocht 09:46:03.

[3]
Choi Y, Lee S. "Injectivity conditions of 2d and 3d uniform cubic b-spline
functions". GRAPHICAL MODELS. 2000;62:2000. Available at: Bezocht maart 25, 
2011.

"""

# K2 and K3 are given
K2 = 2.046392675
K3 = 2.479472335

# Calculate A2 and A3
_3o2 = 3/2.0
A2 = ( _3o2**2 + (K2-_3o2)**2 )**0.5
A3 = ( _3o2**2 + (K2-_3o2)**2 + (K3-K2)**2 )**0.5

# Show
print('A2 =', A2)
print('A3 =', A3)
