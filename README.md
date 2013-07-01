ForcePy
=======

Force matching code.

Dependencies
==========
    matplotlib, scipy, MDAnalysis, python2.7

Implemented Features 
==========
* Read structure/trajectory files, forces
* JSON input file
* Stochastic Gradient Descent force matching
* Neighborlist


Architecture Notes
==================
The main class to utilize is a `ForceMatch` class. This class takes in one or more reference `Force` objects
that define the forces to match. For example, a `FileForce` will read in forces from a file. The
`ForceMatch` class also takes one or more target `Force` objects, which are the functional forms that are 
going to match the reference forces. Some `Force` objects contain a static class variable that points
to a `ForceCategory` that contains useful methods/variables. For example, the `PairwiseCategory` contains
a neighborlist implementation.

The `PairwiseSpectralForce` is a linear combination of basis
functions. 

A given `Force` may be 'specialized' to work on only a certain type or
type pair. This may be done by calling `specialize_type` before it is
added to a `ForceMatch` class. Another point is in the fact that the
gradients don't depend on the weights for the `PairwiseSpectralForce`
force.

In order to simplify constructing potentials for many type pairs,
there are utility functions on the ForceMatch class to construct all
possible pairs.


Meshes
============
*Uniform mesh

Basis functions
=============
*unit step

Forces
=========
*Spectral 


TODO
==========
* Need to separate update code out of individual forces, since the net force deviation should play a role.
* Multiple atom types
* Bonds, Angles etc.
* Make TopoForceCategory
* Parallelize