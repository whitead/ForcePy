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