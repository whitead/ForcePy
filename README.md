ForcePy
=======

Force matching code.

Dependencies
==========
    matplotlib, scipy, MDAnalysis, python2.7

Install
===============

First install the development branch of MDAnalysis

    mkdir ForcePy && cd ForcePy
    git clone https://code.google.com/p/mdanalysis/ mdanalysis
    git checkout develop
    cd mdanalysis/package
    python setup.py install --user
    
Next install ForcePy
     
     cd ../../
     git clone https://github.com/whitead/ForcePy.git
     python setup.py install --user


Implemented Features 
==========
* Read structure/trajectory files, forces
* JSON input file
* Stochastic Gradient Descent force matching
* Neighborlist

Coarse-graining a Trajectory
==========
    from MDAnalysis import Universe
    from ForcePy import *
    fine_uni = Universe("foo.tpr", "foo.trr")
    coarse_uni =CGUniverse(fine_uni, selections=['name OW', 'name HW1 or name HW2'], 
                          names=['O', 'H2'], 
			  collapse_hydrogens=False)
    corse_uni.write_structure("cg_foo.pdb")
    corse_uni.write_structure("cg_foo.pdb", bonds='full')
    corse_uni.write_trajectory("cg_foo.dcd")
    



    

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
possible pairs. `add_and_type_pairs` copies a force as many times as
needed to have a unique force for every possible pair-pair
interaction.

Observable Variable
===============

The target forcefield/potential may be modified to reproduce some
observable parameter.  The observable should be in a tabular file
containing the total energy of the system at each frame in Column
1. Columns 2 and beyond should be the deviation of the observable. The
algorithm will try to minimize the observable. The observable under the
new forcefied will be sum_i O_i * exp(-(U' - U) * beta), where U' is
the new potential. The gradient at each frame will be -beta * U' *
exp(-(U' - U) * beta). The meshes implment integrated forms for
working with potentials. This gradient, by the way, isn't correct
since the derivative of the normalization constant changes. It reamins
to be shown that this gradient is an unbiased estimator. You should do
that.

We now need the U' derivative wrt to weights. It's the integrated
basis function. Except, there is a problem that we're working per
particle. You'll need to work on the math for that one. I'm guessing
that I can sub in the average energy per particle. No, that won't work
due the exponential. For the now the only idea I've got is to
calculate total energy at each frame. 

What needs to be done for this section: Figure out how to calculate
the potentials given the forces. DONE. Now what needs to be done: Figure
out how to integrate over the potentials. Should be done by adding


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
* Bonds, Angles etc.
* Make TopoForceCategory. Why? No.
* Parallelize
* Think about analytical forms
* 