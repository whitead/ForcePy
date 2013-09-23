Summary
=======

Force matching code. Supports input from Gromacs and Lammps. Not
tested for input NAMD simulations. Outputs Lammps tabular potentials
and input files. Also can be used for topology reduction in
coarse-graining.

Usage Examples
===============

1. Taking an all-atom trajectory and mapping it to a coarse-grained trajectory with CG-sites defined using Charmm/VMD-style selection strings
2. Using forces from an all-atom trajectory and creating a force-field that matches those forces
3. Writing out a force-field to a lammps-style tabulated potential


Installing ForcePy
===============

Dependencies
----------

*Python2.7
*scipy/numpy
*MDAnalysis (development branch)

Optional Dependencies
----------

*MPI4Py (for parallel force-matching)
*matplotlib (for plotting)

Install
----------

First install the development branch of MDAnalysis

```sh
git clone https://code.google.com/p/mdanalysis/ mdanalysis
cd mdanalysis
git checkout develop
cd package
python setup.py install --user
```
    
Next install ForcePy

```sh     
cd ../../
git clone https://github.com/whitead/ForcePy.git ForcePy
cd ForcePy
python setup.py install --user
```

If you see a long list of errors, check the first few. If it says it
can't find `arrayobject.h`, then your numpy headers are not being
found. If you're in the Voth group and using the Enthought python
distribution, try adding this line to your `~/.profile` or
`~/.bash_profile` file:

```bash
export C_INCLUDE_PATH=/opt/local/include:/Library/Frameworks/EPD64.framework/Versions/7.2/lib/python2.7/site-packages/numpy/core/include:$C_INCLUDE_PATH
```

Coarse-graining a trajectory
==========

The ForcePy module can be used to coarse-grained a trajectory. In this example, we'll convert 
an all-atom water simulation to a 2-site water model.

The first step is to import the necessary libraries:

    from MDAnalysis import Universe
    from ForcePy import *

Next, we load the fine-grained trajectory. The first file
is the structure (`pdb`, `tpr`, `gro`, or `psf`) and the
second file is the trajectory. Note that the `tpr` file reader depends highly
on what version of gromacs with which the `tpr` file was created. See the 
[help page](https://code.google.com/p/mdanalysis/wiki/TPRReaderDevelopment) 
about the `tpr` file format support in MDAnalysis. The code to load the 
fine-grained trajectory is:

    fine_uni = Universe("foo.tpr", "foo.trr")

Now we create a coarse-grained trajectory using the fine-grained trajectory as 
an input:

    coarse_uni =CGUniverse(fine_uni, selections=['name OW', 'name HW1 or name HW2'], 
                          names=['O', 'H2'], 
			  collapse_hydrogens=False)

The `selections` variable is an array of strings. Each string is a
Charmm atom selection string. Note, these are very similar to VMD
selection string. You may test them out using the following snippet::

    selected_atoms = fine_uni.selectAtoms('name OW')
    for a in selected_atoms:
        print a

In the example above water oxygen is the first string and water hydrogens
are the second string. The next variable, `names`, is optional and is
the array of names to be given to the selections. It is an array of
strings the same length as `selections`. If `names` is not given, then
the atoms in the resulting coarse-grained trajectory will have numbers
as their names. The last variable, `collapse_hydrogens`, can be `True`
or `False`. If it's `True`, then all hydrogens will automatically be
included in neighboring atoms which are selected. So, for example, if
you select a carbon, all its hydrogens will be included. Its default
is `True`.

Now that you have a coarse-grained trajectory, you may write out the
structure or trajectory using the following syntax:

    coarse_uni.write_structure("cg_foo.pdb")
    coarse_uni.write_structure("cg_foo.pdb", bonds='all')
    coarse_uni.write_trajectory("cg_foo.dcd")
    

The coarse-grained trajectory is also a valid MDAnalysis object and
you may perform any analysis techniques on it from the
[MDAnalysis](https://code.google.com/p/mdanalysis/) package.
    
Adding Bonds
--------------------

Bonding information isn't always included in pdb files. To add a bond
manually, use this line of code:

    add_residue_bonds(coarse_uni, 'name O', 'name H2')    

This will bond all atoms named `O` with all atoms named `H2` *within
each residue*.

Force-Matching
===================
Let's use again the example of 2-site water. If the original all-atom
trajectory had forces, then we may use this file for force-matching.

    from MDAnalysis import Universe
    from ForcePy import *
    import pickle
    
    fgu = Universe('topol.tpr', 'traj.trr')
    fgu.trajectory.periodic = True #NOTE: You MUST set this flag yourself, since there is no indication in the TPR files
    cgu = CGUniverse(fgu, ['name OW', 'name HW1 or name HW2'], ['O', 'H2'], False)
    add_residue_bonds(cgu, 'name O', 'name H2')
    fm = ForceMatch(cgu, kT=0.7) #kT is Bolzmann's constant times temperature in the units of the simulation
    
At this point, we have a `ForeMatch` object which contains the
coarse-grained universe. Now we need to set-up the force field which
will be force-matched to fit the coarse-grained trajectory forces
(which themselves came from the all-atom trajectory).

    ff = FileForce() #This just says the forces are found in the universe passed to the ForceMatch object
    #Set this force as the reference force to be matched
    fm.add_ref_force(ff)    

    pair_mesh = Mesh.UniformMesh(0,12,0.05) #This is the mesh on which the force-field will be built. It is in Angstroms
    pairwise_force = SpectralForce(Pairwise, pair_mesh, Basis.UnitStep)                  
    #Copy this force type and clone it for each pair-interaction type
    fm.add_and_type_pair(pairwise_force)
    #This is a harmonic bond that will be fixed to the energy minimum of the bonds with a harmonic constant of 500 kJ/mol
    bond_force = FixedHarmonicForce(Bond, 500, cutoff=1) 
    fm.add_and_type_pair(bond_force)
    
At this point, can now force match. To do it in serial:

    fm.force_match()
    
You may also pass an `iterations` argument to use less than the entire
trajectory. To do it in parallel (note you must have started using
mpirun, mpiexec, or aprun depending on your MPI environment)

    fm.force_match_mpi()

One thing about MPI runs is that it's much faster to pre-compute all
the trajectory mapping at the beginning so that it isn't repeated on
each node. This may be done by changing a line:

    cgu = CGUniverse(fgu, ['name OW', 'name HW1 or name HW2'], ['O', 'H2'], False)
    #this will precompute the cg trajectory at every frame; this may take some time
    cgu = cgu.cache()

Finally, to write out the a set of lammps scripts to use the new force field, run
    
    fm.write_lammps_scripts()

  
Architecture Notes
==================

The main class to utilize is a `ForceMatch` class. This class takes in
one or more reference `Force` objects that define the forces to
match. For example, a `FileForce` will read in forces from a file. The
`ForceMatch` class also takes one or more target `Force` objects,
which are the functional forms that are going to match the reference
forces. Some `Force` objects contain a static class variable that
points to a `ForceCategory` that contains useful
methods/variables. For example, the `Pairwise` contains a
neighborlist implementation.

Regularizers may be added to force objects as well by calling the
`add_regularizer` method.

The `SpectralForce` is a linear combination of basis functions. This
is usually a good choice. The `SpectralForce` requires a mesh and
basis function. Currently only `UniformMesh` is implemented. For the
basis functions, `UnitStep`, `Quartic`, and `Gaussian` are
implemented.

A given `Force` may be 'specialized' to work on only a certain type or
type pair. This may be done by calling `specialize_type` before it is
added to a `ForceMatch` class.

In order to simplify constructing potentials for many type pairs,
there are utility functions on the ForceMatch class to construct all
possible pairs. `add_and_type_pairs` copies a force as many times as
needed to have a unique force for every possible pair-pair
interaction.

Observable Variable
===============

*Under Construction*

The target forcefield/potential may be modified to reproduce some
observable parameter.  The observable should be in a tabular file
containing the total energy of the system at each frame in Column
1. Columns 2 and beyond should be the deviation of the observable. The
algorithm will try to minimize the observable. The observable under
the new forcefied will be sum_i O_i * exp(-(U' - U) * beta), where U'
is the new potential. The gradient at each frame will be -beta * U' *
exp(-(U' - U) * beta). The meshes implment integrated forms for
working with potentials. This gradient, by the way, depends on the
current potential correct since the derivative of the normalization
constant changes.


Meshes
============
*Uniform mesh

Basis functions
=============
*UnitStep
*Quartic
*Gaussian

Forces
=========
* FileForce
* LammpsFileForce
* SpectralPairwiseForce
* AnalyticForce
* LJForce
* FixedHarmonicForce

Regularizers
==========
* SmoothRegularizer
* L2Regularizer







