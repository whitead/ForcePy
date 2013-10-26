Summary
=======

Force-matching and utilities package. Supports input from Gromacs and Lammps. Not
tested for input NAMD simulations. Outputs Lammps tabular potentials, input files, and plain text tabular potentials. 
Can be used independently for topology reduction in coarse-graining. 

## License

Copyright 2013

This code is provided as a preview of an upcoming licensed and peer-reviewed/published version. It is currently unlicensed, meaning modification, distribution, sublicensing and commercial use are forbidden.

Example 1: Coarse-graining a Trajectory of Water
==========

The ForcePy module can be used to coarse-grained a trajectory. In this example, we'll convert 
an all-atom water simulation to a 2-site water model.

The first step is to import the necessary libraries:

```python
from MDAnalysis import Universe
from ForcePy import *
```

Next, we load the fine-grained trajectory. The first file
is the structure (`pdb`, `tpr`, `gro`, or `psf`) and the
second file is the trajectory. Note that the `tpr` file reader depends highly
on what version of gromacs with which the `tpr` file was created. See the 
[help page](https://code.google.com/p/mdanalysis/wiki/TPRReaderDevelopment) 
about the `tpr` file format support in MDAnalysis. The code to load the 
fine-grained trajectory is:

```python
fine_uni = Universe("foo.tpr", "foo.trr")
fine_uni.trajectory.periodic = True #NOTE: You MUST set this flag yourself, since there is no indication in the TPR files
```

Now we create a coarse-grained trajectory using the fine-grained trajectory as 
an input:

```python
coarse_uni =CGUniverse(fine_uni, selections=['name OW', 'name HW1 or name HW2'], 
                      names=['O', 'H2'], 
                       collapse_hydrogens=False)
```

The `selections` variable is an array of strings. Each string is a
Charmm atom selection string. Note, these are very similar to VMD
selection string. You may test them out using the following snippet::

```python
selected_atoms = fine_uni.selectAtoms('name OW')
for a in selected_atoms:
    print a
```

In the example above water oxygen is the first string and water hydrogens
are the second string. The next variable, `names`, is optional and is
the array of names to be given to the selections. It is an array of
strings the same length as `selections`. If `names` is not given, then
the atoms in the resulting coarse-grained trajectory will have numbers
as their names. The last variable, `collapse_hydrogens`, can be `True`
or `False`. If it's `True`, then all hydrogens will automatically be
included in neighboring atoms which are selected. So, for example, if
you select a carbon, all its hydrogens will be included. Its default
is `False`.

Now that you have a coarse-grained trajectory, you may write out the
structure or trajectory using the following syntax:

```python
coarse_uni.write_structure("cg_foo.pdb")
coarse_uni.write_structure("cg_foo.pdb", bonds='all')
coarse_uni.write_trajectory("cg_foo.dcd")
```
    

The coarse-grained trajectory is also a valid MDAnalysis object and
you may perform any analysis techniques on it from the
[MDAnalysis](https://code.google.com/p/mdanalysis/) package.
    
Adding Bonds
--------------------

Bonding information isn't always included in pdb files. To add a bond
manually, use this line of code:

```python
add_residue_bonds(coarse_uni, 'name O', 'name H2')    
```

This will bond all atoms named `O` with all atoms named `H2` *within
each residue*. To add bonds between residues, use this line of code:

```python
add_sequential_bonds(coarse_uni, 'name C', 'name N')    
```

The two selection strings will be the bonded parts of each residue.

Example 2: Coarse-graining a Protein
==================

Here are some examples of coarse-graining a protein with increasingly
more coarse models

3-Beads per Residue
-----------

```python
from MDAnalysis import Universe
from ForcePy import *

protein_file = 'foo.pdb'
fine_uni = Universe(protein_file)

cgu_1 = CGUniverse(fine_uni, ['name O or C', 'name N or name CA', '((not name C) and (not name O)) and ((not name n) and (not name CA))'], ['O', 'CA', 'S'], collapse_hydrogens=True)

add_residue_bonds(cgu_1, 'name O', 'name CA')
add_residue_bonds(cgu_1, 'name CA', 'name S')
add_sequential_bonds(cgu_1, 'name CA', 'name CA')
write_structure(cgu_1, 'cg_1.pdb', bonds='all')
```

The first bead is the carbonyl group, the second is the C-alpha and
nitrogen, and the final is the side-chain.

1-Bead per Residue
---------

```python
cgu_2 = CGUniverse(fine_uni, ['all'])
add_sequential_bonds(cgu_2)
write_structure(cgu_2, 'cgu_2.pdb', bonds='all')
```

This is much simpler. The names are omitted for the beads and it is
not necessary to pass selection strings to `add_sequential_bonds()`
since there is only one atom in each residue.

3-Residues per Bead
--------------

Finally, here is how to put multiple residues into single beads. An
array must be passed to the CGUniverse constructor which has a length
of the desired number of beads and the array contains arrays of
indices corresponding to the fine-grain residue indices. For example,
to put residues `1,2,3` into a bead and residues `4,5` into another
bead this array will accomplish that: `[[1,2,3], [4,5]]`. Here is a
complete example of reducing every three residues into one bead:

```python
protein_length = len(fine_uni.residues)
reduction_map = [[3 * x, 3 * x + 1, 3 * x + 2] for x in range(protein_length / 3)]
cgu_3 = CGUniverse(fine_uni, ['all'],
                   residue_reduction_map=reduction_map)
add_sequential_bonds(cgu_3)
write_structure(cgu_3, 'cgu_3.pdb', bonds='all')
```


Force-Matching
===================
Let's use again the example of 2-site water. If the original all-atom
trajectory had forces, then we may use this file for force-matching.

```python
from MDAnalysis import Universe
from ForcePy import *
import pickle
    
fgu = Universe('topol.tpr', 'traj.trr')
fgu.trajectory.periodic = True #NOTE: You MUST set this flag yourself, since there is no indication in the TPR files
cgu = CGUniverse(fgu, ['name OW', 'name HW1 or name HW2'], ['O', 'H2'], False)
add_residue_bonds(cgu, 'name O', 'name H2')
fm = ForceMatch(cgu) 
```    

At this point, we have a `ForeMatch` object which contains the
coarse-grained universe. Now we need to set-up the force field which
will be force-matched to fit the coarse-grained trajectory forces
(which themselves came from the all-atom trajectory).

```python 
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
```
    
At this point, can now force match. To do it in serial:

```python    
fm.force_match()
```
    
You may also pass an `iterations` argument to use less than the entire
trajectory. To do it in parallel (note you must have started using
mpirun, mpiexec, or aprun depending on your MPI environment)

```python    
fm.force_match_mpi()
```

One thing about MPI runs is that it's much faster to pre-compute all
the trajectory mapping at the beginning so that it isn't repeated on
each node. This may be done by changing a line:

```python    
cgu = CGUniverse(fgu, ['name OW', 'name HW1 or name HW2'], ['O', 'H2'], False)
#this will precompute the cg trajectory at every frame; this may take some time
cgu = cgu.cache()
```

Finally, to write out the a set of lammps scripts to use the new force field, run

```python    
fm.write_lammps_scripts()
```
  

Installing ForcePy
===============

Dependencies
----------

* Python2.7
* scipy/numpy
* MDAnalysis (development branch)

Optional Dependencies
----------

* MPI4Py (for parallel force-matching)
* matplotlib (for plotting)

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


Meshes
------------
* Uniform mesh

Basis functions
------------
* UnitStep
* Quartic
* Gaussian

Forces
------------
* FileForce
* LammpsFileForce
* SpectralPairwiseForce
* AnalyticForce
* LJForce
* FixedHarmonicForce

Regularizers
------------
* SmoothRegularizer
* L2Regularizer

