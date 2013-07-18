from MDAnalysis import Universe
from ForcePy import *
import numpy as np


    
cgu = CGUniverse(Universe("test/water/spc.tpr", "test/water/traj.trr"), ['name OW', 'name HW1 or name HW2'], ['O', 'H2'], collapse_hydrogens=False)
#cgu = CGUniverse(Universe("test/water/spc.tpr", "test/water/traj.trr"), ['all'], ['HOH'], collapse_hydrogens=False)
cgu.add_residue_bonds("name O", "name H2")
fm = ForceMatch(cgu, "test/water/spc_cg.json")
ff = FileForce()
pair_mesh = UniformMesh(0.1,10,0.1)
pwf = SpectralForce(Pairwise, pair_mesh, Basis.Quartic(pair_mesh, 0.5))
bwf = FixedHarmonicForce(Bond, 450, cutoff=1)

pwf.add_regularizer(SmoothRegularizer)
fm.add_ref_force(ff)
fm.add_and_type_pair(pwf)
fm.add_and_type_pair(bwf)
fm.force_match(10)
cgu.write_lammps_scripts(fm, folder='lammps', table_points=1000, lammps_input_file="test/water/lammps.inp")
#fm.observation_match(obs_sweeps = 3, obs_samples = 5)
