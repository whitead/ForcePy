from MDAnalysis import Universe
from ForcePy import *
import random

cgu = CGUniverse(Universe("spc.tpr", "traj.trr"), ['name OW', 'name HW1 or name HW2'], ['O', 'H2'], collapse_hydrogens=False)
#cgu = CGUniverse(Universe("spc.tpr", "traj.trr"), ['all'], ['HOH'], collapse_hydrogens=False)
add_residue_bonds(cgu, "name O", "name H2")
fm = ForceMatch(cgu)
ff = FileForce()
pair_mesh = Mesh.UniformMesh(0.1,10,0.01)
pwf = SpectralForce(Pairwise, pair_mesh, Basis.Quartic(pair_mesh, 0.5))
bwf = FixedHarmonicForce(Bond, 450*4.14)

pwf.add_regularizer(SmoothRegularizer)
fm.add_ref_force(ff)
#fm.add_and_type_pair(pwf)
fm.add_and_type_states(pwf,lambda x: random.choice([[1,0],[0.01, 0.99]]), ["A", "B"])
fm.add_and_type_pair(bwf)
fm.force_match()
cgu.write_lammps_scripts(fm, folder='lammps', table_points=100, lammps_input_file="lammps.inp")
#fm.observation_match(obs_sweeps = 3, obs_samples = 5)
