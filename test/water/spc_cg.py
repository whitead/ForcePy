from MDAnalysis import Universe
from forcematch import *
import numpy as np

def unit_step(x, mesh):
    result = np.zeros(len(mesh))
    result[mesh.mesh_index(x)] = 1
    return result

def int_unit_step(x, mesh):
    result = np.zeros(len(mesh))
    mesh_point = mesh.mesh_index(x)
    for i in range(len(mesh) - 1, mesh_point - 1, -1):
        result[i] = (mesh[i + 1] - mesh[i])
    return -result
    
cgu = CGUniverse(Universe("test/water/spc.tpr", "test/water/traj.trr"), ['name OW', 'name HW1 or name HW2'], ['O', 'H2'], collapse_hydrogens=False)
cgu.add_residue_bonds("name O", "name H2")
fm = ForceMatch(cgu, "test/water/spc_cg.json")
ff = FileForce()
pwf = SpectralForce(Pairwise, UniformMesh(0,10,0.05), unit_step)
bwf = SpectralForce(Bond, UniformMesh(0,1,0.002), unit_step)
pwf.set_potential(int_unit_step)
bwf.set_potential(int_unit_step)
pwf.add_regularizer(SmoothRegularizer)
bwf.add_regularizer(SmoothRegularizer)
fm.add_ref_force(ff)
fm.add_and_type_pair(pwf)
fm.add_and_type_pair(bwf)
fm.force_match(5)
fm.observation_match(obs_sweeps = 3, obs_samples = 5)
