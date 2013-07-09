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
    

fm = ForceMatch("test/lj.json")
#ff = FileForce()
ff = LJForce(3)
pwf = PairwiseSpectralForce(UniformMesh(0,3,0.01), unit_step)
pwf.set_potential(int_unit_step)
#pwf = LJForce(sigma=1.5, epsilon=0.9)
pwf.add_regularizer(SmoothRegularizer)
fm.add_ref_force(ff)
fm.add_and_type_pair(pwf)
fm.force_match(20)
fm.observation_match(10, obs_samples=25)
