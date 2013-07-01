from forcematch import *
import numpy as np

def unit_step(x, mesh):
    result = np.zeros(len(mesh))
    result[mesh.mesh_index(x)] = 1
    return result


fm = ForceMatch("test/methanol.json")
ff = FileForce()
pwf = PairwiseSpectralForce(UniformMesh(0,12,0.1), unit_step)
pwf.specialize_types("name AR")
#pwf = LJForce(sigma=1.5, epsilon=0.9)
#pwf.add_regularizer(SmoothRegularizer, L2Regularizer)
fm.add_ref_force(ff)
fm.add_tar_force(pwf)
fm.force_match()
