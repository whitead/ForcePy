from forcematch import *
import numpy as np

def unit_step(x, mesh):
    result = np.zeros(len(mesh))
    result[mesh.mesh_index(x)] = 1
    return result


ref_pcat = Pairwise()
tar_pcat = Pairwise(12)
#ref_pcat.addForce(LJForce())
ref_pcat.addForce(FileForce())
fm = ForceMatch("test/methanol.json", ref_pcat)
pwf = PairwiseSpectralForce(UniformMesh(0,12,0.1), unit_step)
#pwf = LJForce(sigma=1.5, epsilon=0.9)
#pwf.add_regularizer(SmoothRegularizer, L2Regularizer)
tar_pcat.addForce(pwf)
fm.add_tar_force_cat(tar_pcat)
fm.force_match()
