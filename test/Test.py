from forcematch import *
import numpy as np

def unit_step(x, mesh):
    result = np.zeros(len(mesh))
    result[mesh.mesh_index(x)] = 1
    return result

fm = ForceMatch("test/test.json")
ref_pcat = Pairwise()
tar_pcat = Pairwise(5)
ref_pcat.addForce(LJForce())
#ref_pcat.addForce(FileForce())
pwf = PairwiseSpectralForce(UniformMesh(0,5,0.25), unit_step)
#pwf.add_regularizer(SmoothRegularizer)
tar_pcat.addForce(pwf)
fm.add_tar_force_cat(tar_pcat)
fm.add_ref_force_cat(ref_pcat)
fm.force_match()
