from forcematch.ForceMatch import *

def unit_step(x, mesh):
    result = np.zeros(len(mesh))
    result[mesh.mesh_index(x)] = 1
    return result

fm = ForceMatch("test/methanol.json")
ref_pcat = Pairwise()
tar_pcat = Pairwise()
#ref_pcat.addForce(PairwiseForce(lambda x: 4 * (6 * x**(-7) - 12 * x ** (-13) )))
ref_pcat.addForce(FileForce())
pwf = PairwiseSpectralForce(UniformMesh(0,10,0.01), 100, unit_step)
pwf.add_regularizer(SmoothRegularizer)
tar_pcat.addForce(pwf)
fm.add_tar_force_cat(tar_pcat)
fm.add_ref_force_cat(ref_pcat)
fm.force_match()
