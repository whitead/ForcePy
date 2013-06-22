from forcematch.ForceMatch import *

def unit_step(x, mesh):
    result = np.zeros(len(mesh) - 1)
    for m,i in zip(mesh[1:], range(len(mesh) - 1)):
        if(x < m):
            result[i] = 1
            return result
    return result

fm = ForceMatch("test/test.json")
ref_pcat = Pairwise()
tar_pcat = Pairwise()
ref_pcat.addForce(PairwiseForce(lambda x: 4 * (6 * x**(-7) - 12 * x ** (-13) )))
tar_pcat.addForce(PairwiseSpectralForce(np.arange(0,3,0.05), unit_step))
fm.add_tar_force_cat(tar_pcat)
fm.add_ref_force_cat(ref_pcat)
fm.force_match()
