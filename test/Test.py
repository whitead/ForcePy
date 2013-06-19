from forcematch.ForceMatch import *

fm = ForceMatch("test/test.json")
pcat = Pairwise()
pcat.addForce(PairwiseForce(lambda x: x ** 2))
fm.add_tar_force_cat(pcat)
fm.add_ref_force_cat(pcat)
fm.force_match()
