from MDAnalysis import Universe
from ForcePy import *
import numpy as np

fm = ForceMatch(Universe("cg.pdb", "cg.trr"), 0.7)
ff = FileForce()
pwf = SpectralForce(Pairwise, Mesh.UniformMesh(0,10,0.1), Basis.UnitStep)
#pwf.add_regularizer(SmoothRegularizer)
fm.add_ref_force(ff)
fm.add_and_type_pair(pwf)
fm.force_match()
