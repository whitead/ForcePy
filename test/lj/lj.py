from MDAnalysis import Universe
from ForcePy import *
import numpy as np

fm = ForceMatch(Universe("lj.pdb", "lj.xyz"), "lj.json")
ff = LJForce(3)
pwf = SpectralForce(Pairwise, Mesh.UniformMesh(0,3,0.025), Basis.UnitStep)
pwf.add_regularizer(SmoothRegularizer)
fm.add_ref_force(ff)
fm.add_and_type_pair(pwf)
fm.force_match()

