from MDAnalysis import Universe
from ForcePy import *
import numpy as np

fm = ForceMatch(Universe("lj.pdb", "lj.xyz"), 1, "lj.json")
ff = LJForce(3)
mesh = Mesh.UniformMesh(0.5,3,0.005)
#pwf = SpectralForce(Pairwise, mesh, Basis.Gaussian(mesh, 0.1))
pwf = SpectralForce(Pairwise, mesh, Basis.UnitStep)
fm.add_ref_force(ff)
fm.add_and_type_pair(pwf)
fm.force_match()
fm.force_match()
fm.force_match()
fm.force_match()

