from MDAnalysis import Universe
from ForcePy import *


fm = ForceMatch(Universe("cg.pdb", "cg.trr"))
fm.u.trajectory.periodic = True
ff = FileForce()
pmesh = Mesh.UniformMesh(2,12,0.1)
#pwf = SpectralForce(Pairwise, pmesh, Basis.UnitStep, build_repulsion(pmesh, 3, 500, 3))
pwf = SpectralForce(Pairwise, pmesh, Basis.UnitStep)
#pwf = SpectralForce(Pairwise, pmesh, Basis.Gaussian(pmesh, 0.1))
#pwf.add_regularizer(L2Regularizer)
#pwf.add_regularizer(SmoothRegularizer)
fm.add_ref_force(ff)
fm.add_and_type_pair(pwf)

fm.force_match(1)
fm.finalize()
fm.write('batch_1', table_points=100)
fm.finalize()

fm.force_match(10)
fm.finalize()
fm.write('batch_10', table_points=100)
fm.finalize()

fm.force_match(100)
fm.finalize()
fm.write('batch_100', table_points=100)
fm.finalize()

fm.force_match()
fm.finalize()
fm.write('batch_all', table_points=100)
