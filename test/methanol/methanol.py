from MDAnalysis import Universe
from ForcePy import *


fm = ForceMatch(Universe("cg.pdb", "cg.trr"))
fm.u.trajectory.periodic = True
ff = FileForce()
pmesh = Mesh.UniformMesh(2.7,15,0.1)
pwf = SpectralForce(Pairwise, pmesh, Basis.UnitStep)
#pwf.add_regularizer(L2Regularizer)
#pwf.add_regularizer(SmoothRegularizer)
fm.add_ref_force(ff)
fm.add_and_type_pair(pwf)

#fm.force_match()

#this code will show the force_matching in steps of 10
for i in range(5):
    fm._force_match_task(10 * i, 10 * (i + 1), do_print=True)
    fm.write_lammps_scripts(folder='iter_%d' % i, table_points=10000)
