from MDAnalysis import Universe
from ForcePy import *
import numpy as np


fm = ForceMatch(Universe("cg.pdb", "cg.trr"), 0.7)
fm.u.trajectory.periodic = True
ff = FileForce()
#pmesh = Mesh.UniformMesh(2.7,12.1,0.01)
pmesh = Mesh.UniformMesh(2.7,15,0.01)
pwf = SpectralForce(Pairwise, pmesh, Basis.UnitStep)

#load guess
gf = open('guess.dat')
guess_w = np.empty(len(pmesh), dtype=np.float32)
for i,l in enumerate(gf):
    assert pmesh[i] / 10. - float(l.split()[0]) < 0.005, "%d != float(%s)" % (pmesh[i] * 10., l.split()[0])
    guess_w[i] = -0.7 * float(l.split()[1])
#pwf._setup_update_params(len(pmesh), guess_w, eta=0)
    
pwf.add_regularizer(L2Regularizer)
pwf.add_regularizer(SmoothRegularizer)
fm.add_ref_force(ff)
#fm.add_tar_force(pwf)
fm.add_and_type_pair(pwf)

#fm.force_match()

#fm.plot_frequency = -1
#for i in range(100):
#    fm.force_match(1)

#for i in range(100):
#    fm._force_match_task(25,26, do_print=True)

for i in range(5):
    fm._force_match_task(50 * i, 50 * i + 25, do_print=True)
#    fm.force_match_mpi(batch_size=5, do_plots=True, quiet=False)
    fm.write_lammps_scripts(folder='batch_%d' % i, table_points=10000)


