#!/usr/bin/python

from MDAnalysis import Universe
from ForcePy import *
import numpy as np
import pickle

fm = ForceMatch(Universe("lj.pdb", "lj.xyz"), 1, "lj.json")
ff = LJForce(3)
mesh = Mesh.UniformMesh(0,3,0.005)
#pwf = SpectralForce(Pairwise, mesh, Basis.Gaussian(mesh, 0.1))
pwf = SpectralForce(Pairwise, mesh, Basis.UnitStep)
pwf.add_regularizer(SmoothRegularizer)
fm.add_ref_force(ff)
fm.add_and_type_pair(pwf)
fm.force_match_mpi(do_plots=True, repeats=5, batch_size=5)
