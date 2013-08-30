#!/usr/bin/python

from MDAnalysis import Universe
from ForcePy import *
import numpy as np
import pickle

fm = ForceMatch(Universe("lj.pdb", "lj.xyz"), "lj.json")
ff = LJForce(3)
mesh = Mesh.UniformMesh(0,3,0.025)
pwf = SpectralForce(Pairwise, mesh, Basis.Gaussian(mesh, 1))
pwf.add_regularizer(SmoothRegularizer)
fm.add_ref_force(ff)
fm.add_and_type_pair(pwf)

fm.force_match_mpi(5, do_plots=True)
