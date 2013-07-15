# cython: profile=True
#filename Basis.pyx

import numpy as np
cimport numpy as np
import cython

FTYPE = np.float32
ctypedef np.float32_t FTYPE_t

class UnitStep(object):

    @staticmethod
    def force(FTYPE_t x, mesh):
        result = np.zeros(len(mesh), dtype=FTYPE)
        result[mesh.mesh_index(x)] = 1
        return result

    @staticmethod
    def force_cache(FTYPE_t x, np.ndarray[FTYPE_t, ndim=1] cache, mesh):    
        cache.fill(0)
        cache[mesh.mesh_index(x)] = 1



    @staticmethod
    def potential(FTYPE_t x, mesh):
        cdef int i, lm
        lm = len(mesh)
        result = np.zeros(lm, dtype=FTYPE)
        mesh_point = mesh.mesh_index(x)
        for i in range(lm - 1, mesh_point - 1, -1):
            result[i] = (mesh[i + 1] - mesh[i])
        return -result



