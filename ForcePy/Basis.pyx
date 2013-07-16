# cython: profile=False
#filename Basis.pyx

"""
force: returns force
force_and_grad: puts the forces in cache, the Nx3 grad in the grad vector. Zeros cache, doesn't modify grad, returns magnitude of pair-wise distnace vector

"""



import numpy as np
cimport numpy as np
from ForcePy.Mesh import *
import cython
from libc.math cimport sqrt, pow



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


cdef class Quartic(object):

    #The number of non-zero neighbor bins which must be evaluated 
    cdef int basis_n
    #inverse of the width, needed for scaling
    cdef FTYPE_t inv_width 

    def __init__(self, mesh, width = None):
        """ Construct a Quartic basis.  The mesh must be given so that
            the Quartice mesh can optimize its layout on the mesh. The
            width should be from the left edge to right edge of the basis
        """

        #only works on uniform mesh
        assert type(mesh) is UniformMesh, "Quartic basis only works on uniform mesh currently and not %s" % type(mesh)

        self.basis_n = 0
        cdef int i        

        if(width is None or width < mesh.dx):
            width = mesh.dx
        else:
           #count neighbor non-zero bins in addition to main bin
            #Not sure why I don't calculate this directly, but
            #I think this snippet may be useful for adaptive mehses.
            for i in range(1, len(mesh)):
                if(mesh.cgetitem(i) - mesh.min() < width / 2.):
                    self.basis_n += 1
                else:
                    break

            print self.basis_n
        self.inv_width = (2. / width)
        
        
    cdef inline FTYPE_t _basis(self, FTYPE_t x, FTYPE_t left_edge):
        #Assumes we're given the left edge, instead of center, hence -1       
        x = self.inv_width * (x - left_edge) - 1
        return (15. / 16.) * (1. - x * x)  * (1. - x * x) 

    cdef inline FTYPE_t _int_basis(self, FTYPE_t x, FTYPE_t left_edge):
        #Assumes we're given the left edge, instead of center, hence -1
        x = self.inv_width * (x - left_edge) - 1
        return 3. / 16. * x**5 - 5./8 * x ** 3 + 15. / 16 * x
        
    def force(self, FTYPE_t x, mesh):
        result = np.zeros(len(mesh), dtype=FTYPE)
        self.force_cache(x, result, mesh)
        return result

    cpdef force_cache(self, FTYPE_t x, np.ndarray[FTYPE_t, ndim=1] cache, mesh):
        cache.fill(0)
        cdef int index = mesh.mesh_index(x)
        
        cache[index] = self._basis(x, mesh.cgetitem(index))

        cdef int i
        #upwards on mesh
        for i in range(index + 1, min(len(mesh), index + self.basis_n + 1)):
            cache[i] = self._basis(x, mesh.cgetitem(i))
        #downwards on mesh
        for i in range(index - 1, max(0, index - self.basis_n - 1), -1):
            cache[i] = self._basis(x, mesh.cgetitem(i))
        

    cpdef np.ndarray[FTYPE_t, ndim=1] potential(self, FTYPE_t x, mesh):
        cdef int i, lm
        lm = len(mesh)
        result = np.zeros(lm, dtype=FTYPE)
        mesh_point = mesh.mesh_index(x)
        for i in range(lm - 1, mesh_point, -1):
            result[i] = (mesh.cgetitem(i + 1) - mesh.cgetitem(i))
        result[mesh_point] = self._int_basis(x, mesh.cgetitem(mesh_point))
        return -result

