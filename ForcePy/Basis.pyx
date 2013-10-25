# cython: profile=False
#filename Basis.pyx

"""
force: returns force
force_and_grad: puts the forces in cache, the Nx3 grad in the grad vector. Zeros cache, doesn't modify grad, returns magnitude of pair-wise distnace vector

"""



import numpy as np
cimport numpy as np
from .Mesh import *
import cython
from libc.math cimport sqrt, pow , floor, ceil, exp, erf, acos

FTYPE = np.float32
ctypedef np.float32_t FTYPE_t

cdef FTYPE_t pi = 3.14159265

class UnitStep(object):

    @staticmethod
    def force(FTYPE_t x, mesh):
        result = np.zeros(len(mesh), dtype=FTYPE)
        result[mesh.mesh_index(x)] = 1
        return result

    @staticmethod
    def force_cache(FTYPE_t x, np.ndarray[FTYPE_t, ndim=1] cache, mesh):    
        cache.fill(0)
        i = mesh.mesh_index(x)
        assert i < len(mesh), "Attempted calculation outside of mesh. {} is beyond [{},{}]. Index = {} ( > {})".format(x, mesh.min(), mesh.max(), i, len(mesh))
        cache[mesh.mesh_index(x)] = 1                        

    @staticmethod
    def potential(FTYPE_t x, mesh):
        cdef int i, lm
        lm = len(mesh)
        result = np.zeros(lm, dtype=FTYPE)
        mesh_point = mesh.mesh_index(x)
        for i in range(lm - 1, mesh_point - 1, -1):
            result[i] = (mesh[i + 1] - mesh[i])
        return result


cdef class Quartic(object):

    #The number of non-zero neighbor bins which must be evaluated 
    cdef int basis_n
    #inverse of the width, needed for scaling
    cdef FTYPE_t inv_width 

    def __init__(self, mesh = None, width = None, *pickle_args):
        """ Construct a Quartic basis.  The mesh must be given so that
            the Quartice mesh can optimize its layout on the mesh. The
            width should be from the left edge to right edge of the basis
        """
 
        #only works on uniform mesh
        if(mesh):
            assert type(mesh) is UniformMesh, "Quartic basis only works on uniform mesh currently and not %s" % type(mesh)

            self.basis_n = 0

            if(width is None or width < mesh.dx * 1.5):
                width = mesh.dx * 1.5

            #count neighbor non-zero bins in addition to main bin
            self.basis_n = <int> ceil(width / mesh.dx)
        
            self.inv_width = (2. / width)
        #check for pickling
        else:            
            self.basis_n = pickle_args[0]
            self.inv_width = pickle_args[1]

        
    cdef inline FTYPE_t _basis(self, FTYPE_t x, FTYPE_t left_edge):
        #Assumes we're given the left edge, instead of center, hence -1       
        x = self.inv_width * (x - left_edge) - 1
        if(abs(x) >= 1):
            return 0
        return (15. / 16.) * (1. - x * x)  * (1. - x * x) 

    cdef inline FTYPE_t _int_basis(self, FTYPE_t x, FTYPE_t left_edge):
        #Assumes we're given the left edge, instead of center, hence -1
        x = self.inv_width * (x - left_edge) - 1
        if(x < -1):
            return 1
        elif(x > 1):
            return 0
        return -1. / 16. * (x - 1)**3 * (3 * x**2 + 9 * x + 8)
        
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
        for i in range(index - 1, max(-1, index - self.basis_n - 1), -1):
            cache[i] = self._basis(x, mesh.cgetitem(i))
        

    cpdef np.ndarray[FTYPE_t, ndim=1] potential(self, FTYPE_t x, mesh):
        cdef int i, lm, maxb
        lm = len(mesh)

        result = np.zeros(lm, dtype=FTYPE)
        mesh_point = mesh.mesh_index(x)        
        maxb = min(lm - 1, mesh_point + self.basis_n) # the point at which we must evaluate numerically

        for i in range(lm - 1, maxb, -1):
            result[i] = mesh.dx
        for i in range(maxb, max(-1, mesh_point - self.basis_n - 1), -1):
            result[i] = mesh.dx * self._int_basis(x, mesh.cgetitem(i))
        return result

    def __reduce__(self):
        return Quartic, (None, None, self.basis_n, self.inv_width)




cdef class Gaussian(object):

    #The number of non-zero neighbor bins which must be evaluated 
    cdef int basis_n
    #gaussian sigma inverse
    cdef FTYPE_t inv_sigma

    def __init__(self, mesh = None, sigma = None, *pickle_args):
        """ Construct a Guassian basis.  The mesh must be given so
            that the Gaussian mesh can optimize its layout on the
            mesh. 
        """
 
        #only works on uniform mesh
        if(mesh):
            assert type(mesh) is UniformMesh, "Gaussian basis only works on uniform mesh currently and not %s" % type(mesh)

            self.basis_n = 0

            if(sigma is None or sigma < mesh.dx * 1.5):
                sigma = mesh.dx * 1.5

            #count neighbor non-zero bins in addition to main bin
            self.basis_n = <int> ceil(4 * sigma / mesh.dx)
        
            self.inv_sigma = (1. / sigma)
        #check for pickling
        else:            
            self.basis_n = pickle_args[0]
            self.inv_sigma = pickle_args[1]

        
    cdef inline FTYPE_t _basis(self, FTYPE_t x, FTYPE_t left_edge):

        x = self.inv_sigma * (x - left_edge) - 0.5
        if(abs(x) >= 3.5):
            return 0
        return 1 / sqrt(2 * pi) * exp(-0.5 * x ** 2)

    cdef inline FTYPE_t _int_basis(self, FTYPE_t x, FTYPE_t left_edge):

        x = self.inv_sigma * (x - left_edge) - 0.5
        if(x < -3.5):
            return 1
        elif(x > 3.5):
            return 0
        return 1 - (0.5  + 0.5 * erf(x / sqrt(2.)))
        
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
        for i in range(index - 1, max(-1, index - self.basis_n - 1), -1):
            cache[i] = self._basis(x, mesh.cgetitem(i))
        

    cpdef np.ndarray[FTYPE_t, ndim=1] potential(self, FTYPE_t x, mesh):
        cdef int i, lm, maxb
        lm = len(mesh)

        result = np.zeros(lm, dtype=FTYPE)
        mesh_point = mesh.mesh_index(x)        
        maxb = min(lm - 1, mesh_point + self.basis_n) # the point at which we must evaluate numerically

        for i in range(lm - 1, maxb, -1):
            result[i] = mesh.dx
        for i in range(maxb, max(-1, mesh_point - self.basis_n - 1), -1):
            result[i] = mesh.dx * self._int_basis(x, mesh.cgetitem(i))
        return result

    def __reduce__(self):
        return Gaussian, (None, None, self.basis_n, self.inv_sigma)

