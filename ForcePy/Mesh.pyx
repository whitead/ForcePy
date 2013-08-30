# cython: profile=False
#filename Mesh.pyx

from libc.math cimport ceil, floor
import numpy as np
cimport numpy as np


FTYPE = np.float32
ctypedef np.float32_t FTYPE_t


cdef class UniformMesh(object):
    """Uniform mesh. 
    """
    cdef FTYPE_t l
    cdef FTYPE_t r
    cdef FTYPE_t __dx
    cdef int length


    def __init__(self, l, r, dx):
        self.l = l
        self.r = r
        self.__dx = dx
        self.length = int(ceil((self.r - self.l) / self.__dx))

    cpdef FTYPE_t max(self):
        return self.r

    cpdef FTYPE_t min(self):
        return self.l
    
    cpdef int mesh_index(self, FTYPE_t x):
        return max(0, min(self.length - 1, int(floor( (x - self.l) / self.__dx) )))

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return i * self.__dx + self.l    

    cpdef FTYPE_t cgetitem(self, int i):
        return i * self.__dx + self.l    

    property dx:

        def __get__(self):
            return self.__dx
        def __set__(self, value):
            self.__dx = value


    def __reduce__(self):
        return UniformMesh, (self.l, self.r, self.dx)

