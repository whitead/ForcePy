# cython: profile=False
# filename: Util.pyx

import numpy as np
cimport numpy as np
import cython
from libc.math cimport ceil, floor, sqrt

FTYPE = np.float32
ctypedef np.float32_t FTYPE_t

cdef FTYPE_t cround(FTYPE_t x):
    return ceil(x - 0.5) if x < 0. else floor(x + 0.5)        

@cython.boundscheck(False) # turn off bounds-checking for entire function
def min_img_vec(np.ndarray[FTYPE_t, ndim=1] x, np.ndarray[FTYPE_t, ndim=1] y, np.ndarray[FTYPE_t, ndim=1] img, bint periodic=True):
    cdef np.ndarray[FTYPE_t, ndim=1] dx = np.empty(3, dtype=FTYPE)
    cdef int i
    for i in range(3):
        dx[i] = x[i] - y[i]        
        if(periodic):
            dx[i] -= cround(dx[i] / img[i])
    return dx

@cython.boundscheck(False) # turn off bounds-checking for entire function
def min_img(np.ndarray[FTYPE_t, ndim=1] x, np.ndarray[FTYPE_t, ndim=1] img, bint periodic=True):
    cdef int i
    for i in range(3):
        x[i] -= floor(x[i] / img[i]) * img[i]
    return x

@cython.boundscheck(False) # turn off bounds-checking for entire function
cpdef FTYPE_t min_img_dist_sq(np.ndarray[FTYPE_t, ndim=1] x, np.ndarray[FTYPE_t, ndim=1] y, np.ndarray[FTYPE_t, ndim=1] img, bint periodic=True):
    cdef FTYPE_t dx
    cdef FTYPE_t dist = 0
    cdef int i
    for i in range(3):
        dx = x[i] - y[i]        
        if(periodic):
            dx -= cround(dx / img[i])
        dist += dx
    return dist

def min_img_dist(np.ndarray[FTYPE_t, ndim=1] x, np.ndarray[FTYPE_t, ndim=1] y, np.ndarray[FTYPE_t, ndim=1] img, bint periodic=True):
    return sqrt(min_img_dist_sq(x, y, img, periodic))

@cython.boundscheck(False) # turn off bounds-checking for entire function
cpdef double norm3(np.ndarray[FTYPE_t, ndim=1] x):
    return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])

def spec_force_inner_loop(np.ndarray[FTYPE_t, ndim=1] w, np.ndarray[FTYPE_t, ndim=1] basis_out, 
                          np.ndarray[FTYPE_t, ndim=2] grad, np.ndarray[FTYPE_t, ndim=1] force, 
                          np.ndarray[FTYPE_t, ndim=1] r):
    cdef FTYPE_t value
    for i in range(w.shape[0]):
        for j in range(r.shape[0]):
            force[j] = force[j] + w[i] * basis_out[i] * r[j]
            grad[i,j] = basis_out[i] * r[j] + grad[i,j]
