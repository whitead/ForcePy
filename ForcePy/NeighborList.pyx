# cython: profile=False
# filename: NeighborList.pyx

import pstats, cProfile
import numpy as np
cimport numpy as np
import cython
import time
from libc.stdlib cimport malloc, free
from libc.math cimport ceil, floor, sqrt


DTYPE = np.int32
ctypedef np.int32_t DTYPE_t

FTYPE = np.float32
ctypedef np.float32_t FTYPE_t

cdef FTYPE_t cround(FTYPE_t x):
    return ceil(x - 0.5) if x < 0. else floor(x + 0.5)        


@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef FTYPE_t min_img_dist_sq(np.ndarray[FTYPE_t, ndim=1] x, np.ndarray[FTYPE_t, ndim=1] y, double* img, bint periodic=True):
    cdef FTYPE_t dx
    cdef FTYPE_t dist = 0
    cdef int i
    for i in range(3):
        dx = x[i] - y[i]        
        if(periodic):
            dx -= cround(dx / img[i]) * img[i]
        dist += dx * dx
    return dist


cdef class NeighborList(object):
    """Neighbor list class
    """
    cdef double cutoff
    cdef double* box
    cdef nlist_lengths
    cdef nlist
    cdef int* cell_number
    cdef cell_neighbors #use python object for simplicity and since this isn't yet a bottleneck
    cdef int* cells 
    cdef int* head
    cdef exclusion_list
    cdef int cell_number_total
    cdef bint exclude_14
    
    
    def __init__(self, u, cutoff, exclude_14 = True):
        
        #set up cell number and data
        
        self.cutoff = cutoff
        self.box = <double* > malloc(3 * sizeof(double))
        self.cell_number = <int* > malloc(3 * sizeof(int))
        self.cell_number_total = 1
        cdef i
        for i in range(3):
            self.box[i] = u.dimensions[i]
            self.cell_number[i] = max(1,int(self.box[i] / self.cutoff))
            self.cell_number_total *= self.cell_number[i]

        self.nlist_lengths = [0 for x in range(u.atoms.numberOfAtoms())]
        self.nlist = np.arange(u.atoms.numberOfAtoms() * (u.atoms.numberOfAtoms() - 1), dtype=DTYPE)

        self.cells = <int* > malloc(u.atoms.numberOfAtoms() * sizeof(int))
        self.head = <int*> malloc(self.cell_number_total  * sizeof(int))
        self.exclusion_list = None
        self.exclude_14 = exclude_14

        #pre-compute neighbors. Waste of space, but saves programming effort required for ghost cellls        
        self.cell_neighbors = [[] for x in range(self.cell_number_total)]
        #Leaving all this stuff as python objects because speed is not an issue here
        for xi in range(self.cell_number[0]):
            for yi in range(self.cell_number[1]):
                for zi in range(self.cell_number[2]):
                    #get neighbors
                    index = (xi * self.cell_number[0] + yi) * self.cell_number[1] + zi
                    index_vector = [xi, yi, zi]
                    neighs = [[] for x in range(3)]
                    for i in range(3):
                        neighs[i] = [self.cell_number[i] - 1 if index_vector[i] == 0 else index_vector[i] - 1,
                                     index_vector[i],
                                     0 if index_vector[i] == self.cell_number[i] - 1 else index_vector[i] + 1]
                    for xd in neighs[0]:
                        for yd in neighs[1]:
                            for zd in neighs[2]:
                                neighbor = xd * self.cell_number[0] ** 2 + \
                                                                       yd * self.cell_number[1]  + \
                                                                       zd
                                #check if neighbor is already in cell_neighbor
                                #this is possible if wrapped and cell number is 1
                                if(not neighbor in self.cell_neighbors[index]): 
                                    self.cell_neighbors[index].append(neighbor)
                        
    def __del__(self):
        free(self.box)
        free(self.cell_number)
        free(self.head)
        free(self.cells)

    @cython.boundscheck(False) #turn off bounds checking
    cdef bin_particles(self, u):
        cdef int i,j,icell
        cdef double k
        for i in range(self.cell_number_total):
            self.head[i] = -1

        positions = u.atoms.get_positions(copy=False)
        for i in range(u.atoms.numberOfAtoms()):

            icell = 0
            #fancy index and binning loop over dimensions
            for j in range(3):
                #sometimes things are unwrapped, better to assume they aren't
                k = positions[i][j]/ self.box[j] * self.cell_number[j]
                k = floor(k % self.cell_number[j])      
                icell =  <int> k + icell * self.cell_number[j]
            #push what is on the head into the cells

            self.cells[i] = self.head[icell]
            #add current value
            self.head[icell] = i


    cdef _build_exclusion_list(self, u):
        #what we're building
        self.exclusion_list = [[] for x in range(u.atoms.numberOfAtoms())]
        #The exclusion list at the most recent depth
        temp_list = [[] for x in range(u.atoms.numberOfAtoms())]
        #build 1,2 terms
        for b in u.bonds:
            self.exclusion_list[b.atom1.number].append(b.atom2.number)
            self.exclusion_list[b.atom2.number].append(b.atom1.number)
        # build 1,3 and 1,4
        for i in range( 1 if self.exclude_14 else 2):
            #copy
            temp_list[:] = self.exclusion_list[:]
            for a in range(u.atoms.numberOfAtoms()):
                for b in range(len(temp_list[a])):
                    self.exclusion_list[a].append(b) 

    @cython.boundscheck(False) #turn off bounds checking
    @cython.wraparound(False) #turn off negative indices
    cdef int _build_nlist(self, u):


        if(self.exclusion_list == None):
            self._build_exclusion_list(u)

        #bin the particles
        self.bin_particles(u)

        ntime = time.time()
        positions = u.atoms.get_positions(copy=False)
                                                                        
        cdef int i, j, nlist_count, icell
        cdef double k
        nlist_count = 0
        for i in range(u.atoms.numberOfAtoms()):
            self.nlist_lengths[i] = 0

        periodic = u.trajectory.periodic
        for i in range(u.atoms.numberOfAtoms()):
            icell = 0
            #fancy indepx and binning loop over dimensions
            for j in range(3):
                #sometimes things are unwrapped, better to assume they aren't
                k = positions[i][j]/ self.box[j] * self.cell_number[j]
                k = floor(k % self.cell_number[j])      
                icell =  int(k) + icell * self.cell_number[j]
            for ncell in self.cell_neighbors[icell]:
                j = self.head[ncell]
                while(j != - 1):
                    if(i != j and
                       not (j in self.exclusion_list[i]) and
                        min_img_dist_sq(positions[i], positions[j], self.box, periodic) < self.cutoff ** 2):
                        self.nlist[nlist_count] = j
                        self.nlist_lengths[i] += 1
                        nlist_count += 1
                    j = self.cells[j]

        return nlist_count

    def build_nlist(self, u):        

        return self.nlist[:self._build_nlist(u)], self.nlist_lengths


