from Forces import Force

import numpy as np
import json
from math import ceil
from MDAnalysis import Universe



class ForceMatch:
    """Main force match class.
    """
    
    def __init__(self, input_file):
        self.ref_force_cats = []
        self.tar_force_cats = []
        with open(input_file, 'r') as f:
            self.json = json.load(f)
        self._test_json(self.json)
        self.u = Universe(self.json["structure"], str(self.json["trajectory"]))

                
    def _test_json(self, json, required_keys = [("structure", "Toplogy file"), ("trajectory", "Trajectory File")]):
        for rk in required_keys:
            if(not json.has_key(rk[0])):
                raise IOError("Error in input file, could not find %s" % rk[1])

    def add_tar_force_cat(self, *fcats):
        for f in fcats:
            self.tar_force_cats.append(f)

    def add_ref_force_cat(self, *fcats):
        for f in fcats:
            self.ref_force_cats.append(f)
        
    def force_match(self):

        if(len(self.ref_force_cats) != len(self.tar_force_cats)):
            raise RuntimeError("Must have same number of reference categories as target categories")

        ref_forces = np.zeros( (self.u.atoms.numberOfAtoms(), 3) )
        for ts in self.u.trajectory:
            for (rf, tf) in zip(self.ref_force_cats, self.tar_force_cats):
                rf.calc_forces(ref_forces, self.u)
                tf.update(ref_forces, self.u)
            ref_forces.fill(0)
            


#abstract classes

class ForceCategory:
    """A category of force/potential type.
    
    The forces used in force matching are broken into categories, where
    the sum of each category of forces is what's matched in the force
    matching code. Examples of categories are pairwise forces,
   threebody forces, topology forces (bonds, angles, etc).
   """
    
    def addForce(self, force):
        self.forces.append(force)
        force._register_hook(self)

    def calc_forces(self, forces, u):
        """Calculate net forces in the given universe and stored in the passed numpy array. Assumes numpy is zeroed
        """ 
        self._setup(u)
        for f in self.forces:
            f.calc_forces(forces, u)
        self._teardown()

    def update(self, ref_forces, u):
        """Runs force matching update step"""
        #update
        self._setup_update(u)
        for f in self.forces:            
            f.update(ref_forces, u)
        self._teardown_update()



class NeighborList:
    """Neighbor list class
    """
    def __init__(self, u, cutoff):
        
        #set up cell number and data
        self.cutoff = cutoff
        self.box = [(0,x) for x in u.dimensions[:3]]
        self.nlist_lengths = np.arange(0, dtype='int32')
        self.nlist = np.arange(0, dtype='int32')
        self.cell_number = [int(ceil((x[1] - x[0]) / self.cutoff)) for x in self.box]
        self.bins_ready = False
        self.cells = np.empty(u.atoms.numberOfAtoms(), dtype='int32')
        self.head = np.empty( reduce(lambda x,y: x * y, self.cell_number), dtype='int32')

        #pre-compute neighbors. Waste of space, but saves programming effort required for ghost cellls
        self.cell_neighbors = [[] for x in range(len(self.head))]
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
                                self.cell_neighbors[index].append(neighbor)
        

    def bin_particles(self, u):
        self.head.fill(-1)
        positions = u.atoms.get_positions(copy=False)
        positions = u.atoms.coordinates()
        for i in range(u.atoms.numberOfAtoms()):
            icell = 0
            #fancy index and binning loop over dimensions
            for j in range(3):
                icell = int((positions[i][j] - self.box[j][0]) / (self.box[j][1] - self.box[j][0]) * self.cell_number[j]) + icell * self.cell_number[j]                
            #push what is on the head into the cells
            self.cells[i] = self.head[icell]
            #add current value
            self.head[icell] = i

        self.bins_ready = True

    def build_nlist(self, u):

        self.nlist = np.resize(self.nlist, u.atoms.numberOfAtoms() * u.atoms.numberOfAtoms())
        #check to see if nlist_lengths exists yet
        if(len(self.nlist_lengths) != u.atoms.numberOfAtoms() ):
            self.nlist_lengths.resize(u.atoms.numberOfAtoms())
        
        if(not self.bins_ready):
            self.bin_particles(u)

        self.nlist_lengths.fill(0)
        positions = u.atoms.get_positions(copy=False)
        nlist_count = 0
        for i in range(u.atoms.numberOfAtoms()):
            icell = 0
            #fancy indepx and binning loop over dimensions
            for j in range(3):
                icell = int((positions[i][j] - self.box[j][0]) / (self.box[j][1] - self.box[j][0]) * self.cell_number[j]) + icell * self.cell_number[j]                
            for ncell in self.cell_neighbors[icell]:
                j = self.head[ncell]
                while(j != - 1):
                    if(i != j and np.sum((positions[i] - positions[j])**2) < self.cutoff ** 2):
                        self.nlist[nlist_count] = j
                        self.nlist_lengths[i] += 1
                        nlist_count += 1
                    j = self.cells[j]

        self.nlist = self.nlist[:nlist_count]
        return self.nlist, self.nlist_lengths

        

class Pairwise(ForceCategory):
    """Pairwise force category. It handles constructing a neighbor-list at each time-step. 
    """
    
    def __init__(self, cutoff=12):
        self.cutoff = cutoff                    
        self.forces = []
        self.nlist_ready = False
        self.nlist_obj = None

    def _build_nlist(self, u):
        if(self.nlist_obj is None):
            self.nlist_obj = NeighborList(u, self.cutoff)
        self.nlist, self.nlist_lengths = self.nlist_obj.build_nlist(u)

        self.nlist_ready = True

                    

    def _setup(self, u):
        if(not self.nlist_ready):
            self._build_nlist(u)

    def _teardown(self):
        self.nlist_ready = False
        
    def _setup_update(self,u):
        self._setup(u)


    def _teardown_update(self):
        self._teardown()

