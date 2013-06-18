import numpy as np
import json
from MDAnalysis import *


class ForceMatch:
"""Main force match class.

"""

    def __init__(self, input_file):
        self.ref_force_cats = []
        self.tar_force_cats = []
        with open(input_file, 'r') as f:
            self.json = json.json.load(f)
        _test_json(self.json)
        self.u = Universe(self.json["topology"], self.json["trajectory"])
                
    def _test_json(self, json, required_keys = [("topology", "Toplogy file"), ("trajectory", "Trajectory File")]):
        for rk in required_keys:
            if(!json.hasKey(rk[0])):
                raise IOError("Error in input file, could not find %s" % rk[1])

    def add_tar_force_cat(self, *fcats):
        for(f in fcats):
            self.tar_force_cats.append(f)

    def add_ref_force_cat(self, *fcats):
        for(f in fcats):
            self.ref_force_cats.append(f)
        
    def force_match(self):

        if(len(self.ref_force_cats) != len(self.tar_force_cats)):
            raise RuntimeError("Must have same number of reference categories as target categories")

        ref_force = np.zeros( (3, self.u.atoms.numberOfAtoms()) )
        for((rf, tf) in zip(self.ref_force_cats, self.tar_force_cats)):
            rf.calcForces(ref_force, self.u)
            tf.update(ref_forces, self.u)


#abstract classes

class ForceCategory:
"""A category of force/potential type.

   The forces used in force matching are broken into categories, where
   the sum of each category of forces is what's matched in the force
   matching code. Examples of categories are pairwise forces,
   threebody forces, topology forces (bonds, angles, etc).
"""
    def calcForces(self, forces, u):
        """Calculate net forces in the given universe and stored in the passed numpy array. Assumes numpy is zeroed
        """ 
        self._setup(u)
        for(f in self.forces):
            f.calcForces(forces, u)
        self._teardown()

    def update(self, ref_forces, u):
        """Runs force matching update step"""
        #get sum of forces
        est_forces = zeroes( (3, u.atoms.numberOfAtoms()) )
        self._setup(u)
        for(f in self.forces):
            f.calcForces(est_forces)
        self._teardown()

        #now calculate difference
        loss = ref_forces - est_forces
        #update
        self._setup_update(u)
        for(f in self.forces):
            f.update(loss, u)
        self._teardown_update()

class Force:
"""Can calculate forces from a universe object
"""

    def update(self, loss, u):
        g = self.grad(u)
        self.w = self.w - self.eta / sqrt(self.lip) * g
        self.lip += g ** 2
    

#concrete class

class Pairwise(ForceCategory):
    
    def __init__(self):
        self.

    def _build_neighbor_list(self):
        pass

    def _setup(self):
        
        
