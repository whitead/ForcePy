import numpy as np
import numpy.linalg as ln
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
        for f in fcats:
            self.tar_force_cats.append(f)

    def add_ref_force_cat(self, *fcats):
        for f in fcats:
            self.ref_force_cats.append(f)
        
    def force_match(self):

        if(len(self.ref_force_cats) != len(self.tar_force_cats)):
            raise RuntimeError("Must have same number of reference categories as target categories")

        ref_force = np.zeros( (3, self.u.atoms.numberOfAtoms(), 3) )
        for (rf, tf) in zip(self.ref_force_cats, self.tar_force_cats):
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
    
    def _addForce(self, force):
        self.forces.append(force)

    def calcForces(self, forces, u):
        """Calculate net forces in the given universe and stored in the passed numpy array. Assumes numpy is zeroed
        """ 
        self._setup(u)
        for f in self.forces:
            f.calcForces(forces, u)
        self._teardown()

    def update(self, ref_forces, u):
        """Runs force matching update step"""
        #get sum of forces
        est_forces = zeros( (3, u.atoms.numberOfAtoms(), 3) )
        self._setup(u)
        for f in self.forces:
            f.calcForces(est_forces)
        self._teardown()

        #now calculate difference
        loss = ref_forces - est_forces
        #update
        self._setup_update(u)
        for f in self.forces:
            f.update(loss, u)
        self._teardown_update()

class Force:
    """Can calculate forces from a universe object
    """
    
    def _init_hook(self, category):
        """Register the force with the category and keep a reference to it
        """
        category._addForce(self)
        self.category = category
    
    def update(self, loss, u):
        g = self.grad(u)
        self.w = self.w - self.eta / sqrt(self.lip) * g
        self.lip += g ** 2
    

#concrete class

class Pairwise(ForceCategory):
"""Pairwise force category. It handles constructing a neighbor-list at each time-step. 
"""
    
    def __init__(self, cutoff=3):
        self.cutoff = cutoff                    

    def _build_nlist(self, u):
        self.nlist = np.arange(0)
        #check to see if nlist_lengths exists yet
        try:
            if(len(self.nlist_lengths) != u.atoms.numberOfAtoms() ):
                self.nlist_lengths = np.zeros( (u.atoms.numberOfAtoms()) )
        except NameError:
            self.nlist_lengths = np.zeros( (u.atoms.numberOfAtoms()) )

        self.nlist_lengths[0:len(self.nlist_lengths)] = 0
        positions = u.atoms.get_positions(copy=False)
        for i in range(u.atoms.numberOfAtoms()):
            for j in range(i):
                if((positions[i] - positions[j]) < self.cutoff ** 2):
                    self.nlist.append(j)
                    self.nlist_lengths[i] += 1
        self.nlist_ready = True        
                
                

    def _setup(self, u):
        if(not self.nlist_ready):
            self._build_nlist(u)

    def _teardown(self, u):
        self.nlist_ready = False
        
    def _setup_update(self,u):
        self._setup(u)


    def _teardown_update(self, u):
        self._teardown(u)

    
    
class PairwiseForce(Force):
    """ A pairwise force that takes in a function for calculating the force. The function passed should
    accept the scalar distance between the two particles as its first argument and any additional arguments
    it requires should be passed after the function.
    """
    def __init__(self, category, f, *args):
        self._init_hook(category)
        self.call_force = f
        self.call_force_args = args
        

    def calcForces(self, forces, u):
        if(not (grad is None) ):
            raise NotImplementedError("No gradient implementation for Lennard-Jones force")
        positions = u.atoms.
        nlist_accum = 0
        for i in u.atoms.numberOfAtoms():
            for j in self.nlist[nlist_accum:(nlist_accum + self.nlist_lengths[i])]:
                r = positions[j] - positions[i]
                d = ln.norm(r)
                force = self.call_force(d,*self.call_force_args) * (r / d)
                forces[i] += force
                forces[j] -= force
                nlist_accum += self.nlist_lengths[i]
