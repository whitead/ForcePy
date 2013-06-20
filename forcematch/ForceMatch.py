import numpy as np
import random
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
            self.json = json.load(f)
        self._test_json(self.json)
        self.u = Universe(self.json["structure"], self.json["trajectory"])
                
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
        for (rf, tf) in zip(self.ref_force_cats, self.tar_force_cats):
            rf.calc_forces(ref_forces, self.u)
            tf.update(ref_forces, self.u)


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



class Force:
    """Can calculate forces from a universe object
    """
    
    def _register_hook(self, category):
        """Register the force with the category and keep a reference to it
        """
        self.category = category
        self.sample = np.arange(0)
        #if this is an updatable force, set up stuff for it
        try:
            self.lip = np.zeros( np.shape(self.w) )
            self.grad = np.zeros( np.shape(self.w) )
            self.eta = 1
        except AttributeError:
            pass#not updatable. Oh well

    
    def update(self, ref_forces, u):
        #sample particles and run updates on them
        for i in random.sample(range(u.atoms.numberOfAtoms()), u.atoms.numberOfAtoms()):
            g = self._calc_grad(i, ref_forces[i], u)
            self.lip += np.square(g)
            self.w = self.w - self.eta / np.sqrt(self.lip) * g
            
            
    

#concrete class

class Pairwise(ForceCategory):
    """Pairwise force category. It handles constructing a neighbor-list at each time-step. 
    """
    
    def __init__(self, cutoff=3):
        self.cutoff = cutoff                    
        self.forces = []
        self.nlist_ready = False
        self.nlist_lengths = np.arange(0)

    def _build_nlist(self, u):
        self.nlist = np.arange(0)
        #check to see if nlist_lengths exists yet
        if(len(self.nlist_lengths) != u.atoms.numberOfAtoms() ):
            self.nlist_lengths.resize(u.atoms.numberOfAtoms())
                                      
        self.nlist_lengths.fill(0)
        positions = u.atoms.get_positions(copy=False)
        for i in range(u.atoms.numberOfAtoms()):
            for j in range(u.atoms.numberOfAtoms()):
                if(i != j and np.sum((positions[i] - positions[j])**2) < self.cutoff ** 2):                    
                    self.nlist = np.append(self.nlist, j)
                    self.nlist_lengths[i] += 1
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

    
    
class PairwiseForce(Force):
    """ A pairwise force that takes in a function for calculating the force. The function passed should
    accept the scalar distance between the two particles as its first argument and any additional arguments
    it requires should be passed after the function.
    """
    def __init__(self, f, *args):
        self.call_force = f
        self.call_force_args = args

        

    def calc_forces(self, forces, u):
        positions = u.atoms.get_positions()
        nlist_accum = 0
        for i in range(u.atoms.numberOfAtoms()):
            for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
                r = positions[j] - positions[i]
                d = ln.norm(r)
                force = self.call_force(d,*self.call_force_args) * (r / d)
                print "Applying force %g to particles %d and %d" % (ln.norm(force), i, j)
                forces[i] += force
            nlist_accum += self.category.nlist_lengths[i]

class PairwiseSpectralForce(Force):
    """A pairwise force that is a linear combination of basis functions

    The basis function should take two arguments: the distance and the
    mesh. Additional arguments after the function pointer will be
    passed after the two arguments. For example, the function may be
    defined as so: def unit_step(x, mesh, height)
    """
    
    def __init__(self, mesh, f, *args):
        self.call_basis = f
        self.call_basis_args = args
        self.mesh = mesh
        #create weights 
        self.w = np.ones(len(mesh) - 1)
        self.temp_grad = np.asmatrix(np.zeros( (len(mesh) - 1, 3) ))

    
    def calc_forces(self, forces, u):        
        positions = u.atoms.get_positions()
        nlist_accum = 0        
        for i in range(u.atoms.numberOfAtoms()):
            for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
                r = positions[j] - positions[i]
                d = ln.norm(r)
                force = self.w.dot(self.call_basis(d, self.mesh, *self.call_basis_args)) * (r / d)
                print "Applying force %g to particles %d and %d" % (ln.norm(force), i, j)
                forces[i] += force
                nlist_accum += self.category.nlist_lengths[i]
                
    def _calc_grad(self, i, force, u):        
        positions = u.atoms.get_positions()
        nlist_accum = np.sum(self.category.nlist_lengths[:(i - 1)]) if i > 0  else 0
        accu_force = force.reshape( (3,1) )
        
        self.grad.fill(0)
        for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
            print "%d %d" % (i,j)
            r = positions[j] - positions[i]
            d = ln.norm(r)
            r = r / d            
            temp = np.asmatrix(self.call_basis(d, self.mesh, *self.call_basis_args).reshape((len(self.w), 1)))
            temp_force = self.w * temp * r
            accu_force -= temp_force.reshape( (3,1) )
            self.temp_grad +=  temp *  np.asmatrix(r)
        print "Delta F = %s" % (accu_force)
        return np.asarray(self.temp_grad * np.asmatrix(-2. * accu_force)).reshape( (len(self.w)) )
    
