import numpy as np
import random
import numpy.linalg as ln
import json
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from MDAnalysis import *
from math import *


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



class Force:
    """Can calculate forces from a universe object
    """
    
    def _register_hook(self, category):
        """Register the force with the category and keep a reference to it
        """
        self.category = category
        self.sample = np.arange(0)

            
    

#concrete class

class Pairwise(ForceCategory):
    """Pairwise force category. It handles constructing a neighbor-list at each time-step. 
    """
    
    def __init__(self, cutoff=10):
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

class FileForce(Force):
    """ Reads forces from the trajectory file
    """

    def calc_forces(self, forces, u):
        forces[:] = u.trajectory.ts._forces

    
    
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
                forces[i] += force
            nlist_accum += self.category.nlist_lengths[i]


class Regularizer:
    """grad_fxn: takes in vector returns gradient vector
       reg_fxn: takes in vector, returns scalar
     """
    def __init__(self, grad_fxn, reg_fxn):
        self.grad_fxn = grad_fxn
        self.reg_fxn = reg_fxn

class SmoothRegularizer(Regularizer):
    """ sum_i (w_{i+1} - w{i}) ^ 2
    """

    def __init__(self):
        raise Exception("Smoothregulizer is static and should be instanced")

    @staticmethod
    def grad_fxn(x):
        g = np.empty( np.shape(x) )
        g[0] = 2 * x[0]
        g[-1] = 2 * x[-1]
        g[1:] = 4 * x[1:] - 2 * x[:-1]
        g[:-1] -= 2 * x[1:]
        return g

    @staticmethod
    def reg_fxn(x):
        return 0


class L2Regularizer(Regularizer):
    """ sum_i (w_{i+1} - w{i}) ^ 2
    """

    def __init__(self):
        raise Exception("L2Regularizer is static and should be instanced")

    @staticmethod
    def grad_fxn(x):
        g = 2 * np.copy(x)
        return g

    @staticmethod
    def reg_fxn(x):
        return ln.norm(x)

class UniformMesh:

    def __init__(self, l, r, dx):
        self.l = l
        self.r = r
        self.dx = dx
        self.length = int(ceil((self.r - self.l) / self.dx))

    def max(self):
        return self.r

    def min(self):
        return self.l

    def mesh_index(self, x):
        return max(0, min(self.length - 1, int(floor( (x - self.l) / self.dx) )))
        
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return i * self.dx + self.l
        

class PairwiseSpectralForce(Force):
    """A pairwise force that is a linear combination of basis functions

    The basis function should take two arguments: the distance and the
    mesh. Additional arguments after the function pointer will be
    passed after the two arguments. For example, the function may be
    defined as so: def unit_step(x, mesh, height)
    """
    
    def __init__(self, mesh, updates_per_frame, f, *args):
        self.call_basis = f
        self.call_basis_args = args
        self.mesh = mesh
        #create weights 
        self.w = np.zeros( len(mesh) )
        self.temp_grad = np.asmatrix(np.zeros( (len(mesh), 3) ))
        self.regularization = []
        #if this is an updatable force, set up stuff for it
        try:
            self.lip = np.ones( np.shape(self.w) )
            self.grad = np.zeros( np.shape(self.w) )
            self.eta = 1
        except AttributeError:
            pass#not updatable. Oh well
        self.index = 0
        self.passes = updates_per_frame



    def add_regularizer(self, regularizer):
        """Add regularization to the stochastic gradient descent
           algorithm.
        """
        self.regularization.append((regularizer.grad_fxn, regularizer.reg_fxn))


    def update(self, ref_forces, u):

        for p in range(self.passes):
            net_df = 0
            self.index += 1

            if(self.index % 10 == 0):
                self.plot("set_%d.png" % (self.index))
                print "set_%d.png" % (self.index)

            #sample particles and run updates on them 
            for i in random.sample(range(u.atoms.numberOfAtoms()),u.atoms.numberOfAtoms()):

                df = self.calc_particle_force(i,u) - ref_forces[i].reshape( (3,1) )#get delta force in particle i
                net_df += ln.norm(df)            
                grad = np.asarray(self.temp_grad * df).reshape( (len(self.w)) )
                for r in self.regularization:
                    grad += r[0](self.w)
                self.lip +=  np.square(grad)
                self.w = self.w - self.eta * sqrt(self.passes) / np.sqrt(self.lip) * grad
            print "log error = %g" % (0 if net_df < 1 else log(net_df))

    
    def calc_forces(self, forces, u):        
        positions = u.atoms.get_positions()
        nlist_accum = 0        
        for i in range(u.atoms.numberOfAtoms()):
            for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
                r = positions[j] - positions[i]
                d = ln.norm(r)
                force = self.w.dot(self.call_basis(d, self.mesh, *self.call_basis_args)) * (r / d)
                forces[i] += force
            nlist_accum += self.category.nlist_lengths[i]

    def calc_particle_force(self, i, u):
        positions = u.atoms.get_positions()
        nlist_accum = np.sum(self.category.nlist_lengths[:i]) if i > 0  else 0
        force = np.zeros( (3,1) ) 
        self.temp_grad.fill(0)
        for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
            r = positions[j] - positions[i]
            d = ln.norm(r)
            r = r / d            
            temp = self.call_basis(d, self.mesh, *self.call_basis_args)
            force += (self.w.dot(temp)  * r).reshape( (3,1) )
            self.temp_grad +=  np.asmatrix(temp.reshape((len(self.w), 1))) *  np.asmatrix(r)
        return force
    


    def plot(self, outfile):
        mesh = UniformMesh(self.mesh.min(), self.mesh.max(), self.mesh.dx / 2)
        force = np.empty( len(mesh) )
        true_force = np.empty( len(mesh))
        x = np.empty( np.shape(force) )
        for i in range(len(force)):
            x[i] = (mesh[i] + mesh[i + 1]) / 2.
            force[i] = self.w * np.asmatrix(self.call_basis(x[i], self.mesh, *self.call_basis_args).reshape((len(self.w), 1)))
        fig = plt.figure(figsize=(8, 6), dpi=80)
        ax = plt.subplot(1,1,1)
        ax.plot(x, force, color="blue")
#        ax.plot(x, true_force, color="red")
        ax.axis([min(x), max(x), -25, 25])
        fig.tight_layout()
        plt.savefig(outfile)
                                  
