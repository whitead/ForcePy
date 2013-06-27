from ForceMatch import *
from Mesh import *

import numpy as np
import random
import numpy.linalg as ln
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from MDAnalysis import Universe
from math import ceil,log
from scipy import weave
from scipy.weave import converters

class Force:
    """Can calculate forces from a universe object.

       To be used in the stochastic gradient step, a force should implement all of the methods here
    """
    
    def _register_hook(self, category):
        """Register the force with the category and keep a reference to it
        """
        self.category = category
        self.sample = np.arange(0)

    def _setup_update_params(w_dim):
        self.w = np.zeros( len(mesh) )
        self.temp_grad = np.zeros( (len(mesh), 3) )
        self.w_grad = np.empty( len(mesh) )
        self.regularization = []
        self.lip = np.ones( np.shape(self.w) )
        self.grad = np.zeros( np.shape(self.w) )
        self.eta = 1

    def add_regularizer(self, regularizer):
        """Add regularization to the stochastic gradient descent
           algorithm.
        """
        self.regularization.append((regularizer.grad_fxn, regularizer.reg_fxn))

    def calc_forces(self, forces, u):
        raise NotImplementedError("Must implement this function")

    def calc_particle_force(self, forces, u):
        raise NotImplementedError("Must implement this function")

    


class FileForce(Force):
    """ Reads forces from the trajectory file
    """

    def calc_forces(self, forces, u):
        forces[:] = u.trajectory.ts._forces

    
    
class PairwiseAnalyticForce(Force):
    """ A pairwise analtric force that takes in a function for
    calculating the force. The function passed should accept the
    scalar distance between the two particles as its first argument
    and a scalar vector of length n for its second argument. The
    gradient should take in the pairwise distance and a vector of
    length n (as set in the constructor).  It should return a gradient
    of length n.
    """
    def __init__(self, f, g, n, passes=1):
        self.call_force = f
        self.call_grad = g
        self.w = np.zero( n )
        self._setup_update_params(n)
        

    def calc_forces(self, forces, u):
        positions = u.atoms.get_positions()
        nlist_accum = 0
        for i in range(u.atoms.numberOfAtoms()):
            for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
                r = positions[j] - positions[i]
                d = ln.norm(r)
                force = self.call_force(f,self.w) * (r / d)
                forces[i] += force
            nlist_accum += self.category.nlist_lengths[i]

    def calc_particle_force(self, forces, u):
        positions = u.atoms.get_positions()
        nlist_accum = np.sum(self.category.nlist_lengths[:i]) if i > 0  else 0
        self.temp_force.fill(0)
        self.temp_grad.fill(0)
        #for weaving
        w_length = len(self.w)
        w = self.w
        temp_grad = self.temp_grad
        force = self.temp_force
        for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
            r = positions[j] - positions[i]
            d = ln.norm(r)
            r = r / d            
            temp = self.call_force(d, self.w)
            f_grad = self.call_grad(d, g)            
            code = """
                   #line 185 "Forces.py"
     
                   for(int i = 0; i < w_length; i++) {
                       for(int j = 0; j < 3; j++) {
                           force(j) += temp(i) * r(j);
                           temp_grad(i,j) += f_grad(i) * r(j);
                       }
                    }
                    """
            weave.inline(code, ['w', 'w_length', 'temp', 'f_grad', 'r', 'force', 'temp_grad'],
                         type_converters=converters.blitz,
                         compiler = 'gcc')
            #force += (self.w.dot(temp)  * r).reshape( (3,1) )
            #self.temp_grad +=  np.asmatrix(temp.reshape((len(self.w), 1))) *  np.asmatrix(r)
        return force
    

        

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
        raise Exception("Smoothregulizer is static and should not be instanced")

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
        raise Exception("L2Regularizer is static and should not be instanced")

    @staticmethod
    def grad_fxn(x):
        g = 2 * np.copy(x)
        return g

    @staticmethod
    def reg_fxn(x):
        return ln.norm(x)    


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
        self.temp_force = np.zeros( 3 )

        #if this is an updatable force, set up stuff for it
        self._setup_update_params(len(mesh))

    
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
        self.temp_force.fill(0)
        self.temp_grad.fill(0)
        #needed for weaving code
        w_length = len(self.w)
        w = self.w
        temp_grad = self.temp_grad
        force = self.temp_force
        for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
            r = positions[j] - positions[i]
            d = ln.norm(r)
            r = r / d            
            temp = self.call_basis(d, self.mesh, *self.call_basis_args)
            code = """
                   #line 185 "Forces.py"
     
                   for(int i = 0; i < w_length; i++) {
                       for(int j = 0; j < 3; j++) {
                           force(j) += w(i) * temp(i) * r(j);
                           temp_grad(i,j) += temp(i) * r(j);
                       }
                    }
                    """
            weave.inline(code, ['w', 'w_length', 'temp', 'r', 'force', 'temp_grad'],
                         type_converters=converters.blitz,
                         compiler = 'gcc')
            #force += (self.w.dot(temp)  * r).reshape( (3,1) )
            #self.temp_grad +=  np.asmatrix(temp.reshape((len(self.w), 1))) *  np.asmatrix(r)
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
        ax.axis([min(x), max(x), -max(force), max(force)])
        fig.tight_layout()
        plt.savefig(outfile)
                                  
