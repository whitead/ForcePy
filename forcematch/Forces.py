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
    """Can calculate forces from a universe object
    """
    
    def _register_hook(self, category):
        """Register the force with the category and keep a reference to it
        """
        self.category = category
        self.sample = np.arange(0)

            


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
        self.temp_grad = np.zeros( (len(mesh), 3) )
        self.temp_force = np.zeros( 3 )
        self.w_grad = np.empty( len(mesh) )
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

            #stuff for weave
            grad = self.w_grad
            w_length = len(self.w)
            temp_grad = self.temp_grad

            #sample particles and run updates on them 
            for i in random.sample(range(u.atoms.numberOfAtoms()),u.atoms.numberOfAtoms()):

                df = self.calc_particle_force(i,u) - ref_forces[i]#get delta force in particle i
                net_df += ln.norm(df)
                code = """
                       for(int i = 0; i < w_length; i++) {
                           grad(i) = 0;
                           for(int j = 0; j < 3; j++)
                               grad(i) += temp_grad(i,j) * df(j);
                       }
                """
                weave.inline(code, ['w_length', 'grad', 'df', 'temp_grad'],
                         type_converters=converters.blitz,
                         compiler = 'gcc')
                #grad = np.apply_along_axis(np.sum, 1, self.temp_grad * df)
                for r in self.regularization:
                    grad += r[0](self.w)
                self.lip +=  np.square(grad)
                self.w = self.w - self.eta * np.sqrt(self.passes) / np.sqrt(self.lip) * grad
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
                                  
