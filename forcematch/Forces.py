from .ForceMatch import Pairwise 
from .Mesh import UniformMesh 

import numpy as np
import random
import numpy.linalg as ln
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from MDAnalysis import Universe
from math import ceil,log
from scipy import weave
from scipy.weave import converters

class Force(object):
    """Can calculate forces from a universe object.

       To be used in the stochastic gradient step, a force should implement all of the methods here
    """
    
    def _setup_update_params(self, w_dim, initial_height=-5):
        self.w = initial_height * np.ones( w_dim )
        self.temp_grad = np.empty( (w_dim, 3) )
        self.temp_force = np.empty( 3 )
        self.w_grad = np.empty( w_dim )
        self.regularization = []
        self.lip = np.ones( np.shape(self.w) )
        self.grad = np.zeros( np.shape(self.w) )
        self.eta = max(1, abs(initial_height) * 2)
        self.sel1 = None
        self.sel2 = None

    #In case the force needs access to the universe for setting up, override (and call this method).
    def setup_hook(self, u):
        self._build_mask(self.sel1, self.sel2, u)

    def set_potential(self, u):
        """ Set the basis function for the potential calculation
        """
        self.call_potential = u        

    def get_category(self):
        try:
            return self.category
        except AttributeError:
            pass

        return None

    def specialize_types(self, selection_pair_1 = None, selection_pair_2 = None):
        self.sel1 = selection_pair_1
        self.sel2 = selection_pair_2


    def _build_mask(self, sel1, sel2, u):
        if(sel1 is None):
            self.mask1 = [True for x in range(u.atoms.numberOfAtoms())]
        else:
            self.mask1 = [False for x in range(u.atoms.numberOfAtoms())]
            for a in u.selectAtoms(sel1):
                self.mask1[a.number] = True
            
        if(sel2 is None):
            self.mask2 = self.mask1
        else:
            self.mask2 = [False for x in range(u.atoms.numberOfAtoms())]
            for a in u.selectAtoms(sel2):
                self.mask2[a.number] = True
                          

    def add_regularizer(self, *regularizers):
        """Add regularization to the stochastic gradient descent
           algorithm.
        """
        for r in regularizers:
            self.regularization.append((r.grad_fxn, r.reg_fxn))

    def calc_forces(self, forces, u):
        raise NotImplementedError("Must implement this function")

    def calc_particle_force(self, i, u):
        raise NotImplementedError("Must implement this function")

    def clone_force(self):
        """Instantiates a new Force, with a reference to the same mesh. 
        """
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
    
    This force correctly handles types.
    """

    def __init__(self, f, g, n, cutoff):
        self.call_force = f
        self.call_grad = g
        self.w = np.zeros( n )
        self._setup_update_params(n)        
        self.category = Pairwise.get_instance(cutoff)


    def clone_force(self):
        copy = PairwiseAnalyticForce(self.call_force, self.call_grad, len(self.w), self.category.cutoff)
        if(not self.call_potential is None):
            copy.call_potential = self.call_potential

        return copy


    def calc_potential(self, u):
        if(self.call_potential is None):
            return 0

        positions = u.atoms.get_positions()
        nlist_accum = 0
        potential = 0
        for i in range(u.atoms.numberOfAtoms()):
            #check atom types
            if(self.mask1[i]):
                maskj = self.mask2
            elif(self.mask2[i]):
                maskj = self.mask1
            else:
                continue
            for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
                #do not double count
                if(not maskj[j] or i < j):
                    continue
                r = positions[j] - positions[i]
                d = ln.norm(r)
                potential += self.call_potential(d,self.w)
            nlist_accum += self.category.nlist_lengths[i]
        return potential
                                     

    def calc_forces(self, forces, u):
        
        positions = u.atoms.get_positions()
        nlist_accum = 0
        for i in range(u.atoms.numberOfAtoms()):
            #check atom types
            if(self.mask1[i]):
                maskj = self.mask2
            elif(self.mask2[i]):
                maskj = self.mask1
            else:
                continue
            for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
                if(not maskj[j]):
                    continue
                r = positions[j] - positions[i]
                d = ln.norm(r)
                forces[i] += self.call_force(d,self.w) * (r / d)
            nlist_accum += self.category.nlist_lengths[i]


    def calc_particle_force(self, i, u):

        self.temp_force.fill(0)
        self.temp_grad.fill(0)

        if(self.mask1[i]):
            maskj = self.mask2
        elif(self.mask2[i]):
            maskj = self.mask1
        else:
            return self.temp_force

        positions = u.atoms.get_positions()
        nlist_accum = np.sum(self.category.nlist_lengths[:i]) if i > 0  else 0
        #for weaving
        w_length = len(self.w)
        temp_grad = self.temp_grad
        for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
            if(not self.mask2[j]):
                continue
            r = positions[j] - positions[i]
            d = ln.norm(r)
            r = r / d            
            self.temp_force += self.call_force(d, self.w) * r
            f_grad = self.call_grad(d, self.w)            
            self.temp_grad +=  np.outer(f_grad, r)
        return self.temp_force
    

class LJForce(PairwiseAnalyticForce):
    """ Lennard jones pairwise analytic force. Not shifted (!)
    """
    def __init__(self, cutoff, sigma=1, epsilon=1):
        super(LJForce, self).__init__(LJForce.lj, LJForce.dlj, 2, cutoff)
        self.set_potential(LJForce.ulj)
        self.w[0] = epsilon
        self.w[1] = sigma
        
    @staticmethod
    def lj(d, w):
        return 4 * w[0] * (6 * (w[1] / d) ** 7 - 12 * (w[1] / d) ** 13)

    @staticmethod
    def ulj(d, w):
        return 4 * w[0] * ((w[1] / d) ** 12 - (w[1] / d) ** 6)

    @staticmethod
    def dlj(d, w):
        return np.asarray([4 * (6 * (w[1] / d) ** 7 - 12 * (w[1] / d) ** 13), 4 * w[0] * (42 * (w[1] / d) ** 6 - 156 * (w[1] / d) ** 12)])

    def plot(self, outfile, alt=None):
        mesh = UniformMesh(0, max(5,self.w[0] * 3), 0.01)
        force = np.empty( len(mesh) )
        if(not alt is None):
            alt_force = np.empty( len(mesh) )
        x = np.empty( np.shape(force) )
        for i in range(len(force)):
            x[i] = (mesh[i] + mesh[i + 1]) / 2.
            force[i] = LJForce.lj(x[i], self.w)
            if(not alt is None):
                alt_force[i] = LJForce.lj(x[i], alt)
        fig = plt.figure(figsize=(8, 6), dpi=80)
        ax = plt.subplot(1,1,1)
        ax.plot(x, force, color="blue")
        if(not alt is None): 
            ax.plot(x, alt_force, color="red")
        ax.axis([min(x), max(x), min(-1, self.w[1] * -2), max(1, self.w[1] * 5)])
        fig.tight_layout()
        plt.savefig(outfile)
                                  


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
    defined as so: def unit_step(x, mesh, height).
    """
    
    def __init__(self, mesh, f, *args):
        self.call_basis = f
        self.call_basis_args = args
        self.mesh = mesh
        self.call_potential = None
        #create weights 
        self.temp_force = np.zeros( 3 )
        self.category = Pairwise.get_instance(mesh.max())

        #if this is an updatable force, set up stuff for it
        self._setup_update_params(len(mesh))
        
    def clone_force(self):
        copy = PairwiseSpectralForce(self.mesh, self.call_basis, *self.call_basis_args)
        if(not self.call_potential is None):
            copy.call_potential = self.call_potential
        return copy           

    def calc_potential(self, u):
        if(self.call_potential is None):
            return 0

        positions = u.atoms.get_positions()
        nlist_accum = 0        
        potential = 0
        self.temp_grad.fill(0)
        for i in range(u.atoms.numberOfAtoms()):
            #check to if this is a valid type
            if(self.mask1[i]):
                maskj = self.mask2
            elif(self.mask2[i]):
                maskj = self.mask1
            else:
                continue
            for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
                #do not doulbe count for calculating potential!
                if(not maskj[i] or i < j):
                    continue
                r = positions[j] - positions[i]
                d = ln.norm(r)
                temp = self.call_potential(d, self.mesh, *self.call_basis_args)
                potential += self.w.dot(temp)
                self.temp_grad[:,1] += temp
            nlist_accum += self.category.nlist_lengths[i]
        return potential

    
    def calc_forces(self, forces, u):        
        positions = u.atoms.get_positions()
        nlist_accum = 0        
        for i in range(u.atoms.numberOfAtoms()):
            #check to if this is a valid type
            if(self.mask1[i]):
                maskj = self.mask2
            elif(self.mask2[i]):
                maskj = self.mask1
            else:
                continue
            for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
                if(not maskj[i]):
                    continue
                r = positions[j] - positions[i]
                d = ln.norm(r)
                force = self.w.dot(self.call_basis(d, self.mesh, *self.call_basis_args)) * (r / d)
                forces[i] += force
            nlist_accum += self.category.nlist_lengths[i]

    def calc_particle_force(self, i, u):

        self.temp_force.fill(0)
        self.temp_grad.fill(0)

        #check type
        if(self.mask1[i]):
            maskj = self.mask2
        elif(self.mask2[i]):
            maskj = self.mask1
        else:
            return self.temp_force


        positions = u.atoms.get_positions()
        nlist_accum = np.sum(self.category.nlist_lengths[:i]) if i > 0  else 0

        #needed for weaving code
        w_length = len(self.w)
        w = self.w
        temp_grad = self.temp_grad
        force = self.temp_force
        for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
            
            if(not maskj[j]):
                continue

            r = positions[j] - positions[i]
            d = ln.norm(r)
            r = r / d
            temp = self.call_basis(d, self.mesh, *self.call_basis_args)
            code = """
                   #line 255 "Forces.py"
     
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
            #forces +=  temp * r
            #self.temp_grad +=  np.outer(temp, r)
        return force

    def update_plot(self, true_force=None, true_potential=None):

        #if this is first call, we need to call plot
        try:
            self.current_fig
        except AttributeError:
            self.plot(None, true_force, true_potential)
            return
        
        #use center of our fine mesh and call the force calculations
        for i in range(len(self.plot_force)):
            self.plot_force[i] = self.w * np.asmatrix(self.call_basis(self.plot_x[i], self.mesh, *self.call_basis_args).reshape((len(self.w), 1)))

        #restore cache, update plot, redraw only force            
        force_ax = self.current_fig.get_axes()[0]
        self.force_line.set_ydata(self.plot_force)
        force_ax.set_ylim(1.1*min(min(self.plot_force), -max(self.plot_force)), 1.1*max(self.plot_force))
            
        #plot potential if necessary
        if(not (self.call_potential is None)):
            for i in range(len(self.plot_force)):
                self.plot_potential[i] = self.w * np.asmatrix(self.call_potential(self.plot_x[i], self.mesh, *self.call_basis_args).reshape((len(self.w), 1)))

            #restore cache, update plot, redraw only potential
            potential_ax = self.current_fig.get_axes()[1]
            self.potential_line.set_ydata(self.plot_potential)
            potential_ax.set_ylim(1.1*min(min(self.plot_potential), -max(self.plot_potential)), 1.1*max(self.plot_potential))

        plt.draw()

    def plot(self, outfile=None, true_force=None, true_potential=None):

        #make a mesh finer than the mesh used for finding paramers
        self.plot_mesh = UniformMesh(self.mesh.min(), self.mesh.max(), self.mesh.dx / 2)
        self.plot_force = np.empty( len(self.plot_mesh) )
        self.plot_x = np.empty( np.shape(self.plot_force) )        

        
        #set up figure and check to see if we're plotting the potential
        self.current_fig = plt.figure(figsize=(16, 12), dpi=80)

        if(self.call_potential is None):
            force_ax = plt.subplot(111)

        else:
            force_ax = plt.subplot(211)
            potential_ax = plt.subplot(212)

        #use center of our fine mesh and call the force calculations
        for i in range(len(self.plot_force)):
            self.plot_x[i] = (self.plot_mesh[i] + self.plot_mesh[i + 1]) / 2.
            self.plot_force[i] = self.w * np.asmatrix(self.call_basis(self.plot_x[i], self.mesh, *self.call_basis_args).reshape((len(self.w), 1)))


        #draw true functions, if they are given
        if(not (true_force is None)):
            force = np.empty( len(self.plot_mesh) )
            for i in range(len(force)):
                force[i] = true_force(self.plot_x[i])
            force_ax.plot(self.plot_x, force, color="red")
        if(not (true_potential is None) and not (self.call_potential is None)):
            potential = np.empty( len(self.plot_mesh) )
            for i in range(len(potential)):
                potential[i] = true_potential(self.plot_x[i])
            potential_ax.plot(self.plot_x, potential, color="red")

        #plot force and save reference to line
        self.force_line, = force_ax.plot(self.plot_x, self.plot_force, color="blue")

        force_ax.set_ylim(1.1*min(min(self.plot_force), -max(self.plot_force)), 1.1*max(self.plot_force))

        #plot potential if possible
        if(not (self.call_potential is None)):
            self.plot_potential = np.empty( len(self.plot_mesh) )
            for i in range(len(self.plot_force)):
                self.plot_potential[i] = self.w * np.asmatrix(self.call_potential(self.plot_x[i], self.mesh, *self.call_basis_args).reshape((len(self.w), 1)))

            self.potential_line, = potential_ax.plot(self.plot_x, self.plot_potential, color="green")
            potential_ax.set_ylim(1.1*min(min(self.plot_potential), -max(self.plot_potential)), 1.1*max(self.plot_potential))

        #make layout tighter
        self.current_fig.tight_layout()
        
        #cache background

        #show or output to file
        if(outfile is None):
            plt.ion()
            plt.show()
        else:
            plt.savefig(outfile)
                                  
