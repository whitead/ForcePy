from .ForceMatch import Pairwise, min_img_vec
from .Mesh import UniformMesh 
from ForcePy.util import norm3

import numpy as np
import random
import numpy.linalg as ln
from MDAnalysis import Universe
from math import ceil,log
from scipy import weave
from scipy.weave import converters

class Force(object):
    """Can calculate forces from a universe object.

       To be used in the stochastic gradient step, a force should implement all of the methods here
    """

    def _setup_update_params(self, w_dim, initial_w=-5):
        try:
            if(w_dim != len(initial_w)):
                self.w = initial_w[0] * np.ones( w_dim )
            else:
                self.w = np.copy(initial_w)
            self.eta = max(1, np.median(abs(initial_height)) * 2)
        except TypeError:
            self.w = initial_w * np.ones( w_dim )
            self.eta = max(1, abs(initial_w) * 2)

        self.temp_grad = np.empty( (w_dim, 3) )
        self.temp_force = np.empty( 3 )
        self.w_grad = np.empty( w_dim )
        self.regularization = []
        self.lip = np.ones( np.shape(self.w) )
        self.sel1 = None
        self.sel2 = None

    #In case the force needs access to the universe for setting up, override (and call this method).
    def setup_hook(self, u):
        try:
            self._build_mask(self.sel1, self.sel2, u)
        except AttributeError:
            pass #some forces don't have selections, ie FileForce

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
        self.type_name = "[%s] -- [%s]" % (selection_pair_1, selection_pair_2)


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

    def plot(self, force_ax, potential_ax = None, true_force = None, true_potential = None):
        #make a mesh finer than the mesh used for finding paramers
        self.plot_x = np.arange( self.mind(), self.maxd(), (self.maxd() - self.mind()) / 1000. )
        self.plot_force = np.empty( len(self.plot_x) )


        self.true_force = true_force
        self.true_potential = true_potential
        
        self.force_ax = force_ax
        
        #set the plot title
        try:
            title = "Force" #in case no plot_title
            title = self.plot_name #if there is a plot_title
            title = "%s type %s" % (title, self.type_name) #if this is specialized
        except AttributeError:
            pass
        
        force_ax.set_title(title)
        
        if(potential_ax is None):
            self.potential_ax = self.force_ax
        else:
            self.potential_ax.set_title("Potential of %s" % title)

        #call the force calculations
        self.calc_force_array(self.plot_x, self.plot_force)

        #draw true functions, if they are given
        if(not (true_force is None)):
            true_force_a = np.empty( len(self.plot_x) )
            for i in range(len(true_force_a)):
                true_force_a[i] = true_force(self.plot_x[i])
            force_ax.plot(self.plot_x, true_force_a, color="green")
        if(not (true_potential is None) and not (self.call_potential is None)):
            true_potential_a = np.empty( len(self.plot_x) )
            for i in range(len(true_potential_a)):
                true_potential_a[i] = true_potential(self.plot_x[i])
            potential_ax.plot(self.plot_x, true_potential_a, color="green")

        #plot force and save reference to line
        self.force_line, = force_ax.plot(self.plot_x, self.plot_force, color="blue")

        force_ax.set_ylim(1.1*min(min(self.plot_force), -max(self.plot_force)), 1.1*max(self.plot_force))

        #plot potential if possible
        try:
            self.plot_potential = np.empty( len(self.plot_x) )
            self.calc_potential_array(self.plot_x, self.plot_potential)
            self.potential_line, = self.potential_ax.plot(self.plot_x, self.plot_potential, color="red")
            if(self.force_ax == self.potential_ax):
                self.potential_ax.set_ylim(min(1.1*min(min(self.plot_potential), -max(self.plot_potential)), 1.1*min(min(self.plot_force), -max(self.plot_force))), max(1.1*max(self.plot_force), 1.1*max(self.plot_potential)))
            else:
                self.potential_ax.set_ylim(1.1*min(min(self.plot_potential), -max(self.plot_potential)), 1.1*max(self.plot_potential))
        except NotImplementedError:
            self.plot_potential = None


    def update_plot(self):
        #call the force calculations
        self.calc_force_array(self.plot_x, self.plot_force)
        self.force_line.set_ydata(self.plot_force)
        self.force_ax.set_ylim(1.1*min(min(self.plot_force), -max(self.plot_force)), 1.1*max(self.plot_force))

        #plot potential if possible
        if(not (self.plot_potential is None)):
            self.calc_potential_array(self.plot_x, self.plot_potential)
            self.potential_line.set_ydata(self.plot_potential)
            if(self.force_ax == self.potential_ax):
                self.potential_ax.set_ylim(min(1.1*min(min(self.plot_potential), -max(self.plot_potential)), 1.1*min(min(self.plot_force), -max(self.plot_force))), max(1.1*max(self.plot_force), 1.1*max(self.plot_potential)))
            else:
                self.potential_ax.set_ylim(1.1*min(min(self.plot_potential), -max(self.plot_potential)), 1.1*max(self.plot_potential))

        
                                  
    def calc_force_array(self, d, force):
        raise NotImplementedError("Must implement this function")

    def calc_potential_array(self, d, potentials):
        raise NotImplementedError("Must implement this function")

    def calc_potentials(self, u):
        raise NotImplementedError("Must implement this function")

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

    def clone_force(self):
        copy = FileForce()



class AnalyticForce(Force):
    """ A pairwise analtric force that takes in a function for
    calculating the force. The function passed should accept the
    scalar distance between the two particles as its first argument
    and a scalar vector of length n for its second argument. The
    gradient should take in the pairwise distance and a vector of
    length n (as set in the constructor).  It should return a gradient
    of length n.
    
    This force correctly handles types.
    """


    def __init__(self, category, f, g, n, cutoff=None, potential = None):
        self.call_force = f
        self.call_grad = g
        self.call_potential = potential
        self.w = np.zeros( n )
        self._setup_update_params(n)        
        self.category = category.get_instance(cutoff)
        self.plot_name = "AnalyticForce for %s" % category.__name__


    def clone_force(self):
        copy = AnalyticForce(self.call_force, self.call_grad, len(self.w), self.category.cutoff)
        if(not self.call_potential is None):
            copy.call_potential = self.call_potential

        return copy

    def mind(self):
        return 0

    def maxd(self):
        try:
            return category.cutoff
        except AttributeError:
            return 10


    def calc_force_array(self, d, forces):
        for i in range(len(d)):
            forces[i] = self.call_potential(d[i], self.w)

    def calc_potential_array(self, d, potentials):
        if(self.call_potential is None):
            return
        for i in range(len(d)):
            potentials[i] = self.call_potential(d[i], self.w)

    def calc_potentials(self, u):
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
                r = min_img_vec(positions[j], positions[i], u.trajectory.ts.dimensions, u.trajectory.periodic)
                d = norm3(r)            
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
                r = min_img_vec(positions[j], positions[i], u.trajectory.ts.dimensions, u.trajectory.periodic)
                d = norm3(r)
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

        for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
            if(not self.mask2[j]):
                continue
            r = min_img_vec(positions[j], positions[i], u.trajectory.ts.dimensions, u.trajectory.periodic)
            d = norm3(r)
            r = r / d            
            self.temp_force += self.call_force(d, self.w) * r
            f_grad = self.call_grad(d, self.w)            
            self.temp_grad +=  np.outer(f_grad, r)
        return self.temp_force
    

class LJForce(AnalyticForce):
    """ Lennard jones pairwise analytic force. Not shifted (!)
    """
    def __init__(self, cutoff, sigma=1, epsilon=1):
        super(LJForce, self).__init__(Pairwise, LJForce.lj, LJForce.dlj, 2, cutoff)
        self.set_potential(LJForce.ulj)
        self.w[0] = epsilon
        self.w[1] = sigma
        self.plot_name = "LJForce"
        
    @staticmethod
    def lj(d, w):
        return 4 * w[0] * (6 * (w[1] / d) ** 7 - 12 * (w[1] / d) ** 13)

    @staticmethod
    def ulj(d, w):
        return 4 * w[0] * ((w[1] / d) ** 12 - (w[1] / d) ** 6)

    @staticmethod
    def dlj(d, w):
        return np.asarray([4 * (6 * (w[1] / d) ** 7 - 12 * (w[1] / d) ** 13), 4 * w[0] * (42 * (w[1] / d) ** 6 - 156 * (w[1] / d) ** 12)])
                                  
    def mind(self):
        return w[1] * 0.5
    
    def maxd(self):
        return w[1] * 5


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


class SpectralForce(Force):
    """A pairwise force that is a linear combination of basis functions
    The basis function should take two arguments: the distance and the
    mesh. Additional arguments after the function pointer will be
    passed after the two arguments. For example, the function may be
    defined as so: def unit_step(x, mesh, height).
    """
    
    def __init__(self, category, mesh, basis):
        self.basis = basis
        self.mesh = mesh
        #create weights 
        self.temp_force = np.zeros( 3 )
        self.category = category.get_instance(mesh.max())
        self.plot_name = "SpectralForce for %s" % category.__name__

        #if this is an updatable force, set up stuff for it
        self._setup_update_params(len(mesh))

    def mind(self):
        return self.mesh.min()

    def maxd(self):
        return self.mesh.max()
        
    def clone_force(self):
        copy = SpectralForce(self.category.__class__, self.mesh, self.basis)
        return copy           

    def calc_force_array(self, d, forces):
        for i in range(len(d)):
            forces[i] = self.w.dot(self.basis.force(d[i], self.mesh))        

    def calc_potential_array(self, d, potentials):
        for i in range(len(d)):
            potentials[i] = self.w.dot(self.basis.potential(d[i], self.mesh))        


    def calc_potentials(self, u):

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
                r = min_img_vec(positions[j], positions[i], u.trajectory.ts.dimensions, u.trajectory.periodic)
                d = norm3(r)
                temp = self.basis.potential(d, self.mesh)
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
                r = min_img_vec(positions[j], positions[i], u.trajectory.ts.dimensions, u.trajectory.periodic)
                d = norm3(r)
                force = self.w.dot(self.basis.force(d, self.mesh)) * (r / d)
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
        temp = np.empty( len(self.w) , dtype=np.float32)
        for j in self.category.nlist[nlist_accum:(nlist_accum + self.category.nlist_lengths[i])]:
            
            if(not maskj[j]):
                continue

            r = min_img_vec(positions[j], positions[i], u.trajectory.ts.dimensions, u.trajectory.periodic)
            d = norm3(r)
            r = r / d
            self.basis.force_cache(d, temp, self.mesh)
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
            #force +=  self.w.dot(temp) * r
            #self.temp_grad +=  np.outer(temp, r)
        return force

