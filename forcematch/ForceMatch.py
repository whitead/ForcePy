import random
import numpy as np
import numpy.linalg as ln
import json
from math import ceil
from MDAnalysis import Universe
from scipy import weave
from scipy.weave import converters
from math import ceil, log, exp, sqrt

class ForceMatch:
    """Main force match class.
    """
    
    def __init__(self, cguniverse, input_file):
        self.ref_cats = []
        self.tar_cats = []
        self.ref_forces =  []
        self.tar_forces = []
        self.u = aauniverse
        self._load_json(input_file) 
        self.force_match_calls = 0
        self.plot_frequency = 10
    
    def _load_json(self, input_file):
        with open(input_file, 'r') as f:
            self.json = json.load(f)
        self._test_json(self.json)
        self.kt = self.json["kT"]
        if("observable" in self.json):
            self.do_obs = True
            self.obs = [0 for x in range(self.u.trajectory.numframes)]
            with open(self.json["observable"], 'r') as f:
                lines = f.readlines()
                if(len(lines) < len(self.obs)):
                    raise IOError("Number of the frames (%d) does not match number of lines in observation file (%d)" %
                                  (len(self.obs), len(lines)))
                for i, line in zip(range(len(self.obs)), lines[:len(self.obs)]):
                    self.obs[i] = float(line.split()[0])


        if("box" in self.json):
            if(len(self.json["box"]) != 3):
                raise IOError("Input file JSON: box must look like \"box\":[5,5,5]. It must have 3 dimensions in an array")

                
                
    def _test_json(self, json, required_keys = [("structure", "Toplogy file"), ("trajectory", "Trajectory File"), ("kT", "Boltzmann's constant times temperature")]):
        for rk in required_keys:
            if(not json.has_key(rk[0])):
                raise IOError("Error in input file, could not find %s\n. Set using %s keyword" % (rk[1], rk[0]))

    def add_tar_force(self, *forces):
        for f in forces:
            self.tar_forces.append(f)
            cat = f.get_category()
            if(not (cat is None)):
                self.tar_cats.append(cat)
            f.setup_hook(self.u)

    def add_ref_force(self, *forces):
        for f in forces:
            self.ref_forces.append(f)
            cat = f.get_category()
            if(not (cat is None)):
                self.ref_cats.append(cat)
            f.setup_hook(self.u)


    def swap_match_parameters_cache(self):
        try:
            for f in self.tar_forces:
                self.cache[f], f.lip = f.lip, self.cache[f]
        except AttributeError:
            self.cache = {}
            for f in self.tar_forces:
                self.cache[f] = np.copy(f.lip)
                        
        
    def force_match(self, iterations = 0):
        
        if(iterations == 0):
            iterations = self.u.trajectory.numframes
        
        ref_forces = np.zeros( (self.u.atoms.numberOfAtoms(), 3) )
        self.u.trajectory.rewind() # just in case this is called after some analysis has been done

        for ts in self.u.trajectory:
            
            #set box if necessary
            if("box" in self.json):
                #strange ordering due to charm
                self.u.trajectory.ts._unitcell[0] = self.json["box"][0]
                self.u.trajectory.ts._unitcell[2] = self.json["box"][1]
                self.u.trajectory.ts._unitcell[5] = self.json["box"][2]

            self._setup()

            for rf in self.ref_forces:
                rf.calc_forces(ref_forces, self.u)            

            #make plots
            if(iterations % self.plot_frequency == 0):
                for f in self.tar_forces:
                    f.update_plot(true_force=lambda x:4 * (6 * x**(-7) - 12 * x**(-13) ), true_potential=lambda x: 4 * (x**(-12) - x**(-6)))
                    try:
                        pass
                        #f.update_plot(true_force=lambda x:4 * (6 * x**(-7) - 12 * x**(-13) ), true_potential=lambda x: 4 * (x**(-12) - x**(-6)))
                    except AttributeError:
                        #doesn't have plotting method, oh well
                        pass

            #track error
            net_df = 0
            self.force_match_calls += 1

            #sample particles and run updates on them 
            for i in random.sample(range(self.u.atoms.numberOfAtoms()),self.u.atoms.numberOfAtoms()):
                #calculate net forces deviation
                df = ref_forces[i]
                for f in self.tar_forces:
                    df -= f.calc_particle_force(i,self.u)
                net_df += ln.norm(df)


                #inline C code to accumulate gradient
                code = """
                       for(int i = 0; i < w_length; i++) {
                           grad(i) = 0;
                           for(int j = 0; j < 3; j++)
                               grad(i) -= temp_grad(i,j) * df(j); //negative due to df being switched
                       }
                """

                #now run gradient update step on all the force types
                for f in self.tar_forces:
                    #setup temps for inlince C code
                    w_length = len(f.w)
                    grad = f.w_grad
                    temp_grad = f.temp_grad
                    weave.inline(code, ['w_length', 'grad', 'df', 'temp_grad'],
                         type_converters=converters.blitz,
                         compiler = 'gcc')

                    #the code which is being weaved:
                    #grad = np.apply_along_axis(np.sum, 1, self.temp_grad * df)

                    #apply any regularization
                    for r in f.regularization:
                        grad += r[0](f.w)
                    f.lip +=  np.square(grad)

                    f.w = f.w - f.eta / np.sqrt(f.lip) * grad

            ref_forces.fill(0)
            self._teardown()

            #log of the error
            print "log error at %d  = %g" % (iterations, 0 if net_df < 1 else log(net_df))
            
            iterations -= 1
            if(iterations == 0):
                break


    def observation_match(self, obs_sweeps = 25, obs_samples = None, reject_tol = None):
        """ Match observations
        """

        if(obs_samples is None):
            obs_samples = max(5, self.u.trajectory.numframes / obs_sweeps)
        if(reject_tol is None):
            reject_tol = obs_samples
                
        #in case force mathcing is being performed simultaneously,
        #we want to cache any force specific parameters so that we
        #can swap them back in afterwards
        self.swap_match_parameters_cache()

        #we're going to sample the covariance using importance
        #sampling. This requires a normalization coefficient, so
        #we must do multiple random frames

        s_grads = {} #this is to store the sampled gradients. Key is Force and Value is gradient
        for f in self.tar_forces:
            s_grads[f] = [None for x in range(obs_samples)]
        s_obs = [0 for x in range(obs_samples)] #this is to store the sampled observations
        s_weights = [0 for x in range(obs_samples)]
            
        for s in range(obs_sweeps):

            self.force_match_calls += 1
            #make plots
            for f in self.tar_forces:
                try:
                    f.update_plot(true_force=lambda x:4 * (6 * x**(-7) - 12 * x**(-13) ), true_potential=lambda x: 4 * (x**(-12) - x**(-6)))
                except AttributeError:
                     #doesn't have plotting method, oh well
                    pass

            #now we esimtate gradient of the loss function via importance sampling
            normalization = 0
                
            #note, this reading method is so slow. I should implement the frame jump in xyz
            rejects = 0
            i = 0

            while i < obs_samples:

                index = self._sample_ts() #sample a trajectory frame
                self._setup()

                 #get weight
                dev_energy = 0
                for f in self.tar_forces:
                    dev_energy -= f.calc_potential(self.u)
                        
                for f in self.ref_forces:
                    dev_energy += f.calc_potential(self.u)

                    
                if(abs(dev_energy /self.kt) > 250):
                    rejects += 1
                    if(rejects == reject_tol):
                        print "Rejection rate of frames is too high, restarting force matching"
                        self.swap_match_parameters_cache()
                        self.force_match(rejects) #arbitrarily using number of rejects for number matces to use
                        self.swap_match_parameters_cache()
                        rejects = 0
                        continue
                    else:
                        continue

                weight = exp(dev_energy / self.kt)
                s_weights[i] = weight
                normalization += weight

                #store gradient and observabels
                s_obs[i] = weight * self.obs[i]
                for f in self.tar_forces:
                    s_grads[f][i] = weight * np.copy(f.temp_grad[:,1])

                i += 1
                self._teardown()

            #normalize and calculate covariance
            for f in self.tar_forces:
                f.w_grad.fill(0)
                grad = f.w_grad

                #two-pass covariance calculation, utilizing the temp_grad in f
                meanobs = sum(s_obs) / normalization
                meangrad = f.temp_grad[:,2]
                meangrad.fill(0)
                for x in s_grads[f]:
                    meangrad += x  / normalization
                for x,y in zip(s_obs, s_grads[f]):
                    grad += (x - meanobs) * (y - meangrad) / normalization

                #recall we need the negative covariance times the inverse temperature
                grad *= -1. / self.kt

                #now update the weights
                f.lip += np.square(grad)
                change = f.eta / np.sqrt(f.lip) * grad
                f.w = f.w - f.eta / np.sqrt(f.lip) * grad

                print "Obs Mean: %g, reweighted mean: %g" % (sum(self.obs) / len(self.obs) ,meanobs)

            

    def add_and_type_pair(self, force):
        types = []
        for a in self.u.atoms:
            if(not a.type in types):
                types.append(a.type)
        for i in range(len(types)):
            for j in range(i,len(types)):
                f = force.clone_force()
                f.specialize_types("type %s" % types[i], "type %s" % types[j])
                self.add_tar_force(f)

    def _sample_ts(self):        
        self.u.trajectory.rewind()
        index = random.randint(0,self.u.trajectory.numframes - 1)
        [self.u.trajectory.next() for x in range(index)]
        return index
                   

        
            
    def _setup(self):
        for rfcat in self.ref_cats:
            rfcat._setup(self.u)
        for tfcat in self.tar_cats:
            tfcat._setup_update(self.u)        

    def _teardown(self):
        for rfcat in self.ref_cats:
            rfcat._teardown()
        for tfcat in self.tar_cats:
            tfcat._teardown_update()        


#abstract classes

class ForceCategory(object):
    """A category of force/potential type.
    
    The forces used in force matching are broken into categories, where
    the sum of each category of forces is what's matched in the force
    matching code. Examples of categories are pairwise forces,
   threebody forces, topology forces (bonds, angles, etc).
   """
    pass



class NeighborList:
    """Neighbor list class
    """
    def __init__(self, u, cutoff):
        
        #set up cell number and data
        self.cutoff = cutoff
        self.box = [(0,x) for x in u.dimensions[:3]]
        #self.box = [(0,5.2) for x in u.dimensions[:3]]
        self.nlist_lengths = np.arange(0, dtype='int32')
        self.nlist = np.arange(0, dtype='int32')
        self.cell_number = [max(1,int(ceil((x[1] - x[0]) / self.cutoff))) for x in self.box]        
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
                                if(not neighbor in self.cell_neighbors[index]): #this is possible if wrapped and cell number is 1
                                    self.cell_neighbors[index].append(neighbor)

    def dump_cells(self):
        print self.cell_number
        print self.box
        print self.cell_neighbors

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

    def _build_exclusion_list(self, u):
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
                for b in range(temp_list[a]):
                    self.exclusion_list[a].append(b) 

    def build_nlist(self, u):
        
        if(self.exclusion_list == None):
            self._build_exclusion_list(u)

        self.nlist = np.resize(self.nlist, (u.atoms.numberOfAtoms() - 1) * u.atoms.numberOfAtoms())
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
                    if(i != j and 
                       not (j in self.exclusion_list[i]) and
                       np.sum((positions[i] - positions[j])**2) < self.cutoff ** 2):
                        self.nlist[nlist_count] = j
                        self.nlist_lengths[i] += 1
                        nlist_count += 1
                    j = self.cells[j]

        self.nlist = self.nlist[:nlist_count]
        return self.nlist, self.nlist_lengths


class PairwiseCat(ForceCategory):
    """Pairwise force category. It handles constructing a neighbor-list at each time-step. 
    """
    instance = None

    @staticmethod
    def get_instance(cutoff):        
        if(PairwiseCat.instance is None):
            PairwiseCat.instance = PairwiseCat(cutoff)
        else:
            #check cutoff
            if(PairwiseCat.instance.cutoff != cutoff):
                raise RuntimeError("Incompatible cutoffs")
        return PairwiseCat.instance
    
    def __init__(self, cutoff=12):
        super(PairwiseCat, self).__init__()
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

    
class BondCat(ForceCategory):

    """Bond category. It caches each atoms bonded neighbors when constructued
    """
    instance = None

    @staticmethod
    def get_instance():        
        if(BondCat.instance is None):
            BondCat.instance = BondCat()
        return BondCat.instance
    
    def __init__(self):
        super(BondCat, self).__init__()
         self.blist_ready = False

    def _build_blist(self, u):
        self.blist = [[] for x in range(u.atoms.numberOfAtoms())]
        for b in u.bonds:
            self.blist[b.atom1.number].append(b.atom2.number)
            self.blist[b.atom2.number].append(b.atom1.number)

        self.blist_ready = True                    

    def _setup(self, u):
        if(not self.blist_ready):
            self._build_blist(u)

    def _teardown(self):
        self.nlist_ready = False
        
    def _setup_update(self,u):
        self._setup(u)

    def _teardown_update(self):
        self._teardown()

    @property
    def nlist():
        return self.blist
