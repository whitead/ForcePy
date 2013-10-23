import random, os, json
import numpy as np
import numpy.linalg as ln
from math import ceil
from MDAnalysis import Universe
from math import *
import ForcePy.ForceCategories as ForceCategories
try:
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    plotting_support = True
except ImportError as e:
    plotting_support = False
    plotting_error = e
    
from ForcePy.Util import *
from ForcePy.ForceCategories import *
from ForcePy.CGMap import CGUniverse, apply_mass_map, create_mass_map, write_lammps_data
try:
    from mpi4py import MPI
    mpi_support = True
except ImportError as e:
    mpi_support = False
    mpi_error = e


class ForceMatch:
    """Main force match class.
    """
    
    def __init__(self, cguniverse, input_json = None):
        self.ref_cats = []
        self.tar_cats = []
        self.ref_forces =  []
        self.tar_forces = []
        self.u = cguniverse
        if(input_json):
            self._load_json(input_json) 
        else:
            self.json = []
        self.force_match_calls = 0
        self.plot_frequency = 1 if plotting_support else -1
        self.plot_output = None
        self.atom_type_map = None
        self.tar_force_buffer = None
        self.send_buffer = None
        self.rec_buffer = None


    
    def _load_json(self, input_file):
        with open(input_file, 'r') as f:
            self.json = json.load(f)
        self._test_json(self.json, [])

        if("observable" in self.json):
            self._test_json(self.json, [('kT', 'Boltzmann\'s constant times temperature'), 
                                        ('observable', 'File containing two columns where the number of rows is equal to frames. First column is total potential energy and second is observable')])

            self.kt = self.json['kT']
            assert type(self.kt) == type(0.), 'kT must be a floating point number'

            self.do_obs = True
            self.obs = [0 for x in range(self.u.trajectory.numframes)]
            self.obs_energy = [0 for x in range(self.u.trajectory.numframes)]
            with open(self.json['observable'], 'r') as f:
                lines = f.readlines()
                if(len(lines) < len(self.obs)):
                    raise IOError('Number of the frames (%d) does not match number of lines in observation file (%d)' %
                                  (len(self.obs), len(lines)))
                for i, line in zip(range(len(self.obs)), lines[:len(self.obs)]):
                    self.obs_energy[i] = float(line.split()[0])
                    self.obs[i] = float(line.split()[1])

            if('target_observable' in self.json):
                self.target_obs = self.json['target_observables']
                assert type(self.target_obs) == type(0.), 'observable target must be a floating point number'

        if('box' in self.json):
            if(len(self.json['box']) != 3):
                raise IOError('Input file JSON: box must look like \"box\":[5,5,5]. It must have 3 dimensions in an array')

                
                
    def _test_json(self, json, required_keys = [("kT", "Boltzmann's constant times temperature")]):
        for rk in required_keys:
            if(not json.has_key(rk[0])):
                raise IOError("Error in input file, could not find %s\n. Set using %s keyword" % (rk[1], rk[0]))

    def __getstate__(self):
        odict = self.__dict__.copy()
        #remove universe, we don't want to serialize all that
        del odict['u']
        
        if(type(self.u ) == CGUniverse):
            raise ValueError("Cannot pickle a CGUniverse. cache() must be called on it to convert to Universe")
        
        #store the filename and trajectory of the universe
        odict['structure_filename'] = self.u.filename
        odict['trajectory_filename'] = self.u.trajectory.filename
        odict['mass_map'] = create_mass_map(self.u)
        odict['trajectory_is_periodic'] = self.u.trajectory.periodic

        return odict
        
    def __setstate__(self, dict):
        self.__dict__.update(dict)
        #reconstruct the universe
        self.u = Universe(dict['structure_filename'], dict['trajectory_filename'])
        apply_mass_map(self.u, dict['mass_map'])
        self.u.trajectory.periodic = dict['trajectory_is_periodic']

        

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
                                
    def force_match_mpi(self, batch_size = None, do_plots = False, repeats = 1, frame_number=0, quiet=False):
        
        if(not mpi_support):
            raise mpi_error
        
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        
        if(size == 1):
            raise RuntimeError("MPI should not be run on a single process. Call force_match instead")

        if(do_plots and rank == 0):
            self._setup_plot()

        frame_number = frame_number if frame_number > 0 else self.u.trajectory.numframes


        if(batch_size):
            index = 0
            while(index * size * batch_size < frame_number * repeats):
                try:
                    self._distribute_tasks(batch_size, index * batch_size, quiet=quiet, frame_number=frame_number)
                except (EOFError, IOError):
                    #just finished reading the file, eat the exception. Will be rewound in force_match_task
                    pass

                self._reduce_tasks()
                index +=1
                if(rank == 0 and not quiet):
                    print "%d / %d iterations" % (index * size * batch_size, frame_number * repeats)  
                    if(do_plots):                    
                        self._plot_forces()

        else:
            for i in range(repeats):
                try:
                    self._distribute_tasks(quiet=quiet, frame_number=frame_number)
                except (EOFError, IOError):
                    #just finished reading the file, eat the exception. Will be rewound in force_match_task                    
                    pass
                    
                self._reduce_tasks()
                if(rank == 0 and not quiet):
                    print "%d / %d iterations" % (i+1, repeats)  
                    if(do_plots):                    
                        self._plot_forces()

        if(rank == 0):
            print "Complete"
            if(do_plots):
                self._teardown_plot()

        

    def force_match(self, iterations = 0):

        if(iterations == 0):
            iterations = self.u.trajectory.numframes
        
        ref_forces = np.zeros( (self.u.atoms.numberOfAtoms(), 3) )
        self.u.trajectory.rewind() # just in case this is called after some analysis has been done
        
        #setup plots
        if(self.plot_frequency != -1):
            self._setup_plot()

 
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
            if(self.plot_frequency != -1 and iterations % self.plot_frequency == 0):
                self._plot_forces()

            #track error
            net_df = 0
            self.force_match_calls += 1

            #sample particles and run updates on them 
            for i in random.sample(range(self.u.atoms.numberOfAtoms()),self.u.atoms.numberOfAtoms()):

                #calculate net forces deviation
                df = np.array(ref_forces[i], dtype=np.float32)
                mag_temp = ln.norm(df)
                for f in self.tar_forces:
                    df -= f.calc_particle_force(i,self.u)                    
                net_df +=  ln.norm(df) / mag_temp

                #now run gradient update step on all the force types
                for f in self.tar_forces:
                    f.update(df)

            ref_forces.fill(0)
            self._teardown()

            print "avg relative magnitude error at %d  = %g" % (iterations, net_df / self.u.atoms.numberOfAtoms())
            
            iterations -= 1
            if(iterations == 0):
                break

        if(not self.plot_output is None):
            self._save_plot()
        if(self.plot_frequency != -1):
            self._teardown_plot()

    def _force_match_task(self, start, end, do_print = False):
        ref_forces = np.zeros( (self.u.atoms.numberOfAtoms(), 3) )

        
        self.u.trajectory.rewind()
        for i in range(start):
            self.u.trajectory.next()
        
        for tsi in range(start,end):
            ts = self.u.trajectory.ts
            
            #set box if necessary
            if("box" in self.json):
                #strange ordering due to charm
                self.u.trajectory.ts._unitcell[0] = self.json["box"][0]
                self.u.trajectory.ts._unitcell[2] = self.json["box"][1]
                self.u.trajectory.ts._unitcell[5] = self.json["box"][2]

            self._setup()

            for rf in self.ref_forces:
                rf.calc_forces(ref_forces, self.u)            

            #track error
            net_df = 0            
            self.force_match_calls += 1

            #sample particles and run updates on them 
            for i in random.sample(range(self.u.atoms.numberOfAtoms()),self.u.atoms.numberOfAtoms()):
                #calculate net forces deviation
                df = np.array(ref_forces[i], dtype=np.float32)
                mag_temp = ln.norm(df)
                for f in self.tar_forces:
                    df -= f.calc_particle_force(i,self.u)
                net_df += ln.norm(df) / mag_temp

                #now run gradient update step on all the force types
                for f in self.tar_forces:
                    f.update(df)

            ref_forces.fill(0)
            self._teardown()

            if(do_print):
                print "avg relative magnitude error  = %g" % (net_df / self.u.atoms.numberOfAtoms())
                

            if(tsi != end - 1):
                self.u.trajectory.next()



    def _pack_tar_forces(self):
        if(self.send_buffer is None):
            count = 0
            for f in self.tar_forces:
                count += len(f.w)
            self.send_buffer = np.empty((count,), dtype=np.float32)
            
        index = 0
        for f in self.tar_forces:
            self.send_buffer[index:(index + len(f.w))] = f.w[:]
            index += len(f.w)
        
        return self.send_buffer

    def _unpack_tar_forces(self):
        index = 0
        for f in self.tar_forces:
            f.w[:] = self.rec_buffer[index:(index + len(f.w))]
            index += len(f.w)

    def _reduce_tasks(self):

        self._pack_tar_forces()
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        #sum all of em
        if(self.rec_buffer is None):
            self.rec_buffer = np.copy(self.send_buffer)

        comm.Reduce([self.send_buffer, MPI.FLOAT], [self.rec_buffer, MPI.FLOAT])
        
        #average
        if rank == 0:
            self.rec_buffer /= comm.Get_size()
            
        self.rec_buffer = comm.bcast(self.rec_buffer) #broadcast the average
        
        self._unpack_tar_forces()

    def _distribute_tasks(self, batch_size = None, offset = 0, quiet=False, frame_number = 0):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        frame_number = frame_number if frame_number > 0 else self.u.trajectory.numframes
        span = frame_number / size
        
        #get remainder
        spanr = frame_number - size * span

        if(batch_size):
            #use batch size
            self._force_match_task(spanr / 2 + rank * span + offset, spanr / 2 + rank * span + batch_size + offset, rank == 0 and not quiet)
        else:
            #distribute equally on the trajectory
            if(rank < spanr):
                self._force_match_task(rank * (span + 1), (rank + 1) * (span + 1), rank == 0 and not quiet)
            else:
                self._force_match_task(rank * span + spanr, (rank + 1) * span + spanr, rank == 0 and not quiet)
        
    def observation_match(self, target_obs = None, obs_sweeps = 25, obs_samples = None, reject_tol = None, do_plots = True):
        """ Match observations.
        """
        
        #check for obs_samples
        try:
            self.obs
        except AttributeError:
            raise ValueError("Must set observations before calling observation match")  


        if(target_obs is None):
            try:
                target_obs = self.target_obs
            except AttributeError:
                print "Assuming maintainance of observation mean is desired"
                target_obs = sum(self.obs) / len(self.obs)
        if(obs_samples is None):
            obs_samples = max(5, self.u.trajectory.numframes / obs_sweeps)
        if(reject_tol is None):
            reject_tol = obs_samples
                
        #in case force mathcing is being performed simultaneously,
        #we want to cache any force specific parameters so that we
        #can swap them back in afterwards
        self.swap_match_parameters_cache()


        if(self.plot_frequency != -1):
            self._setup_plot()


        #we're going to sample the covariance using importance
        #sampling. This requires a normalization coefficient, so
        #we must do multiple random frames

        s_grads = {} #this is to store the sampled gradients. Key is Force and Value is gradient
        for f in self.tar_forces:
            s_grads[f] = [None for x in range(obs_samples)]
        s_obs = [0 for x in range(obs_samples)] #this is to store the sampled observations
        s_weights = [0 for x in range(obs_samples)]

        if(self.plot_output is None):
            plt.ion()                        
            #set-up plots for 16/9 screen
        plot_w = ceil(sqrt(len(self.tar_forces)) * 4 / 3.)
        plot_h = ceil(plot_w * 9. / 16.)
        for i in range(len(self.tar_forces)):
            self.tar_forces[i].plot(plt.subplot(plot_w, plot_h, i+1))
        plt.show()

            
        for s in range(obs_sweeps):

            self.force_match_calls += 1
            #make plots
            for f in self.tar_forces:
                for f in self.tar_forces:
                    f.update_plot()
                plt.draw()

            #now we esimtate gradient of the loss function via importance sampling
            normalization = 0
                
            #note, this reading method is so slow. I should implement the frame jump in xyz
            rejects = 0
            i = 0

            while i < obs_samples:

                index = self._sample_ts() #sample a trajectory frame
                self._setup()

                 #get weight
                dev_energy = self.obs_energy[index]
                for f in self.tar_forces:
                    dev_energy -= f.calc_potentials(self.u)
                                            
                if(abs(dev_energy /self.kt) > 250):
                    rejects += 1
                    if(rejects == reject_tol):
                        print "Rejection rate of frames is too high, restarting force matching"
                        self._teardown()
                        self.swap_match_parameters_cache()
                        self.force_match(rejects) #arbitrarily using number of rejects for number matces to use
                        self.swap_match_parameters_cache()
                        rejects = 0
                        continue
                    else:
                        self._teardown()
                        continue

                weight = exp(dev_energy / self.kt)
                s_weights[i] = weight
                normalization += weight

                #store gradient and observabels
                s_obs[i] = weight * self.obs[i]
                #we use the index of 1 here because temp_grad normally stores an M x 3 gradient for each dimension. There
                #is no direction though with the potential functional derivative
                for f in self.tar_forces:
                    s_grads[f][i] = self.kt * weight * np.copy(f.temp_grad[:,1])

                i += 1
                self._teardown()

            #At this point, we have the ratios of probabilties under the new vs old potentials along with the observation at each point and potential functional derivative (`temp_grad') 
            #Now we use these weights to actually caclulate the covariance.
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

                #Now the target comes in.
                grad = -2 * (meanobs - target_obs) * grad

                #Update the lipschitz estimate
                f.lip += np.square(grad)
                change = f.eta / np.sqrt(f.lip) * grad
                f.w = f.w - f.eta / np.sqrt(f.lip) * grad

                print "Obs Mean: %g, reweighted mean: %g, target mean:" % (sum(self.obs) / len(self.obs), meanobs, target_obs)

             #make plots
            if(self.plot_frequency != -1):
                self._plot_forces()

        if(self.plot_frequency != -1):
            self._teardown_plot()



    def _teardown_plot(self):
        plt.close()

    def _setup_plot(self):

        if(not plotting_support):
            raise plotting_error

        plot_fig = plt.figure()

        #try to maximize the window
        mng = plt.get_current_fig_manager()
        try:
            mng.frame.Maximize(True)
        except AttributeError:
            try:
                mng.resize(*mng.window.maxsize())
            except AttributeError:
                #mac os x
                mng.resize(1200, 900)



        if(self.plot_output is None):
            plt.ion()                        

        #set-up plots for 16/9 screen
        plot_w = ceil(sqrt(len(self.tar_forces)) * 4 / 3.)
        plot_h = ceil(plot_w * 9. / 16.)
        for i in range(len(self.tar_forces)):
            self.tar_forces[i].plot(plt.subplot(plot_w, plot_h, i+1))
        plt.show()
        plt.ioff()
        

    def _plot_forces(self):
        for f in self.tar_forces:
            f.update_plot()
        plt.draw()

    def _save_plot(self):
        plot_fig.tight_layout()
        plt.savefig(self.plot_output)

    def add_and_type_states(self, force, state_function, state_names):
        if(type(self.u) != CGUniverse):
            raise ValueError("Must use CGUniverse for states")
        masks = self.u.make_state_mask(state_function, len(state_names))
        for i in range(len(state_names)):
            for j in range(i,len(state_names)):
                f = force.clone_force()
                f.specialize_states(masks[i],
                                    masks[j],
                                    state_names[i],
                                    state_names[j])
                self.add_tar_force(f)


    def add_and_type_pair(self, force):
        types = []
        for a in self.u.atoms:
            if(not a.type in types):
                types.append(a.type)
        for i in range(len(types)):
            for j in range(i,len(types)):
                if(force.category.pair_exists(self.u, 'type %s' % types[i], 'type %s' % types[j])):
                    f = force.clone_force()
                    f.specialize_types(types[i], types[j])
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
            tfcat._setup(self.u)        

    def _teardown(self):
        for rfcat in self.ref_cats:
            rfcat._teardown()
        for tfcat in self.tar_cats:
            tfcat._teardown()        


    def write_lammps_tables(self, prefix, force_conv=1, energy_conv=1, dist_conv=1, points=1000):
        
        print "conversion = %g" % force_conv
        
        #table file names
        table_names = {}
        table_names[Pairwise] = open("%s_pair.table" % prefix, 'w')
        table_names[Bond] = open("%s_bond.table" % prefix, 'w')
        table_names[Angle] = open("%s_angle.table" % prefix, 'w')
        table_names[Dihedral] = open("%s_dihedral.table" % prefix, 'w')
        
        #write the files, one file for each category
        for rf in self.tar_forces:
            of = table_names[type(rf.category)]
            rf.write_lammps_table(of, force_conv, energy_conv, dist_conv,points)
            of.write("\n\n")
            
        for f in table_names:
            table_names[f].close()

            

        #generate a snippet of lammps code to load the table
        
        #pairs
        string = ["\npair_style table linear %d\n\n" % points]
        
        for f in self.tar_forces:
            #we try a few times, because not all forces are for 2 types
            if(type(f.category) == Pairwise):
                try:
                    string.append("pair_coeff %d %d %s %s %d\n" % (self.get_atom_type_index(f.sel1),
                                                                   self.get_atom_type_index(f.sel2),
                                                                   table_names[Pairwise].name,
                                                                   f.short_name,
                                                                   f.maxd))
                except AttributeError:
                    try:
                        string.append("pair_coeff * %d %s %s %d\n" % (self.get_atom_type_index(f.sel1),
                                                                      table_names[Pairwise].name,
                                                                      f.short_name,
                                                                      f.maxd))
                    except AttributeError:
                        string.append("pair_coeff * * %s %s %d\n" % (table_names[Pairwise].name,
                                                                     f.short_name,
                                                                     f.maxd))
        #bonds
        index = 0
        for f in self.tar_forces:
            #we try a few times, because not all forces are for 2 types
            if(type(f.category) == Bond):
                if(index == 0):
                    string.append("\nbond_style table linear %d\n\n" % points)
                index += 1
                string.append("bond_coeff %d %s %s\n" % (index,
                                                         table_names[Bond].name,
                                                         f.short_name))

        #Angles
        index = 0
        for f in self.tar_forces:
            #we try a few times, because not all forces are for 2 types
            if(type(f.category) == Angle):
                if(index == 0):
                    string.append("\nangle_style table linear %d\n\n" % points)
                index += 1
                string.append("angle_coeff %d %s %s %d\n" % (index,
                                                             table_names[Angle].name,
                                                            f.short_name))

        #Dihedrals

        index = 0
        for f in self.tar_forces:
            #we try a few times, because not all forces are for 2 types
            if(type(f.category) == Dihedral):
                if(index == 0):
                    string.append("\ndihedral_style table linear %d\n\n" % points)
                index += 1
                string.append("dihedral_coeff %d %s %s %d\n" % (index,
                                                                table_names[Dihedral].name,
                                                                f.short_name))

        #Impropers
        #no table style in lammps, not sure what to do about this one
        return "".join(string)
        
        

    def get_pair_type_index(self, atom1, atom2):
        """Return the index of the pairwise force for this pair
        """
        return self._get_category_type(Pairwise, atom1, atom2)

    def get_atom_type_index(self, atom_type):
        """Return the atom type index for the given type string. Index starts from 1.
        """
        if(self.atom_type_map is None):
            self.atom_type_map = {}
            index = 1
            for a in self.u.atoms:
                if(not a.type in self.atom_type_map):
                    self.atom_type_map[a.type] = index
                    index += 1
        if(type(atom_type) != type("")):
            assert(type(atom_type) == type(self.u.atoms[0]))
            atom_type = atom_type.type
        try:
            return self.atom_type_map[atom_type]
        except KeyError:
            pass
        return -1
            

    def get_bond_type_index(self, atom1, atom2):
        """Return the index of the bond force for this pair
        """
        return self._get_category_type_index(Bond, atom1, atom2)


    def _get_category_type_index(self, category, atom1, atom2):
        """ Count the number of forces with the given category and return
            the index of the force for which atom1 and atom2 are valid.
            Indexing starts at 1.
        """
        index = 0
        for f in self.tar_forces:
            if(type(f.category) == category):
                index += 1
                if f.valid_pair(atom1, atom2):
                    return index

    def get_force_type_count(self):
        """ Returns a dictionary containing the number of types for each force
        """
        type_count = {Bond:0, Angle:0, Pairwise:0, Dihedral:0, Improper:0}
        for f in self.tar_forces:
            try:
                type_count[f.category.__class__] += 1
            except AttributeError:
                pass
        return type_count

        
    def write_lammps_scripts(self, prefix='cg', folder = os.curdir, lammps_units="real", table_points=10000, lammps_input_file=None, force_conversion = -1.0, energy_conversion=None):
        """Using the given ForceMatch and Universe object, this will create a set of input files for Lammps.
    
        The function will create the given folder and put all files
        in it. Tables are generated for each type of force from the
        ForceMatch object, a datafile derived from the current
        timestep of this Universe object and an input script that
        loads the force fields. The given lammps input file will be appended 
        to the input script.
        """
        #before we change directories, we need to get the path of the lammps input file 
        if(lammps_input_file is not None):
            lammps_input_file = os.path.abspath(lammps_input_file)

        if(not os.path.exists(folder)):
            os.mkdir(folder)
        original_dir = os.path.abspath(os.getcwd())
        os.chdir(folder)
 
        #write force tables
        force_info = self.write_lammps_tables('%s_force' % prefix, 
                                        force_conv = force_conversion,
                                        energy_conv = energy_conversion if energy_conversion else -force_conversion,
                                        dist_conv = 1,
                                        points=table_points)

        #write data file                   
        #determin sim type
        type_count = self.get_force_type_count()
        
        sim_type = write_lammps_data(self.u, '%s_fm.data' % prefix, bonds=type_count[ForceCategories.Bond] > 0,
                          angles=type_count[ForceCategories.Angle] > 0, dihedrals=False, impropers=False, 
                                     force_match=self)
            
        #alright, now we prepare an input file
        with open("%s_fm.inp" % prefix, 'w') as output:
            output.write("#Lammps input file generated by ForcePy\n")
            output.write("units %s\n" % lammps_units)
            output.write("atom_style %s\n" % sim_type)
            output.write("read_data %s_fm.data\n" % prefix)            
            output.write(force_info)
            output.write("\n")
            
        #now if an input file is given, add that
            if(lammps_input_file is not None):
                with open(lammps_input_file, 'r') as infile:
                    for line in infile.readlines():
                        output.write(line)

        #now write a pdb, I've found that can come in handy
        try:            
            self.u.atoms.write("%s_start.pdb" % prefix, bonds='all')
        except Exception as e:
            print "Failed to write start PDB ", e

        
        #now go back to original directory
        os.chdir(original_dir)
