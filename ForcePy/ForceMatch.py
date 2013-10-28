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
                                        ('observable', 'File containing one column where the number of rows is equal to frames. First column is total potential energy')])

            self.kt = self.json['kT']
            assert type(self.kt) == type(0.), 'kT must be a floating point number'

            self.do_obs = True
            self.obs = [0 for x in range(self.u.trajectory.numframes)]
            with open(self.json['observable'], 'r') as f:
                lines = f.readlines()
                if(len(lines) < len(self.obs)):
                    raise IOError('Number of the frames (%d) does not match number of lines in observation file (%d)' %
                                  (len(self.obs), len(lines)))
                for i, line in zip(range(len(self.obs)), lines[:len(self.obs)]):
                    self.obs[i] = float(line.split()[0])

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
            self.u = self.u.cache()
        
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
        for f in self.tar_forces + self.ref_forces:
            cat = f.get_category()
            if(not (cat is None)):
                self.ref_cats.append(cat)
            f.setup_hook(self.u)


        

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
            for c,f in zip(self.cache, self.tar_forces):
                c,f.eta = f.eta,c
        except AttributeError:
            self.cache = [f.eta for f in self.tar_forces]


    def output_energies_mpi(self, outfile):
        '''This is for assesing the energy coming out of the force-matching
        '''
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        batch = range(0,len(self.u.trajectory), int(ceil(len(self.u.trajectory) / size)))
        
        self.u.trajectory.rewind()
        
        for i in range(batch[rank]):
            self.u.trajectory.next()

        #build buffer
        if(self.send_buffer is None or len(self.send_buffer) != (batch[1] - batch[0])):
            self.send_buffer = np.empty(batch[1] - batch[0], dtype=np.float32)            

        for i in range(batch[rank], batch[rank + 1]):
            try:
                self.u.trajectory.next()
            except (EOFError, IOError):
                #finished reading the file
                break
            self._setup()
            energy = 0
            for f in self.tar_forces:
                energy += f.calc_potentials(self.u)
            self.send_buffer[i - batch[rank]] = energy
            self._teardown()

        rec_buffer = comm.gather([self.send_buffer, MPI.FLOAT])

        if rank == 0:
            with open(outfile, 'w') as f:
                f.write('{:<16} {:<16}\n'.format('frame', 'pot'))
                i = 1
                for ebatch in rec_buffer:
                    for e in ebatch[0]:
                        if(i == len(self.u.trajectory)):
                            break
                        f.write('{:<16} {:<16}\n'.format(i,e))
                        i += 1
    
        #clear buffers 
        self.send_buffer = self.rec_buffer = None
        

                                
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
                    self._distribute_fm_tasks(batch_size, index * batch_size, quiet=quiet, frame_number=frame_number)
                except (EOFError, IOError):
                    #just finished reading the file, eat the exception. Will be rewound in force_match_task
                    pass

                self._reduce_fm_tasks()
                index +=1
                if(rank == 0 and not quiet):
                    print "%d / %d iterations" % (index * size * batch_size, frame_number * repeats)  
                    if(do_plots):                    
                        self._plot_forces()

        else:
            for i in range(repeats):
                try:
                    self._distribute_fm_tasks(quiet=quiet, frame_number=frame_number)
                except (EOFError, IOError):
                    #just finished reading the file, eat the exception. Will be rewound in force_match_task                    
                    pass
                    
                self._reduce_fm_tasks()
                if(rank == 0 and not quiet):
                    print "%d / %d iterations" % (i+1, repeats)  
                    if(do_plots):                    
                        self._plot_forces()

        if(rank == 0):
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
                    f.update_avg()

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


    def finalize(self):
        '''Call this method before writing to use the average over the force-matching, instead of the last observed weights. This
           is important when convergence is oscillatory or observation matching is being used.
        '''
        
        for f in self.tar_forces:
            f.swap_avg()

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
            f.update_avg()

    def _reduce_fm_tasks(self):

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

    def _distribute_fm_tasks(self, batch_size = None, offset = 0, quiet=False, frame_number = 0):
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

    def _setup_obs_buffers(self):
        #     0          1           2                3-->
        #sample-index  weight weighted-sample   weighted-gradients
        buffer_size = 3
        for f in self.tar_forces:
            buffer_size += len(f.temp_grad[:,1])
        #we need to use the buffer to store from a gather too, so check that size
        if(mpi_support):
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            if(buffer_size < size):
                buffer_size = size

        self.send_buffer = np.empty((buffer_size,), dtype=np.float32)
        self.rec_buffer = np.copy(self.send_buffer)

    def _sample_observation_task(self):
        index = self._sample_ts() #sample a trajectory frame
        self._setup()

        #get weight
        dev_energy = 0
        for f in self.tar_forces:
            dev_energy -= f.calc_potentials(self.u)

        buffer_index = 0
        self.send_buffer[buffer_index] = index
        buffer_index += 1

        #energy diff
        self.send_buffer[buffer_index] = dev_energy
        buffer_index += 1

        #store gradient and observables.
        self.send_buffer[buffer_index] = self.obs[index]
        buffer_index += 1
        #we use the index of 1 here because temp_grad normally stores
        #an M x 3 gradient for each dimension. There is no direction
        #though with the potential functional derivative
        for f in self.tar_forces:
            gsize = len(f.temp_grad[:,1])
            self.send_buffer[buffer_index:(buffer_index + gsize)] = self.kt * f.temp_grad[:,1]
            buffer_index += gsize

        self._teardown()
        
    def _observation_match_mpi_step(self, target_obs, curvature, do_print = True):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        
        #sample covariances, unweighted
        self._sample_observation_task()        

        #find minimum, so we know the offset between them
        #deal with confusing and stupid mpi4py syntax
        data_send = np.array(self.send_buffer[1], dtype=np.float32)
        data_receive = np.empty(1, dtype=np.float32)
        comm.Allreduce([data_send, MPI.FLOAT], [data_receive, MPI.FLOAT], op=MPI.MAX)

        
        #now do weighting
        self.send_buffer[1] = exp((self.send_buffer[1] - data_receive[0]) / self.kt)
        #reuse data_send to store information about 0-weighted frames
        data_send = np.array(self.send_buffer[1] < 0.0000001, dtype=np.float32)
        comm.Allreduce([data_send, MPI.FLOAT], [data_receive, MPI.FLOAT], op=MPI.SUM)


        #before going further, check if we got enough accepted frames (> 1)
        if(size - data_receive[0] <= 3):
            #not close enough
            #update results
            if rank == 0:
                print "{:<16} {:<16} {:<16} {:<16}" .format(sum(self.obs) / len(self.obs), 'NA', target_obs, int(size - data_receive[0]))

            return False            


        #looks good, let's reweight and continue
        self.send_buffer[2:] *= self.send_buffer[1]
        self.rec_buffer.fill(0)
        comm.Reduce([self.send_buffer, MPI.FLOAT], [self.rec_buffer, MPI.FLOAT], op=MPI.SUM)

        #now average results
        if rank == 0:
            normalization = self.rec_buffer[1]
            self.rec_buffer[2:] /= normalization
            meanobs = self.rec_buffer[2]

        #update results
        if rank == 0:
            print "{:<16} {:<16} {:<16} {:<16}" .format(sum(self.obs) / len(self.obs), meanobs, target_obs, int(size - data_receive[0]))

        self.rec_buffer = comm.bcast(self.rec_buffer)

        #now, the second pass covariance calculation
        buffer_index = 3
        for f in self.tar_forces:
            l = buffer_index
            r = buffer_index + len(f.temp_grad[:,1])
            #the rec_buffer contains expected values.
            self.send_buffer[l:r] = (f.temp_grad[:,1] - self.rec_buffer[l:r]) * (self.obs[int(self.send_buffer[0])] - self.rec_buffer[2])
            buffer_index = r
            
        #reduce the covariances
        self.rec_buffer.fill(0)
        comm.Reduce([self.send_buffer, MPI.FLOAT], [self.rec_buffer, MPI.FLOAT], op=MPI.SUM)
        
        #normalize them
        if rank == 0:
            self.rec_buffer[2:] /= normalization            
            #These are the new w_gradients
            #do update
            buffer_index = 3
            for curve, f in zip(curvature, self.tar_forces):
                l = buffer_index
                r = buffer_index + len(f.temp_grad[:,1])
                grad = self.rec_buffer[l:r]
                #target comes in here
                grad = -2 * (meanobs - target_obs) * grad

                #apply any regularization
                for r in f.regularization:
                    grad += r[0](f.w)

                #update
                f.lip += np.square(grad)
                f.w = f.w - f.eta * curve / np.sqrt(f.lip) * grad
                f.update_avg()                

                buffer_index = r
            #pack forces for node 0
            self._pack_tar_forces()
    
        #now we update w for everyone by sending the packed node 0 forces
        self.rec_buffer = comm.bcast(self.send_buffer)
        self._unpack_tar_forces()
                
        return True
            
        
        
    
                            
    def observation_match_mpi(self, target_obs = None, 
                              obs_sweeps = 25, do_plots = True,
                              curve_regularize=True):
        """ Match observations.
        """

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        
        #check for obs_samples
        try:
            self.obs
        except AttributeError:
            raise ValueError("Must set observations before calling observation match")  

        if(target_obs is None):
            try:
                target_obs = self.target_obs
            except AttributeError:
                if(rank == 0):
                    print "Assuming maintainance of observation mean is desired"
                target_obs = sum(self.obs) / len(self.obs)
                
        #Move areas only where there exists curvature
        if(curve_regularize):
            curvature = [np.power(f.w_avg, 2) for f in self.tar_forces]
        else:
            curvature = [np.ones(np.size(f.w_avg), dtype=np.float32) for f in self.tar_forces]

        do_plots = do_plots and rank == 0
        if(do_plots):
            self._setup_plot()

        self._setup_obs_buffers()

        if(rank == 0):
            print "{:<16} {:<16} {:<16} {:<16}" .format("observed" , "reweighted", "target", "acceptance")

        for s in range(obs_sweeps):

            #make plots
            if(do_plots and rank == 0):
                self._plot_forces()
            
            while(not self._observation_match_mpi_step(target_obs, curvature)):
#                if(rank == 0):
#                    print "Rejection rate of frames is too high, restarting force matching"
                self.force_match_mpi(do_plots=False, quiet=True)
#                if(rank == 0):
#                    print "{:<16} {:<16} {:<16} {:<16}" .format("observed" , "reweighted", "target", "acceptance")


             #make plots
            if(do_plots):
                self._plot_forces()

        if(do_plots):
            self._teardown_plot()



    def _teardown_plot(self):
        plt.close()
        for f in self.tar_forces:
            f.teardown_plot()

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

            
    def write(self, folder = os.curdir, table_points=10000, force_conversion = 1.0, energy_conversion=1, distance_conversion=1):

        
        if(mpi_support):
            #check if we're running with MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            if(rank != 0):
                return
        

        if(not os.path.exists(folder)):
            os.mkdir(folder)
        original_dir = os.path.abspath(os.getcwd())
        os.chdir(folder)
 
        try:
            for rf in self.tar_forces:
                with open("{}.txt".format(rf.short_name), 'w') as f:
                    rf.write_table(f, force_conversion, energy_conversion, distance_conversion, table_points)
            import pickle
            pickle.dump(self, open('restart.pickle', 'wb'))
        except (IOError,AttributeError) as e:
            print e
        finally:
            os.chdir(original_dir)

        


    def write_lammps_tables(self, prefix, force_conv=1, energy_conv=1, dist_conv=1, points=1000):

        if(mpi_support):
            #check if we're running with MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            if(rank != 0):
                return
        
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

        
    def write_lammps_scripts(self, prefix='cg', folder = os.curdir, lammps_units="real", table_points=10000, lammps_input_file=None, force_conversion = 1.0, energy_conversion=None):
        """Using the given ForceMatch and Universe object, this will create a set of input files for Lammps.
    
        The function will create the given folder and put all files
        in it. Tables are generated for each type of force from the
        ForceMatch object, a datafile derived from the current
        timestep of this Universe object and an input script that
        loads the force fields. The given lammps input file will be appended 
        to the input script.
        """
        if(mpi_support):
            #check if we're running with MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            if(rank != 0):
                return
                

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
