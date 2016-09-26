from MDAnalysis import Writer
from MDAnalysis.core.AtomGroup import Universe, AtomGroup, Atom, Residue, Segment
from MDAnalysis.core.topologyobjects import Bond
from MDAnalysis.units import get_conversion_factor
import MDAnalysis.coordinates.base as base
import numpy as np
import scipy.sparse as npsp
from ForceMatch import same_img
from States import State_Mask
import os
import ForceCategories as ForceCategories

try:
    from mpi4py import MPI
    mpi_support = True
except ImportError as e:
    mpi_support = False
    mpi_error = e




class AtomGroupWrapper(object):
    def __init__(self, ag):
        self.ag = ag
        self.state = None

    def clear_state(self):
        self.state = None
    

class CGUniverse(Universe):
    ''' Class which uses center of mass mappings to reduce the number
        of degrees of freedom in a given trajectory/structure
        file. The selections define how atoms are grouped PER
        residue. Thus, if there is only one residue defined, then all
        atoms in that selection will be one bead in the CG system.
        
        A residue_reduction_map may be used so that multiple residues
        are combined. It should be an array with a length equal to the
        total number of desired output residues. Each element should
        be an array containing indices that point to the fine-grain
        universe residue indices.
    '''

    def __init__(self, otherUniverse, selections, names = None, collapse_hydrogens = False, lammps_force_dump = None, residue_reduction_map = None):
        #this line of code here should work,
        super(Universe, self).__init__()
        #but I don't know why it doesn't. So instead I also do this:
        self._cache = dict()
        self._topology = dict()
        self.atoms = AtomGroup([])


        if(names is None):
            names = [ '%dX' % x for x in range(len(selections))]
        if(len(names) != len(selections)):
            raise ValueError('Length of slections (%d) must match lenght of names (%d)' % (len(selections), len(names)))
        self.fgref_u = otherUniverse
        self.atoms = AtomGroup([])
        self.selections = selections
        self.names = names
        self.chydrogens = collapse_hydrogens
        self.residue_reduction_map = residue_reduction_map

        if(lammps_force_dump):
            self.lfdump = open(lammps_force_dump, 'r')
            #find which column is id and which are forces
            line = ""
            while(not line.startswith('ITEM: ATOMS')):
                line = self.lfdump.readline()

            self.lfdump_map = {'id':-1, 'force':-1}
            for i, item in enumerate(line.split()):
                if(item == 'id'):
                    self.lfdump_map['id'] = i-2
                elif(item == 'fx'):
                    self.lfdump_map['force'] = i-2
            if(self.lfdump_map['id'] == -1 or self.lfdump_map['force'] == -1):
                raise IOError('The lammps force dump, {}, does not contain a force or id column'.format(lammps_force_dump))
            
                
        else:
            self.lfdump = None
            self.lfdump_map = None


        self._build_structure()
        
        print('Topology mapping by center of mass, forces by sum')
        print('This is %s a periodic trajectory. %d Frames' % ('' if self.trajectory.periodic else 'not', len(self.trajectory)))

    @property
    def universe(self):
        return self


    def _build_structure(self):
        #atoms cannot be written out of index order, so we 
        #need to iterate residue by residue
        index = 0
        fgtocg_incl_map = {} #keys = fine atoms, values = coarse beads
        residues = {}
        #keep track of selections, so we can throw a useful error if we don't end up selecting anything
        selection_count = {}
        segments = {}
        for s in self.selections:
            selection_count[s] = 0

        #if we're reducing the residues, we'll need to take care of that
        ref_residues = self.fgref_u.residues
        if(self.residue_reduction_map):
            #reduce them
            ref_residues = []
            for i,ri in enumerate(self.residue_reduction_map):
                ref_residues.append(Residue(name='CMB', id=i+1, 
                                            atoms=reduce(lambda x,y: x+y, [self.fgref_u.residues[j] for j in ri]),
                                            resnum=i+1))
        total_masses = []
        for r in ref_residues:
            residue_atoms = []
            for s,n in zip(self.selections, self.names):
                group = r.select_atoms(s)

                #check if there were any selected atoms
                if(len(group) == 0):
                    continue

                selection_count[s] += len(group)
                
                #calculate the total mass lumped into the new CG atom
                total_mass = sum([x.mass if x in group else 0 for x in r])
                if(sum([1 if x in group else 0 for x in r]) > 0 and total_mass == 0):                    
                    raise ValueError('Zero mass CG particle found! Please check all-atom masses and/or set them manually via \"fine_grain_universe.select_atoms(...).set_mass(...)\"')
                #make new atom, using total mass as mass
                #masses are corrected after hydrogens are collapsed in
                #and the fg to cg maps are defined 
                a = Atom(index, n, n, r.resname, r.resid, r.atoms[0].segid, total_mass, 0, universe=self) 
                
                index += 1

                for ra in group:
                    if(ra in fgtocg_incl_map):
                        raise ValueError('Attemtping to map {} to {} and {}'.format(ra, a, fgtocg_incl_map[ra]))
                    fgtocg_incl_map[ra] = a

                #append atom to new residue atom group
                residue_atoms.append(a)
                #add the atom to Universe
                self.atoms += a
            #now actually create new residue and give atoms a reference to it
            residues[r.resid] = Residue(r.resname, r.resid, residue_atoms, resnum=r.resnum)
            for a in residue_atoms:
                a.residue = residues[r.resid]

            #take care of putting residue into segment
            segid = None if len(residue_atoms) == 0 else residue_atoms[0].segid
            if(segid in segments):
                segments[segid].append(residues[r.resid])
            elif(segid):
                segments[segid] = [residues[r.resid]]


        #check to make sure we selected something
        total_selected = 0
        for s in self.selections:
            count = selection_count[s]
            total_selected += count
            if(count == 0):
                raise ValueError('Selection "%s" matched no atoms' % s)        

        #check counting
        if(len(self.fgref_u.atoms) < total_selected):
            print 'Warining: some atoms placed into more than 1 CG Site'
        elif(len(self.fgref_u.atoms) > total_selected):
            print 'Warning: some atoms not placed into CG site'
        

        #find hydrogens and collapse them into beads 
        if(self.chydrogens):
            for b in self.fgref_u.bonds:
                #my hack for inferring a hydrogen
                for a1,a2 in [(b[0], b[1]), (b[1], b[0])]:
                    if(a1.type.startswith('H') and a1.mass < 4.):
                        fgtocg_incl_map[a1] = fgtocg_incl_map[a2]
                        #add the mass
                        fgtocg_incl_map[a2].mass += a1.mass        

        #generate matrix mappings for center of mass and sum of forces
        # A row is a mass normalized cg site defition. or unormalized 1s for forces
        self.pos_map = npsp.lil_matrix( (len(self.atoms), len(self.fgref_u.atoms)) , dtype=np.float32)
        self.force_map = npsp.lil_matrix( (len(self.atoms), len(self.fgref_u.atoms)) , dtype=np.float32)

        #keep a forward map for use in state-masks
        #The key is a CG atom number and the value is an atom group of the fine-grain atoms
        self.cgtofg_fiber_map = [[] for x in range(len(self.atoms))]

        for a in self.fgref_u.atoms:
            try:
                self.pos_map[fgtocg_incl_map[a].index, a.index] = a.mass / fgtocg_incl_map[a].mass
                self.force_map[fgtocg_incl_map[a].index, a.index] = 1.
                self.cgtofg_fiber_map[fgtocg_incl_map[a].index] += [a.index]
            except KeyError:
                #was not selected
                pass

        #set the masses correctly now that the total is used and
        #a cg to fg map is available
        #see Noid et al 2008 (MS-CG I), M_I = (\sum_i c_Ii^2 / m_i)^-1 for nonzero c_Ii
        #c_Ii = fg atom mass / cg atom total mass for all fg atoms included
        #in the definition of cg atom I
        for cg_idx in range(len(self.atoms)):
            new_mass = 1.0 / sum([(self.fgref_u.atoms[fg_idx].mass / self.atoms[cg_idx].mass ** 2) for fg_idx in self.cgtofg_fiber_map[cg_idx]])
            self.atoms[cg_idx].mass = new_mass

        for i in range(len(self.atoms)):
            #convert the list of atom indices into an AtomGroup object
            #it has to be wraped so we can manipulate it later
            self.cgtofg_fiber_map[i] = AtomGroupWrapper(reduce(lambda x,y: x + y, [self.fgref_u.atoms[x] for x in self.cgtofg_fiber_map[i]]))
        
        #Put them into efficient sparse matrix.
        self.pos_map = self.pos_map.tobsr()
        self.force_map = self.force_map.tobsr()
                                    
        #add bonds using the reverse map
        #There is new syntax in 0.92 that uses topolgy groups instead of lists.
        #delete by removing
        for b in self.bonds:
            self.bonds -= b

        for b in self.fgref_u.bonds:
            try:
                cgatom1 = fgtocg_incl_map[b[0]]
                cgatom2 = fgtocg_incl_map[b[1]]
                exists = False
                for cbg in self.bonds:
                    if(cbg[0] in [cgatom1, cgatom2] and cbg[1] in [cgatom1, cgatom2]):
                        exists = True
                        break
                if not exists and cgatom1 != cgatom2:
                    self.bonds += Bond( (cgatom1, cgatom2) )
                    
            except KeyError:
                #was not in selection
                pass

        self.__trajectory = CGReader(self, self.fgref_u.trajectory, self.pos_map, self.force_map, self.lfdump, self.lfdump_map)
        
        #take care of segments now
        segment_groups = {}
        for k,v in segments.iteritems():
            segment_groups[k] = Segment(k, v)
        for a in self.atoms:
            a.segment = segment_groups[a.segid]
            
        self.atoms._rebuild_caches()


    
    @property
    def trajectory(self):
        return self.__trajectory

    def cache(self, directory='cg_cache', maximum_frames=0):
        '''This precomputes the trajectory and structure so that it doesn't need to be 
           recalculated at each timestep. Especially useful for random access.
           Returns a Universe object corresponding to the cached trajectory
        '''

        write_files = True
        
        if(mpi_support):
            #check if we're running with MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            write_files = rank == 0
        
        if(not os.path.exists(directory) and write_files):
            os.mkdir(directory)

        structure = os.path.join(directory, 'cg.pdb')
        trajectory = os.path.join(directory, 'cg.trr')

        if(write_files):
            write_structure(self, structure, bonds='all')
            write_trajectory(self, trajectory, maximum_frames)        

        #sync area
        if(mpi_support):
            
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            #synchronize by using a broadcast
            comm.bcast(0)
            

        u = Universe(structure, trajectory)
        u.trajectory.periodic = self.trajectory.periodic
        apply_mass_map(u, create_mass_map(self))
        return u

    def make_state_mask(self, state_function, state_number):
        """Creates a State_Mask object. The sate_function should take
           an MDAnalysis atom group from the fine-grain universe that
           made up the CG site. The function should return a vector of
           membership percentages for each state. The number of states
           is state_number. This returns an array of masks for each
           possible state"""
        return [State_Mask(self.cgtofg_fiber_map, state_function, x) for x in range(state_number)]

class Timestep(base.Timestep):
    
    def set_ref_traj(self, aatraj):
        self.aatraj = aatraj
        
    @property
    def dimensions(self):        
        return self.__dimensions
    @dimensions.setter
    def dimensions(self, value):
        self.__dimensions = value

class CGReader(base.Reader):

    def __init__(self, universe, aatraj, pos_map, force_map, lfdump, lfdump_map):

        self.u = universe
        self.aatraj = aatraj

        self.pos_map = pos_map
        self.force_map = force_map
        
        self.lfdump = lfdump
        self.lfdump_map = lfdump_map

        self.units = aatraj.units
        self.numatoms = np.shape( pos_map )[0] #The topology matrix mapping should have a number of rows equal to cg atoms
        try:
            self.periodic = aatraj.periodic
        except AttributeError:
            self.periodic = False
            
        self.__len__ = len(aatraj)
        self.n_frames = len(aatraj)
        try:
            self.skip_timestep  = aatraj.skip_timestep
        except AttributeError:
            self.skip_timestep = 0
        self.ts = Timestep(self.numatoms)
        self.ts.set_ref_traj(aatraj)
        self.ts.dimensions = aatraj.ts.dimensions
        self.ts._pos = np.empty((self.numatoms,3), dtype=np.float32)
        self.ts._velocities = np.copy(self.ts._pos)
        self.ts._forces = np.copy(self.ts._pos)
        self._read_next_timestep(ts=self.aatraj.ts)
        
    def close(self):
        self.aatraj.close()
    
    def __del__(self):
        self.close()
    
    def __iter__(self):
        def iterCG():
            for i in range(0, len(self), self.skip):
                try: yield self._read_next_timestep()
                except IOError: raise StopIteration
        return iterCG()

    def _read_next_timestep(self, ts=None):
        if(ts is None):
            ts = self.aatraj.next()
        self.ts.frame = ts.frame

        print self.ts.frame

        #now, we must painstakingly and annoyingly put each cg group into the same periodic image
        #as its residue 
        if(self.periodic):
            dims = ts.dimensions
            for r in self.u.fgref_u.residues:
                centering_vector = np.copy(r.atoms[0].pos)
                for a in r.atoms:
                    a.pos[:] =  same_img(a.pos[:], centering_vector, dims)
                                            
        self.ts._pos = self.pos_map.dot( ts._pos )
        try:
            self.ts._velocities[:] = self.pos_map.dot( ts._velocities ) #COM motion
        except AttributeError:
            self.ts._velocities[:] = np.zeros( (self.numatoms, 3) ,dtype=np.float32)

        #check to see if we have a lammps force dump for forces
        if(self.lfdump):
            #we do, let's read it
            forces = np.zeros( (np.shape(self.pos_map)[1], 3), dtype=np.float32)
            while(not self.lfdump.readline().startswith('ITEM: ATOMS')):
                pass
            for i in range(len(forces)):
                sline = self.lfdump.readline().split()
                #NOTE: Forces from lammps seem to be negative
                try:
                    forces[int(sline[self.lfdump_map['id']]) - 1,:] = [-float(x) for x in sline[self.lfdump_map['force']:]]                        
                except ValueError:
                    raise IOError( 'Invalid forces line at %s' % reduce(lambda x,y: x + ' ' + y, sline))
            self.ts._forces[:] = self.force_map.dot( forces) 
        else:
            try:
                self.ts._forces[:] = self.force_map.dot( ts._forces )
            except AttributeError:
                #can't find any forces, use 0
                self.ts._forces[:] = np.zeros( (self.numatoms, 3) ,dtype=np.float32)


        #if we're doing multistate stuff, we need to clear the states each step
        if(type(self.u) == CGUniverse):
            for ag in self.u.cgtofg_fiber_map:
                ag.clear_state()

        return self.ts

    def rewind(self):
        self.aatraj.rewind()

#END CGUniverse Stuff

def write_structure(universe, filename, **args):
    '''A symmetric version of the write_trajectory method
    '''
    universe.atoms.write(filename, **args)

def _bond_neighbor_generator(origin, bonds, exclude = []):
    '''generator that yields bonded neighbors of a given atom. Pass
    exclusion list to avoid certain atoms
    '''
    exclude.append(origin)
    for a1, a2 in [(x.atom1, x.atom2) for x in bonds]:
        if(origin in (a1,a2)):
            if a1 not in exclude:
                yield a1
            if a2 not in exclude:
                yield a2

def _angle_generator(bonds):
    '''Generator that yields all possible angles from a set of bonds
    '''
    for a1, a2 in [(x.atom1, x.atom2) for x in bonds]:
        for a3 in _bond_neighbor_generator(a2, bonds, [a1]):
            yield (a1, a2, a3)

def _dihedral_generator(bonds):        
    '''Generator that yields all possible dihedrals from a set of bonds
    '''
    for a1, a2 in [(x.atom1, x.atom2) for x in bonds]:
        for a3 in _bond_neighbor_generator(a2, bonds, [a1]):
            for a4 in _bond_neighbor_generator(a3, bonds, [a1, a2]):
                yield (a1, a2, a3, a4)
                


def write_lammps_data(universe, filename, atom_type=None, bonds=True, angles=False, dihedrals=False, impropers=False, force_match=None):

    '''Write out a lammps file from a Universe. This will return the
    atom_type of the data file written out.
    '''

    #cases not implemented
    if(force_match and angles):
        raise NotImplementedError()
    if(impropers):
        raise NotImplementedError()

    possible_atom_types = ('atomic', 'full', 'molecular', 'bond', 'charge')
    if atom_type and atom_type not in possible_atom_types:
        raise ValueError('atom_type must be one of {}'.format(possible_atom_types))

    atom_section = []
    mass_section = []
    bond_section = []
    angle_section = []
    dihedral_section = []
    type_map = {}
    atom_types = 1
    positions = universe.atoms.get_positions()
    has_charges = False
    for a in universe.atoms:
        if(abs(a.charge) > 0):
            has_charges = True
    if(force_match):
        type_count = force_match.get_force_type_count()

    #determine atom type if necessary
    if(not atom_type):
        atom_type = 'atomic'
        if(bonds):
            if(angles):
                if(has_charges):
                    atom_type = 'full'
                else:
                    atom_type = 'molecular'
            else:
                atom_type = 'bond'
        elif(has_charges):
            atom_type = 'charge'

        
    for i,a in zip(range(len(universe.atoms)), universe.atoms):
        assert i == a.index, 'Atom indices are jumbled. Atom %d has number %d' % (i, a.index)

        if(not a.type in type_map):
            type_map[a.type] = atom_types
            mass_section.append('%d %f\n' % (atom_types, a.mass))
            atom_types += 1


        if(atom_type == 'full'):
            atom_section.append('%d %d %d %f %f %f %f\n' % (i+1, a.resid, 
                                                            type_map[a.type],
                                                            a.charge, 
                                                            positions[i,0],
                                                            positions[i,1],
                                                            positions[i,2]))                
        elif(atom_type == 'molecular' or atom_type == 'bond'):
            atom_section.append('%d %d %d %f %f %f\n' % (i+1, a.resid, 
                                                         type_map[a.type],
                                                         positions[i,0],
                                                         positions[i,1],
                                                         positions[i,2]))
        elif(atom_type == 'charge'):
            atom_section.append('%d %d %f %f %f %f\n' % (i+1, type_map[a.type],
                                                         a.charge,
                                                         positions[i,0],
                                                         positions[i,1],
                                                         positions[i,2]))
        elif(atom_type == 'atomic'):
            atom_section.append('%d %d %f %f %f\n' % (i+1, type_map[a.type],
                                                      positions[i,0],
                                                      positions[i,1],
                                                      positions[i,2]))



    bindex = 1
    if(bonds):
        btypes = {}
        for b in universe.bonds:
            if(force_match):
                btype = force_match.get_bond_type_index(b.atom1, b.atom2)
            else:
                #create type which is cat of types, eg HOH 
                btype = ''.join([x.type for x in [b.atom1, b.atom2]])
                temp =  ''.join([x.type for x in [b.atom2, b.atom1]])
                #alphabeticl ends
                if(temp > btype):
                    btype = temp
                if(btype not in btypes):
                    btypes[btype] = len(btypes)
                btype = btypes[btype]
              
  
            if(btype is not None):
                bond_section.append('%d %d %d %d\n' % (bindex, btype,
                                                   b.atom1.index+1, b.atom2.index+1))
                bindex += 1
    aindex = 1
    if(angles):
        atypes = {}
        for a in _angle_generator(universe.bonds):
            #create type which is cat of types, eg HOH 
            atype = ''.join([x.type for x in a])
            temp =  ''.join([x.type for x in reversed(a)])
            #alphabeticl ends
            if(temp > atype):
                atype = temp
            
            if(atype not in atypes):
                atypes[atype] = len(atypes)
            
            angle_section.append('%d %d %d %d %d\n' % (aindex, atypes[atype],
                                                       a[0].index+1, a[1].index+1, a[2].index+1))
            aindex += 1

    dindex =1 
    if(dihedrals):
        dtypes = {}
        for d in _dihedral_generator(universe.bonds):
            #create type which is cat of types, eg HOH 
            dtype = ''.join([x.type for x in d])
            temp =  ''.join([x.type for x in reversed(d)])
            #alphabeticl ends
            if(temp > dtype):
                dtype = temp
            
            if(dtype not in dtypes):
                dtypes[dtype] = len(dtypes)
            
            dihedral_section.append('%d %d %d %d %d %d\n' % (dindex, dtypes[dtype],
                                                       d[0].index+1, d[1].index+1, d[2].index+1, d[3].index+1))
            dindex += 1
            
            
            
    improper_section = []
        
    with open(filename, 'w') as output:

        #make header
        output.write('Generated by ForcePy.CGMap.py\n\n')
        output.write('%d atoms\n' % len(universe.atoms))
        output.write('%d bonds\n' % (bindex - 1))
        output.write('%d angles\n' % (aindex - 1))
        output.write('%d dihedrals\n' % (dindex - 1))
        output.write('%d impropers\n\n' % 0)

        output.write('%d atom types\n' % (atom_types - 1))
        output.write('%d bond types\n' % (type_count[ForceCategories.Bond] if force_match else len(btypes)))
        output.write('%d angle types\n' % (type_count[ForceCategories.Angle] if force_match else len(atypes)))
        output.write('%d dihedral types\n' % (type_count[ForceCategories.Dihedral] if force_match else len(dtypes)))
        output.write('%d improper types\n\n' % (type_count[ForceCategories.Improper] if force_match else 0))

        output.write('%f %f xlo xhi\n' % (0,universe.trajectory.ts.dimensions[0]))
        output.write('%f %f ylo yhi\n' % (0,universe.trajectory.ts.dimensions[1]))
        output.write('%f %f zlo zhi\n\n' % (0,universe.trajectory.ts.dimensions[2]))

        #the rest of the sections        
        output.write('Masses\n\n')
        output.write(''.join(mass_section))
        output.write('\nAtoms\n\n')
        output.write(''.join(atom_section))
        if(bonds):
            output.write('\nBonds\n\n')
            output.write(''.join(bond_section))
            if(angles):
                output.write('\nAngles\n\n')
                output.write(''.join(angle_section))
                if(dihedrals):
                    output.write('\nDihedrals\n\n')
                    output.write(''.join(dihedral_section))
                    if(impropers):
                        output.write('\nImpropers\n\n')
                        output.write(''.join(improper_section))

    return atom_type
    
    

def write_trajectory(universe, filename, maximum_frames=0):
    '''A simplified method for writing trajectories
    '''
    w = Writer(filename, len(universe.atoms))
    frame_count = 0
    for ts in universe.trajectory:
        if(maximum_frames != 0 and frame_count == maximum_frames):
            break
        w.write(ts)
        frame_count += 1
    w.close()

def create_mass_map(universe):
    '''Create a map of masses for atom types in a universe so that they can be applied to a universe without masses assigned
       to atoms. This is useful if writing a universe with a format that doesn't include masses (gro, pdb).
    '''
    mass_map = {}
    for a in universe.atoms:
        if(not a.type in mass_map):
            mass_map[a.type] = a.mass
    return mass_map

def apply_mass_map(universe, mass_map):
    '''Apply a mass_map created by a call to `create_mass_map`
    '''
    for k,v in mass_map.iteritems():
        universe.select_atoms('type {}'.format(k)).set_mass(v)


def add_residue_bonds(universe, selection1, selection2):
    '''This function will add bonds between atoms mathcing selections and 2
    within any residue
    '''
    count = 0 
    for r in universe.atoms.residues:        
        group1 = r.select_atoms(selection1)
        group2 = r.select_atoms(selection2)            
        for a1 in group1:
            for a2 in group2:
                if(a1 != a2):
                    universe.bonds += Bond( (a1, a2) )
                    count += 1
    print 'Added %d bonds' % count

def add_residue_bonds_table(universe, table, residue_name):
    '''This function will add bonds within a residue by a table.  The
       table should be 2 columns, where each row is a bond. Index
       starts at 0.
    '''
    count = 0
    for r in universe.atoms.residues:
        if(r.name != residue_name):
            continue
        for row in table:
            universe.bonds += ( Bond(r[row[0]], r[row[1]]) )
            count += 1
    print 'Added %d bonds' % count
            

def add_sequential_bonds(universe, selection1=None, selection2=None):
    '''this function will add bonds between sequential residues. The atoms within the residue
       chosen for the bonding are simply the first and last if no selections are given
    '''
    count = 0
    if((selection1 or selection2) and not (selection1 and selection2)):
        raise ValueError('Must give 2 selections for two connecting sites')
    for s in universe.atoms.segments:
        last = None
        for r in s.atoms.residues:
            if(last is not None):
                if(selection1):
                    try:
                        universe.bonds.append( Bond(r.select_atoms(selection1)[0], last) )
                    except IndexError:
                        raise ValueError('Could not find {} in {}'.format(selection1, r))
                else:
                    universe.bonds.append( Bond(r.atoms[0], last) )
                count += 1
            if(selection2):
                last = r.atoms.select_atoms(selection2)[-1]
            else:
                last = r.atoms[-1]

    print 'Added %d bonds' % count
