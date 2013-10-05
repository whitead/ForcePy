from MDAnalysis import Writer
from MDAnalysis.core.AtomGroup import Universe, AtomGroup, Atom, Residue, Segment
from MDAnalysis.topology.core import Bond
from MDAnalysis.core.units import get_conversion_factor
import MDAnalysis.coordinates.base as base
import numpy as np
import scipy.sparse as npsp
from ForcePy.ForceMatch import same_img
import os
import ForcePy.ForceCategories as ForceCategories



#note to self, build atoms, residues, segs add to AtomGroup and then
# call build_cache. Can also call set position on the
# atomgroup. Initializing a residue correctly sets the residue type of
# atoms.


class CGUniverse(Universe):
    """ Class which uses center of mass mappings to reduce the number
        of degrees of freedom in a given trajectory/structure
        file. The selections define how atoms are grouped PER
        residue. Thus, if there is only one residue defined, then all
        atoms in that selection will be one bead in the CG system.
        
        A residue_reduction_map may be used so that multiple residues
        are combined. It should be an array with a length equal to the
        total number of desired output residues. Each element should
        be an array containing indices that point to the fine-grain
        universe residue indices.
    """

    def __init__(self, otherUniverse, selections, names = None, collapse_hydrogens = False, lammps_force_dump = None, residue_reduction_map = None):        
        if(names is None):
            names = [ "%dX" % x for x in range(len(selections))]
        if(len(names) != len(selections)):
            raise ValueError("Length of slections (%d) must match lenght of names (%d)" % (len(selections), len(names)))
        self.ref_u = otherUniverse
        self.atoms = AtomGroup([])
        self.selections = selections
        self.names = names
        self.chydrogens = collapse_hydrogens
        self.universe = self
        self.residue_reduction_map = residue_reduction_map

        if(lammps_force_dump):
            self.lfdump = open(lammps_force_dump, 'r')
        else:
            self.lfdump = None


        self._build_structure()
        
        print("Topology mapping by center of mass, forces by sum")
        print("This is %s a periodic trajectory. %d Frames" % ("" if self.trajectory.periodic else "not", self.trajectory.numframes))



    def _build_structure(self):
        #atoms cannot be written out of index order, so we 
        #need to iterate residue by residue
        index = 0
        reverse_map = {}
        residues = {}
        #keep track of selections, so we can throw a useful error if we don't end up selecting anything
        selection_count = {}
        segments = {}
        for s in self.selections:
            selection_count[s] = 0

        #if we're reducing the residues, we'll need to take care of that
        ref_residues = self.ref_u.residues
        if(self.residue_reduction_map):
            #reduce them
            ref_residues = []
            for i,ri in enumerate(self.residue_reduction_map):
                ref_residues.append(Residue(name="CMB", id=i+1, 
                                            atoms=reduce(lambda x,y: x+y, [self.ref_u.residues[j] for j in ri]),
                                            resnum=i+1))

        for r in ref_residues:
            residue_atoms = []
            for s,n in zip(self.selections, self.names):
                group = r.selectAtoms(s)

                #check if there were any selected atoms
                if(len(group) == 0):
                    continue

                selection_count[s] += len(group)
                
                #make new atom
                new_mass = sum([x.mass if x in group else 0 for x in r])
                if(sum([1 if x in group else 0 for x in r]) > 0 and new_mass == 0):                    
                    raise ValueError('Zero mass CG particle found! Please check all-atom masses and/or set them manually via "fine_grain_universe.selectAtoms(...).set_mass(...)')

                a = Atom(index, n, n, r.name, r.id, r.atoms[0].segid, new_mass, 0) 
                    
                index += 1
                for ra in group:
                    if(ra in reverse_map):
                        raise ValueError("Attemtping to map {} to {} and {}".format(ra, a, reverse_map[ra]))
                    reverse_map[ra] = a

                #append atom to new residue atom group
                residue_atoms.append(a)
                #add the atom to Universe
                self.atoms += a
            #now actually create new residue and give atoms a reference to it
            residues[r.id] = Residue(r.name, r.id, residue_atoms, resnum=r.resnum)
            for a in residue_atoms:
                a.residue = residues[r.id]

            #take care of putting residue into segment
            segid = None if len(residue_atoms) == 0 else residue_atoms[0].segid
            if(segid in segments):
                segments[segid].append(residues[r.id])
            elif(segid):
                segments[segid] = [residues[r.id]]


        #check to make sure we selected something
        total_selected = 0
        for s in self.selections:
            count = selection_count[s]
            total_selected += count
            if(count == 0):
                raise ValueError("Selection '%s' matched no atoms" % s)        

        #check counting
        if(len(self.ref_u.atoms) < total_selected):
            print "Warining: some atoms placed into more than 1 CG Site"
        elif(len(self.ref_u.atoms) > total_selected):
            print "Warning: some atoms not placed into CG site"
        

        #find hydrogens and collapse them into beads 
        if(self.chydrogens):
            for b in self.ref_u.bonds:
                #my hack for inferring a hydrogen
                for a1,a2 in [(b.atom1, b.atom2), (b.atom2, b.atom1)]:
                    if(a1.type.startswith("H") and a1.mass < 4.):
                        reverse_map[a1] = reverse_map[a2]
                        #add the mass
                        reverse_map[a2].mass += a1.mass        

        #generate matrix mappings for center of mass and sum of forces
        # A row is a mass normalized cg site defition. or unormalized 1s for forces
        self.top_map = npsp.lil_matrix( (self.atoms.numberOfAtoms(), self.ref_u.atoms.numberOfAtoms()) , dtype=np.float32)
        self.force_map = npsp.lil_matrix( (self.atoms.numberOfAtoms(), self.ref_u.atoms.numberOfAtoms()) , dtype=np.float32)

        for a in self.ref_u.atoms:
            try:
                self.top_map[reverse_map[a].number, a.number] = a.mass / reverse_map[a].mass
                self.force_map[reverse_map[a].number, a.number] = 1.
            except KeyError:
                #was not selected
                pass
        
        #Put them into efficient sparse matrix.
        self.top_map = self.top_map.tobsr()
        self.force_map = self.force_map.tobsr()
                                    
        #add bonds using the reverse map
        self.bonds = set()
        for b in self.ref_u.bonds:
            try:
                cgatom1 = reverse_map[b.atom1]
                cgatom2 = reverse_map[b.atom2]
                for cbg in self.bonds:
                    if(not (cbg.atom1 in [cgatom1, cgatom2]) and not( cbg.atom2 in [cgatom1, cgatom2])):
                    #OK, no bond exists yet
                        self.bonds.add( Bond(cgatom1, cgatom2) )
            except KeyError:
                #was not in selection
                pass

        self.__trajectory = CGReader(self, self.ref_u.trajectory, self.top_map, self.force_map, self.lfdump)
        for a in self.atoms:
            a.universe = self

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

    def cache(self, directory="cg_cache"):
        """This precomputes the trajectory and structure so that it doesn't need to be 
           recalculated at each timestep. Especially useful for random access.
           Returns a Universe object corresponding to the cached trajectory
        """

        if(not os.path.exists(directory)):
            os.mkdir(directory)
        structure = os.path.join(directory, "cg.pdb")
        trajectory = os.path.join(directory, "cg.trr")
        write_structure(self, structure, bonds='all')
        write_trajectory(self, trajectory)        
        u = Universe(structure, trajectory)
        u.trajectory.periodic = self.trajectory.periodic
        apply_mass_map(u, create_mass_map(self))
        return u
    

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

    def __init__(self, universe, aatraj, top_map, force_map, lfdump):

        self.u = universe
        self.aatraj = aatraj

        self.top_map = top_map
        self.force_map = force_map
        
        self.lfdump = lfdump

        self.units = aatraj.units
        self.numatoms = np.shape( top_map )[0] #The topology matrix mapping should have a number of rows equal to cg atoms
        self.periodic = aatraj.periodic
        self.numframes = aatraj.numframes
        self.skip  = aatraj.skip
        self.ts = Timestep(self.numatoms)
        self.ts.set_ref_traj(aatraj)
        self.ts.dimensions = aatraj.ts.dimensions
        self.ts._pos = np.empty((self.numatoms,3), dtype=np.float32)
        self.ts._velocities = np.copy(self.ts._pos)
        self.ts._forces = np.copy(self.ts._pos)
        self._read_next_timestep(ts=aatraj.ts)
        
    def close(self):
        self.aatraj.close()
    
    def __del__(self):
        self.close()
    
    def __iter__(self):
        def iterCG():
            for i in range(0, self.numframes, self.skip):
                try: yield self._read_next_timestep()
                except IOError: raise StopIteration
        return iterCG()

    def _read_next_timestep(self, ts=None):
        if(ts is None):
            ts = self.aatraj.next()
        self.ts.frame = ts.frame        

        #now, we must painstakingly and annoyingly put each cg group into the same periodic image
        #as its residue 
        if(self.aatraj.periodic):

            for r in self.u.ref_u.residues:
                centering_vector = np.copy(r.atoms[0].pos)
                for a in r.atoms:
                    a.pos[:] =  same_img(a.pos[:], centering_vector, ts.dimensions)
                                            
        self.ts._pos = self.top_map.dot( ts._pos )
        try:
            self.ts._velocities[:] = self.top_map.dot( ts._velocities ) #COM motion
        except AttributeError:
            self.ts._velocities[:] = np.zeros( (self.numatoms, 3) ,dtype=np.float32)

        #check to see if we have a lammps force dump for forces
        if(self.lfdump):
            #we do, let's read it
            forces = np.zeros( (np.shape(self.top_map)[1], 3), dtype=np.float32)
            while(not self.lfdump.readline().startswith('ITEM: ATOMS')):
                pass
            for i in range(len(forces)):
                sline = self.lfdump.readline().split()
                    #NOTE NOTE NOTE NOTE: Lammps forces seem to be negative of what I use.
                try:
                    forces[int(sline[0]) - 1,:] = [-float(x) for x in sline[1:]]                        
                except ValueError:
                    raise IOError( "Invalid forces line at %s" % reduce(lambda x,y: x + " " + y, sline))
            self.ts._forces[:] = self.force_map.dot( forces) 
        else:
            try:
                self.ts._forces[:] = self.force_map.dot( ts._forces )
            except AttributeError:
                #can't find any forces, use 0
                self.ts._forces[:] = np.zeros( (self.numatoms, 3) ,dtype=np.float32)

        return self.ts

    def rewind(self):
        self.aatraj.rewind()

#END CGUniverse Stuff

def write_structure(universe, filename, **args):
    """A symmetric version of the write_trajectory method
    """
    universe.atoms.write(filename, **args)


def write_trajectory(universe, filename):
    """A simplified method for writing trajectories
    """
    w = Writer(filename, universe.atoms.numberOfAtoms())
    for ts in universe.trajectory:
        w.write(ts)
    w.close()

def create_mass_map(universe):
    """Create a map of masses for atom types in a universe so that they can be applied to a universe without masses assigned
       to atoms. This is useful if writing a universe with a format that doesn't include masses (gro, pdb).
    """
    mass_map = {}
    for a in universe.atoms:
        if(not a.type in mass_map):
            mass_map[a.type] = a.mass
    return mass_map

def apply_mass_map(universe, mass_map):
    """Apply a mass_map created by a call to `create_mass_map`
    """
    for k,v in mass_map.iteritems():
        universe.selectAtoms('type {}'.format(k)).set_mass(v)


def add_residue_bonds(universe, selection1, selection2):
    """This function will add bonds between atoms mathcing selections and 2
    within any residue
    """
    count = 0
    for r in universe.atoms.residues:        
        group1 = r.selectAtoms(selection1)
        group2 = r.selectAtoms(selection2)            
        for a1 in group1:
            for a2 in group2:
                if(a1 != a2):
                    universe.bonds.add( Bond(a1, a2) )
                    count += 1
    print "Added %d bonds" % count
            

def add_sequential_bonds(universe, selection1=None, selection2=None):
    """this function will add bonds between sequential residues. The atoms within the residue
       chosen for the bonding are simply the first and last if no selections are given
    """
    count = 0
    if((selection1 or selection2) and not (selection1 and selection2)):
        raise ValueError("Must give 2 selections for two connecting sites")
    for s in universe.atoms.segments:
        last = None
        for r in s.atoms.residues:
            if(last is not None):
                if(selection1):
                    try:
                        universe.bonds.add( Bond(r.selectAtoms(selection1)[0], last) )
                    except IndexError:
                        raise ValueError("Could not find {} in {}".format(selection1, r))
                else:
                    universe.bonds.add( Bond(r.atoms[0], last) )
                count += 1
            if(selection2):
                last = r.atoms.selectAtoms(selection2)[-1]
            else:
                last = r.atoms[-1]

    print "Added %d bonds" % count
