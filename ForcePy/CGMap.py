from MDAnalysis import Writer
from MDAnalysis.core.AtomGroup import Universe, AtomGroup, Atom, Residue
from MDAnalysis.topology.core import Bond
import MDAnalysis.coordinates.base as base
import numpy as np
from ForceMatch import min_img_dist



#note to self, build atoms, residues, segs add to AtomGroup and then
# call build_cache. Can also call set position on the
# atomgroup. Initializing a residue correctly sets the residue type of
# atoms.


#TODO: Rewrite this as a subclass of Universe. That would be much more elegant.

class CGUniverse(Universe):
    """ Class which uses center of mass mappings to reduce the number
        of degrees of freedom in a given trajectory/structure
        file. The selections define how atoms are grouped PER
        residue. Thus, if there is only one residue defined, then all
        atoms in that selection will be one bead in the CG system.
    """

    def __init__(self, otherUniverse, selections, names = None, collapse_hydrogens = True):
        if(names is None):
            names = [ "%d" % x for x in range(0, len(selections))]
        if(len(names) != len(selections)):
            raise ValueError("Length of slections (%d) must match lenght of names (%d)" % (len(selections), len(names)))
        self.ref_u = otherUniverse
        self.atoms = AtomGroup([])
        self.selections = selections
        self.names = names
        self.chydrogens = collapse_hydrogens
        self.universe = self
        self._build_structure()
        

    def _build_structure(self):
        index = 0
        reverse_map = {}
        residues = {}
        for s,n in zip(self.selections, self.names):
            group = self.ref_u.selectAtoms(s)
            if(group.numberOfAtoms() == 0):
                raise ValueError("Selection '%s' matched no atoms" % s)        
            for r in group.residues:
                a = Atom(index, n, n, r.name, r.id, r.atoms[0].segid, sum([x.mass if x in group else 0 for x in r]), 0) 
                index += 1
                for ra in r.atoms:
                    if ra in group:
                        reverse_map[ra] = a

                #check to see if this residue exists yet
                #and add the atom
                if( r.id in residues ):
                    residues[r.id] += a
                else:
                    residues[r.id] = Residue(r.name, r.id, [a], resnum=r.resnum)
                a.residue = residues[r.id]
                #add the atom
                self.atoms += a
        

        #find hydrogens and collapse them into beads 
        if(self.chydrogens):
            for b in self.ref_u.bonds:
                #my hack for inferring a hydrogen
                for a1,a2 in zip((b.atom1, b.atom2), (b.atom2, b.atom1)):
                    if(a1.type.startswith("H") and a1.mass < 4.):
                        reverse_map[a1] = reverse_map[a2]
                        #add the mass
                        reverse_map[a1].mass += a1.mass
        
        #generate matrix mappings for center of mass and sum of forces
        # A row is a mass normalized cg site defition. or unormalized 1s for forces
        self.top_map = np.zeros( (self.atoms.numberOfAtoms(), self.ref_u.atoms.numberOfAtoms()) , dtype=np.float32)
        self.force_map = np.zeros( (self.atoms.numberOfAtoms(), self.ref_u.atoms.numberOfAtoms()) , dtype=np.float32)

        for a in self.ref_u.atoms:
            self.top_map[reverse_map[a].number, a.number] = a.mass / reverse_map[a].mass
            self.force_map[reverse_map[a].number, a.number] = 1.
                                    
        #add bonds using the reverse map
        self.bonds = set()
        for b in self.ref_u.bonds:
            cgatom1 = reverse_map[b.atom1]
            cgatom2 = reverse_map[b.atom2]
            for cbg in self.bonds:
                if(not (cbg.atom1 in [cgatom1, cgatom2]) and not( cbg.atom2 in [cgatom1, cgatom2])):
                    #OK, no bond exists yet
                    self.bonds.add( Bond(cgatom1, cgatom2) )

        self.__trajectory = CGReader(self.ref_u.trajectory, self.top_map, self.force_map)
        for a in self.atoms:
            a.universe = self
        self.atoms._rebuild_caches()
    
    @property
    def trajectory(self):
        return self.__trajectory

    def write_structure(self, filename, *args):
        self.atoms.write(filename, *args)


    def write_trajectory(self, filename):
        w = Writer(filename, self.atoms.numberOfAtoms())
        for ts in self.trajectory:
            w.write(ts)
        w.close()

    def add_residue_bonds(self, selection1, selection2):
        """This function will add bonds between atoms mathcing selections and 2
           within any residue
        """

        for r in self.atoms.residues:
            group1 = r.selectAtoms(selection1)
            group2 = r.selectAtoms(selection2)            
            for a1 in group1:
                for a2 in group2:
                    self.bonds.add( Bond(a1, a2) )

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

    _Timestep = Timestep
    
    def __init__(self, aatraj, top_map, force_map):
        
        self.aatraj = aatraj

        self.top_map = top_map
        self.force_map = force_map

        self.units = aatraj.units
        self.numatoms = np.shape( top_map )[0] #The topology matrix mapping should have a number of rows equal to cg atoms
        self.periodic = aatraj.periodic
        self.numframes = aatraj.numframes
        self.skip  = aatraj.skip
        self.ts = Timestep(self.numatoms)
        self.ts.set_ref_traj(aatraj)
        self.ts.dimensions = aatraj.ts.dimensions
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
        #collapse them, and then unwrap them
        if(self.aatraj.periodic):

            dim = shape(ts._pos)[0]
            self.ts_pos = np.zeros( shape(self.top_map)[0], dim, dtype=np.float32)
            self.ts_velocities = np.zeros( shape(self.top_map)[0], dim, dtype=np.float32)
            self.ts_forces = np.zeros( shape(self.top_map)[0], dim, dtype=np.float32)
            
            centering_vector = np.zeros(dim)
            for cgi in range(shape(self.top_map)[0]):
                #get min image coordinate average
                for aai in range(shape(self.top_map)[1]):
                    mat[:,cgi] += self.top_map[cgi,aai] * min_img_dist(ts._pos[:,aai], centering_vector, ts.dimensions)
                #make min image
                mat[:,cgi] = min_img(mat[:,cgi], ts.dimensions)
            #same for velocites, but we might not have them so use try/except
            try:
                for cgi in range(shape(self.force_map)[0]):
                    #get min image coordinate average
                    for aai in range(shape(self.force_map)[1]):
                        mat[:,cgi] += self.force_map[cgi,aai] * min_img_dist(ts._velocities[:,aai], centering_vector, ts.dimensions)
                        #make min image
                        mat[:,cgi] = min_img(mat[:,cgi], ts.dimensions)
            except AttributeError:
                pass
            #same for forces
            try:
                for cgi in range(shape(self.force_map)[0]):
                    #get min image coordinate average
                    for aai in range(shape(self.force_map)[1]):
                        mat[:,cgi] += self.force_map[cgi,aai] * min_img_dist(ts_forces[:,aai], centering_vector, ts.dimensions)
                        #make min image
                        mat[:,cgi] = min_img(mat[:,cgi], ts.dimensions)
            except AttributeError:
                pass

        else:
            #SUPER EASY if not periodic!
            self.ts._pos = self.top_map.dot( ts._pos )
            try:
                self.ts._velocities = self.top_map.dot( ts._velocities ) #COM motion
            except AttributeError:
                self.ts._velocities = np.zeros( (self.numatoms, 3) ,dtype=np.float32)
            try:
                self.ts._forces = self.force_map.dot( ts._forces )
            except AttributeError:            
                self.ts._forces = np.zeros( (self.numatoms, 3) ,dtype=np.float32)

        return self.ts

    def rewind(self):
        self.aatraj.rewind()
            
