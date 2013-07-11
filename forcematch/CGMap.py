import MDAnalysis
from MDAnalysis.core.AtomGroup import Universe, AtomGroup, Atom, Residue
from MDAnalysis.coordinates.base import Timestep, Reader
import numpy as np



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
                a = Atom(index, n, n, r.name, r.id, r.atoms[0].segid, sum([x.mass for x in r]), 0) 
                index += 1
                for ra in r.atoms:
                    reverse_map[ra] = a

                #check to see if this residue exists yet
                #and add the atom
                if( r.id in residues ):
                    residues[r.id] += a
                else:
                    residues[r.id] = Residue(r.name, r.id, [a], resnum=r.resnum)
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
        self.top_map = np.zeros( (self.atoms.numberOfAtoms(), self.ref_u.atoms.numberOfAtoms()) )
        self.force_map = np.zeros( (self.atoms.numberOfAtoms(), self.ref_u.atoms.numberOfAtoms()) )

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

        #build cache of residues, resnames, etc
        self.atoms._rebuild_caches()
        
        self.trajectory = CGReader(self.ref_u.trajectory, self.top_map, self.force_map)
        for a in self.atoms:
            a.universe = self
            
        


#    def write_structure(self, sfile_out):
##        writer = MDAnalysis.coordinates.writer(sfile_out, multiframe=False)
##        writer.write(self.tar_u)
##        writer.close()
##
##        
##    def write_trajectory(self, tfile_out):
##        writer = MDAnalysis.Writer(tfile_out, self.tar_u.numberOfAtoms())
##        for ts in self.tar_u.universe.trajectory:
##            writer.write(ts)
#        writer.close()



class CGReader(Reader):
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
        try:
            if(ts is None):
                ts = self.aatraj.next()
        except EOFError:
            raise IOError
        self.ts._unitcell = ts._unitcell
        self.ts.frame = ts.frame        
        self.ts._pos = np.array(self.top_map.dot( ts._pos ), dtype=np.float32)
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
        
        

def main():
    cg = CGUniverse(Universe("../../../Downloads/2GHL.pdb"), ['all'], collapse_hydrogens=True)
    cg.atoms.write("foo.pdb", bonds="all")
    #cg.write_trajectory("foo.trr")
    
        
if __name__ == '__main__': main()

