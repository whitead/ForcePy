from MDAnalysis import Writer
from MDAnalysis.core.AtomGroup import Universe, AtomGroup, Atom, Residue
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
    """

    def __init__(self, otherUniverse, selections, names = None, collapse_hydrogens = True, lammps_force_dump = None):
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
        for s in self.selections:
            selection_count[s] = 0
        for r in self.ref_u.residues:
            residue_atoms = []
            for s,n in zip(self.selections, self.names):
                group = r.selectAtoms(s)
                selection_count[s] += len(group)
                
                #make new atom
                a = Atom(index, n, n, r.name, r.id, r.atoms[0].segid, sum([x.mass if x in group else 0 for x in r]), 0) 
                index += 1
                for ra in group:
                    reverse_map[ra] = a

                #append atom to new residue atom group
                residue_atoms.append(a)
                #add the atom to Universe
                self.atoms += a
            #now actually create new residue and give atoms a reference to it
            residues[r.id] = Residue(r.name, r.id, residue_atoms, resnum=r.resnum)
            for a in residue_atoms:
                a.residue = residues[r.id]
        #check to make sure we selected something
        for s,count in zip(self.selections, selection_count):
            if(count == 0):
                raise ValueError("Selection '%s' matched no atoms" % s)        

        

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
        
        #Put them into efficient sparse matrix. Should probably test this more
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
        structure = os.path.join(directory, "cg.gro")
        trajectory = os.path.join(directory, "cg.trr")
        write_structure(self, structure, bonds='all')
        write_trajectory(self, trajectory)
        return Universe(structure, trajectory)
    

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
        self.ts._pos = None
        self.ts._velocities = None
        self.ts._forces = None
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
            self.ts._velocities = self.top_map.dot( ts._velocities ) #COM motion
        except AttributeError:
            self.ts._velocities = np.zeros( (self.numatoms, 3) ,dtype=np.float32)
        try:
            self.ts._forces = self.force_map.dot( ts._forces )
        except AttributeError:
            #check to see if we have a lammps force dump
            if(self.lfdump):
                #we do, let's read it
                ts._forces = np.zeros( (np.shape(self.top_map)[1], dim), dtype=np.float32)
                while(not self.lfdump.readline().startswith('ITEM: ATOMS')):
                    pass
                for i in range(len(ts._forces)):
                    sline = self.lfdump.readline().split()
                    #NOTE NOTE NOTE NOTE: Lammps forces seem to be negative of what I use.
                    try:
                        ts._forces[int(sline[0]) - 1,:] = [-float(x) for x in sline[1:]]
                    except ValueError:
                        raise IOError( "Invalid forces line at %s" % reduce(lambda x,y: x + " " + y, sline))
                self.ts._forces = self.force_map.dot( ts._forces) 

            else: #if not, then use 0
                self.ts._forces = np.zeros( (self.numatoms, 3) ,dtype=np.float32)

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

def add_residue_bonds(universe, selection1, selection2):
    """This function will add bonds between atoms mathcing selections and 2
    within any residue
    """
    count = 0
    for r in universe.atoms.residues:        
        print r.atoms
        group1 = r.selectAtoms(selection1)
        group2 = r.selectAtoms(selection2)            
        for a1 in group1:
            for a2 in group2:
                universe.bonds.append( Bond(a1, a2) )
                count += 1
    print "Added %d bonds" % count
            

def write_lammps_scripts(fm, universe=None, prefix='cg', folder = os.curdir, lammps_units="real", table_points=1000, lammps_input_file=None):
    """Using the given ForceMatch and Universe object, this will create a set of input files for Lammps.
    
    The function will create the given folder and put all files
    in it. Tables are generated for each type of force from the
    ForceMatch object, a datafile derived from the current
    timestep of this Universe object and an input script that
    loads the force fields. The given lammps input file will be appended 
    to the input script.
    """
    if(universe is None):
        universe = fm.u

    #before we change directories, we need to get the path of the lammps input file 
    if(lammps_input_file is not None):
        lammps_input_file = os.path.abspath(lammps_input_file)

    if(not os.path.exists(folder)):
        os.mkdir(folder)
    os.chdir(folder)
 
    #write force tables
    force_info = fm.write_lammps_tables('%s_force' % prefix, 
                                        force_conv = -0.278,
                                        energy_conv = 0.278,
                                        dist_conv = 1,
                                        points=table_points)

        #write data file

                   
    atom_section = []
    mass_section = []
    bond_section = []
    type_map = {}
    atom_types = 0
    positions = universe.atoms.get_positions()
    has_charges = False
    for a in universe.atoms:
        if(abs(a.charge) > 0):
            has_charges = True


        #determin sim type
    type_count = fm.get_force_type_count()
    sim_type = "atomic"
    if(type_count[ForceCategories.Bond] > 0):
        if(type_count[ForceCategories.Angle] > 0):
            if(has_charges):
                sim_type = "full"
            else:
                sim_type = "molecular"
        else:
            sim_type = "bond"
    elif(has_charges):
        sim_type = "charge"

        
    for i,a in zip(range(len(universe.atoms)), universe.atoms):
        assert i == a.number, "Atom indices are jumbled. Atom %d has number %d" % (i, a.number)
        if(sim_type == "full"):
            atom_section.append("%d %d %d %f %f %f %f\n" % (i+1, a.resid, 
                                                            fm.get_atom_type_index(a),
                                                            a.charge, 
                                                            positions[i,0],
                                                            positions[i,1],
                                                            positions[i,2]))                
        elif(sim_type == "molecular" or sim_type == "bond"):
            atom_section.append("%d %d %d %f %f %f\n" % (i+1, a.resid, 
                                                         fm.get_atom_type_index(a),
                                                         positions[i,0],
                                                         positions[i,1],
                                                         positions[i,2]))
        elif(sim_type == "charge"):
            atom_section.append("%d %d %f %f %f %f\n" % (i+1, fm.get_atom_type_index(a),
                                                         a.charge,
                                                         positions[i,0],
                                                         positions[i,1],
                                                         positions[i,2]))
        elif(sim_type == "atomic"):
            atom_section.append("%d %d %f %f %f\n" % (i+1, fm.get_atom_type_index(a),
                                                      positions[i,0],
                                                      positions[i,1],
                                                      positions[i,2]))





        if(not a.type in type_map):
            type_map[a.type] = atom_types
            atom_types += 1
            mass_section.append("%d %f\n" % (atom_types, a.mass))


    bindex = 1
    for b in universe.bonds:
        btype = fm.get_bond_type_index(b.atom1, b.atom2)
        if(btype is not None):
            bond_section.append("%d %d %d %d\n" % (bindex, btype,
                                                   b.atom1.number+1, b.atom2.number+1))
            bindex += 1
        
    angle_section = []
    dihedral_section = []
    improper_section = []
        
    with open('%s_fm.data' % prefix, 'w') as output:

        #make header
        output.write('Generated by ForcePy.CGMap.py\n\n')
        output.write('%d atoms\n' % len(universe.atoms))
        output.write('%d bonds\n' % (bindex - 1))
        output.write('%d angles\n' % 0)
        output.write('%d dihedrals\n' % 0)
        output.write('%d impropers\n\n' % 0)

        output.write("%d atom types\n" % (atom_types))
        output.write("%d bond types\n" % type_count[ForceCategories.Bond])
        output.write("%d angle types\n" % type_count[ForceCategories.Angle])
        output.write("%d dihedral types\n" % type_count[ForceCategories.Dihedral])
        output.write("%d improper types\n\n" % type_count[ForceCategories.Improper])

        output.write('%f %f xlo xhi\n' % (0,universe.trajectory.ts.dimensions[0]))
        output.write('%f %f ylo yhi\n' % (0,universe.trajectory.ts.dimensions[1]))
        output.write('%f %f zlo zhi\n\n' % (0,universe.trajectory.ts.dimensions[2]))

            #the rest of the sections
                
        output.write("Masses\n\n")
        output.write("".join(mass_section))
        output.write("\nAtoms\n\n")
        output.write("".join(atom_section))
        if(type_count[ForceCategories.Bond] > 0):
            output.write("\nBonds\n\n")
            output.write("".join(bond_section))
            if(type_count[ForceCategories.Angle] > 0):
                output.write("\nAngles\n\n")
                output.write("".join(angle_section))
                if(type_count[ForceCategories.Dihedrals] > 0):
                    output.write("\nDihedrals\n\n")
                    output.write("".join(dihedral_section))
                    if(type_count[ForceCategories.Impropers] > 0):
                        output.write("\nImpropers\n\n")
                        output.write("".join(improper_section))
            
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
    universe.atoms.write("%s_start.pdb" % prefix, bonds='all')
        
