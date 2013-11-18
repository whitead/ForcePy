from ForcePy.ForceCategories import Pairwise, Bond, Angle, Dihedral
from ForcePy.States import State_Mask
from MDAnalysis import Universe
import numpy as np
from math import floor, pi


class Analysis(object):
    """ An abstract class that is used for analysis classes. For
        example, a radial distribution function calculation.  The
        methods in the class are for specializing the analysis to
        certain types/states and interfacing with neighborlists for
        pairs or bonds.
    """
    def __init__(self, category, period, outfile, cutoff=None):
        self.category = category.get_instance(cutoff)
        self.period = period
        self.outfile = outfile
        self.update_counts = 0
        self.sel1 = None
        self.sel2 = None
        self.mask1 = None
        self.mask2 = None

    def get_category(self):
        return self.category

    def update(self, u):
        if(self.update_counts % self.period == 0):
            self.do_update(u)
        self.update_counts += 1
        
    #In case the force needs access to the universe for setting up, override (and call this method).
    def setup_hook(self, u):
        try:
            self._build_mask(self.sel1, self.sel2, u)
        except AttributeError:
            pass #some forces don't have selections, ie FileForce

    def specialize_types(self, selection_pair_1 = None, selection_pair_2 = None):
        self.sel1 = selection_pair_1
        self.sel2 = selection_pair_2
        self.type_name = "[%s] -- [%s]" % (selection_pair_1, selection_pair_2)

    def specialize_states(self, mask1, mask2, name1 = None, name2 = None):
        self.mask1 = mask1
        self.mask2 = mask2
        self.type_name = "[state %s] -- [state %s]" % (name1, name2)


    def _build_mask(self, sel1, sel2, u):
        if(self.mask1 is None and sel1 is not None):
            self.mask1 = [False for x in range(u.atoms.numberOfAtoms())]
            for a in u.selectAtoms('type %s' % sel1):
                self.mask1[a.number] = True
        elif(self.mask1 is None):
            self.mask1 = [True for x in range(u.atoms.numberOfAtoms())]

            
        if(self.mask2 is None and sel2 is not None):
            self.mask2 = [False for x in range(u.atoms.numberOfAtoms())]
            for a in u.selectAtoms('type %s' % sel2):
                self.mask2[a.number] = True
        elif(self.mask2 is None):
            self.mask2 = self.mask1
        
    def valid_pair(self, atom1, atom2):
        """Checks the two atoms' types to see if they match the type
           specialization. If no type selections are set, returns true
        """
        #Don't use the selection class since it's a little heavy for this
        import re
        if(type(atom1) != type("")):
            atom1 = atom1.type
        if(type(atom2) != type("")):
            atom2 = atom2.type
        
        try:
            if(re.match(self.sel1, atom1) is not None and re.match(self.sel2, atom2) is not None):
                return True
            if(re.match(self.sel2, atom1) is not None and re.match(self.sel1, atom2) is not None):
                return True
        except AttributeError:
            return True

        return False


class RDF(Analysis):
    def __init__(self, category, period, outfile, binsize, cutoff):
        super(RDF, self).__init__(category, period, outfile, cutoff)
        self.hist = [0 for x in np.arange(0,cutoff, binsize)]
        self.binsize = binsize
        self.cutoff = cutoff

    
    def do_update(self, u):
        for i in range(u.atoms.numberOfAtoms()):
            #check to if this is a valid type
            if(self.mask1[i]):
                maskj = self.mask2
            elif(self.mask2[i]):
                maskj = self.mask1
            else:
                continue
            for r,d,j in self.category.generate_neighbor_vecs(i, u, maskj):
                if(i < j):
                    b = int(floor(d / self.binsize))
                    self.hist[b] += 1

    def write(self):        
        
        if(type(self.outfile) != file):
            self.outfile = open(self.outfile, 'w')

        self.outfile.seek(0)

        N = sum(self.hist)
        density = N / (4/3. * pi * self.cutoff ** 3)
        
        for rl,rr,h in zip(np.arange(0,self.cutoff, self.binsize), 
                           np.arange(self.binsize,self.cutoff + self.binsize, self.binsize), 
                           self.hist):
            r = 0.5 * (rl + rr)
            gr = (3 * h / (density * 4 * pi * (rr**3 - rl**3)))
            self.outfile.write('{} {}\n'.format(r, gr))

        
