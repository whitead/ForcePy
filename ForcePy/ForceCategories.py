from ForcePy.NeighborList import NeighborList
import numpy as np
from ForcePy.Util import norm3, min_img_vec

class ForceCategory(object):
    """A category of force/potential type.
    
    The forces used in force matching are broken into categories, where
    the sum of each category of forces is what's matched in the force
    matching code. Examples of categories are pairwise forces,
   threebody forces, topology forces (bonds, angles, etc).
   """

    def __init__(self):
        self.nlist_ready = False

    def generate_nlist(self, i):
        assert self.nlist_ready, "Neighbor list not built yet"
        nlist_accum = np.sum(self.nlist_lengths[:i]) if i > 0  else 0
        for j in self.nlist[nlist_accum:(nlist_accum + self.nlist_lengths[i])]:
            yield j

    def generate_neighbor_vecs(self, i, u, mask = None):
        positions = u.atoms.get_positions()
        dims = u.trajectory.ts.dimensions

        for j in self.generate_nlist(i):
            if(mask and not mask[j]):
                continue
            r = min_img_vec(positions[j], positions[i], dims, u.trajectory.periodic)
            d = norm3(r)
            r = r / d
            yield (r,d,j)

        

class Angle(ForceCategory):
    pass

class Dihedral(ForceCategory):
    pass

class Improper(ForceCategory):
    pass

class Pairwise(ForceCategory):
    """Pairwise force category. It handles constructing a neighbor-list at each time-step. 
    """
    instance = None

    @staticmethod
    def get_instance(*args):
        
        if(len(args) == 0 or args[0] is None):
            #doesn't care about cutoff
            return Pairwise.instance

        if(Pairwise.instance is None):
            Pairwise.instance = Pairwise(args[0])
        else:
            #check cutoff
            if(Pairwise.instance.cutoff - args[0] > 0.00001):
                raise RuntimeError("Incompatible cutoffs: Already set to %g, not %g" % (Pairwise.instance.cutoff,args[0]))
        return Pairwise.instance
    
    def __init__(self, cutoff=12):
        super(Pairwise, self).__init__()
        self.cutoff = cutoff                    
        self.forces = []
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

    def pair_exists(self, u, type1, type2):
        return True

    def __reduce__(self):
        return Pairwise, (self.cutoff,)
    
class Bond(ForceCategory):

    """Bond category. It caches each atoms bonded neighbors when constructued
    """
    instance = None

    @staticmethod
    def get_instance(*args):        
        if(Bond.instance is None):
            Bond.instance = Bond()
        return Bond.instance
    
    def __init__(self):
        super(Bond, self).__init__()
    

    def _build_nlist(self, u):
        temp = [[] for x in range(u.atoms.numberOfAtoms())]
        #could be at most everything bonded with everything
        self.nlist = np.empty((u.atoms.numberOfAtoms() - 1) * (u.atoms.numberOfAtoms() / 2), dtype=np.int32)
        self.nlist_lengths = np.empty(u.atoms.numberOfAtoms(), dtype=np.int32)
        nlist_accum = 0
        for b in u.bonds:
            temp[b.atom1.number].append(b.atom2.number)
            temp[b.atom2.number].append(b.atom1.number)

        #unwrap the bond list to make it look like neighbor lists
        for i,bl in zip(range(u.atoms.numberOfAtoms()), temp):
            self.nlist_lengths[i] = len(temp[i])
            for b in bl:
                self.nlist[nlist_accum] = b
                nlist_accum += 1

        #resize now we know how many bond items there are
        self.nlist = self.nlist[:nlist_accum]
        self.nlist_ready = True

    def _setup(self, u):
        if(not self.nlist_ready):
            self._build_nlist(u)

    def _teardown(self):
        self.nlist_ready = False
        
    def pair_exists(self, u, type1, type2):
        """Check to see if a there exist any pairs of the two types given
        """
        if(not self.nlist_ready):
            self._build_nlist(u)        

        sel2 = u.atoms.selectAtoms(type2)        
        for a in u.atoms.selectAtoms(type1):
            i = a.number
            nlist_accum = np.sum(self.nlist_lengths[:i]) if i > 0  else 0
            for j in self.nlist[nlist_accum:(nlist_accum + self.nlist_lengths[i])]:
                if(u.atoms[int(j)] in sel2):
                    return True

        return False

