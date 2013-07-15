from ForcePy.NeighborList import NeighborList
import numpy as np

class ForceCategory(object):
    """A category of force/potential type.
    
    The forces used in force matching are broken into categories, where
    the sum of each category of forces is what's matched in the force
    matching code. Examples of categories are pairwise forces,
   threebody forces, topology forces (bonds, angles, etc).
   """
    pass



class Pairwise(ForceCategory):
    """Pairwise force category. It handles constructing a neighbor-list at each time-step. 
    """
    instance = None

    @staticmethod
    def get_instance(*args):        
        if(Pairwise.instance is None):
            Pairwise.instance = Pairwise(args[0])
        else:
            #check cutoff
            if(Pairwise.instance.cutoff != args[0]):
                raise RuntimeError("Incompatible cutoffs")
        return Pairwise.instance
    
    def __init__(self, cutoff=12):
        super(Pairwise, self).__init__()
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

    def pair_exists(self, u, type1, type2):
        return True

    
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
        self.blist_ready = False
    

    def _build_blist(self, u):
        temp = [[] for x in range(u.atoms.numberOfAtoms())]
        #could be at most everything bonded with everything
        self.blist = np.empty((u.atoms.numberOfAtoms() - 1) * (u.atoms.numberOfAtoms() / 2), dtype=np.int32)
        self.blist_lengths = np.empty(u.atoms.numberOfAtoms(), dtype=np.int32)
        blist_accum = 0
        for b in u.bonds:
            temp[b.atom1.number].append(b.atom2.number)
            temp[b.atom2.number].append(b.atom1.number)

        #unwrap the bond list to make it look like neighbor lists
        for i,bl in zip(range(u.atoms.numberOfAtoms()), temp):
            self.blist_lengths[i] = len(temp[i])
            for b in bl:
                self.blist[blist_accum] = b
                blist_accum += 1

        #resize now we know how many bond items there are
        self.blist = self.blist[:blist_accum]
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
    def nlist(self):
        return self.blist

    @property
    def nlist_lengths(self):
        return self.blist_lengths

    def pair_exists(self, u, type1, type2):
        """Check to see if a there exist any pairs of the two types given
        """
        if(not self.blist_ready):
            self._build_blist(u)        

        sel2 = u.atoms.selectAtoms(type2)        
        for a in u.atoms.selectAtoms(type1):
            i = a.number
            blist_accum = np.sum(self.blist_lengths[:i]) if i > 0  else 0
            for j in self.blist[blist_accum:(blist_accum + self.blist_lengths[i])]:
                if(u.atoms[int(j)] in sel2):
                    return True

        return False

