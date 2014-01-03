import random

class State_Mask(object):
    """This object generates a mask on the fly for a particle given
    its state function acting on a MDAnalysis Universe"""

    def __init__(self, cgtofg_fiber_map, state_function, state_index):
        self.fxn = state_function
        self.map = cgtofg_fiber_map
        self.state = state_index
        
    
    def __getitem__(self, index):
        ag = self.map[index]

        if(ag.state is None):            
            x = random.random()
            vsum = 0
            for i,v in enumerate(self.fxn(ag.ag)):
                vsum += v
                if(x <= vsum):
                    break
            ag.state = i
        
        return ag.state == self.state
