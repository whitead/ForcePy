import numpy as np
import math
import scipy.io as sio
from ForceCategories import *
from MDAnalysis import Universe




class WeightFactor:
    '''this class holds the weighting factor that will be used when calculating the
       forcematching forces. The class will access a function which when given an
       index will return an appropriate weight (a float that will be 1.00 if no weighting is
       needed)
    '''

    
    
    def __init__(self, use_neighbor_dist = False, use_neighbor_vel_dot = False, max_dist = 100):
        #maybe include an additional arg for where the data is coming from? WIP
        '''when the weight objects are made they determine which factors will be used
           for weighting based on the boolean KWargs. Each factor is then added to a
           an array that will be itterated through for the weighting of a given index. 
           *In the case that no factor is selected the weight will always be returned
            as one and will not affect anything.
        '''

        global CUTOFF
        CUTOFF = max_dist
        self.factors = []
        self.use_neighbor_dist = use_neighbor_dist
        self.use_neighbor_vel_dot = use_neighbor_vel_dot
        self.neighbor_dist_data = np.genfromtxt('fm-swarm283-200-2-quartic-y-y/weighting_factor.txt')
        self.pairwise_cat = Pairwise.get_instance(max_dist) #max dist will be passed in as a value which is the maximum distance which a "nearest neighbor" could be

    def _get_nearest_dist(self, i, u):
        global CUTOFF
        #find a min dist here for first neighbor
        #start min at 1+max_dist - if loop doesnt run then we return "101"
        #get neighbor_dist_weight should handle 101 appropriatley then (no weight)
        min_dist = CUTOFF + 1
        for r,d,j in self.pairwise_cat.generate_neighbor_vecs(i, u):
            #r - vector
            #d - magnitude
            #j - the other particle
            #calc weight -Prof
            #track the distance
            if(d<min_dist):
                min_dist = d
        return min_dist
        
        
         

    def calc_weight(self, i, u):
        '''given an index this will return the appropriate weight for a particle
           by using each of the factors that are denoted to use
        '''
        weight = 1
        
        if(self.use_neighbor_dist):
            weight*=self.get_neighbor_dist_weight(i, u)
        if(self.use_neighbor_vel_dot):
            weight*=self.get_neighbor_vel_dot_weight(i, u)
        
        return weight


#    def add_neighbor_dist():
#        '''adds a factor that is based on the distance to the nearest neighbor'''
        
        

    def get_neighbor_dist_weight(self, i, u):
        global CUTOFF
        #get distance
        distance = self._get_nearest_dist(i, u)
#        print 'distance:', distance

        if(distance == CUTOFF + 1):
            weight = 1
        else:
            #interpolate the data betweeen 2 data points
            #use the fact that the data will be equally spaced to advantage
            #assuming that the data is of the form data[0] = list of lengths, data[1] = list of weights
            steps = len(self.neighbor_dist_data)
            #        print steps
            spread = self.neighbor_dist_data[-1][0] - self.neighbor_dist_data[0][0]
            #        print '\n ndd', self.neighbor_dist_data 
            #        print spread
            step_size = spread/steps
            #        print step_size
            table_min = self.neighbor_dist_data[0][0]
            #        print table_min
            true_index = (distance - table_min)/step_size
            #        print true_index
            bot = int(math.ceil(true_index))
            #        print bot
            top = int(math.floor(true_index))
            #        print top
            lower = self.neighbor_dist_data[bot][1]
            #        print lower
            upper = self.neighbor_dist_data[top][1]
            #        print upper
            weight = (true_index-bot)*(upper - lower) + lower
#        print 'weight:', weight 
        return weight
                    
        



            
        
