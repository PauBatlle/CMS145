import numpy as np 
import math 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from tqdm import tqdm 
from scipy.spatial import KDTree
from numpy import random
from scipy.stats import entropy

#TODO: 
"""
Main issues: 
- Value iteration does not involve iteration? ***A: current implementation is just straight backwards induction 
- No way to verify solution as of yet- should create some visualizations
- Need to extract policy from the value matrix 
- Implemented Experiments are currently just a node picked to measure, not a matrix as in the .ipynb 
"""

# HELPER FUNCTIONS
#TODO: Make them part of classes
def sample_simplex(dimension, n_points):
    return np.random.dirichlet([1]*dimension, size = n_points)


def dKL(posterior, prior):
    return entropy(posterior, prior, base = 2)

def Random_walk_dynamics(G):
    aux = np.array(nx.adjacency_matrix(G).todense())
    return aux/aux.sum(axis=1, keepdims=True)
# END HELPER FUNCTIONS 

class sOED:
    def __init__(self,N,L,T,p,RW,experiments,S = 50,samples = None):
        """
        Initializes a sequential optimal experiment design with the following variables: 
        N: number of states (nodes)
        L: number of value/policy updates
        T: number of experiments (discrete)
        p: prior beliefs over states
        S: number of belief states in discretization
        """
        # I think we could make the variable names more expressive, otherwise I find it hard to keep track
    
        self.N = N
        self.T = T
        self.prior = p
        self.S = S
        self.possible_experiments = experiments #A list of matrices of possible experimetns
        #Implicitly assuming for now that the same experiments are available at all timesteps 
        self.values = random.rand(S,T)
        if samples is None:
            self.samples = sample_simplex(self.N,S)
        else: self.samples = samples
        self.RW = RW # A RW object passed in to sOED
        self.NNTree = KDTree(data= self.samples)

    def reward(self,k, xk): 
        '''
        *** for now we are assuming 0 stage reward so the reward does not depend on the observation directly
        prior: vector over m states
        outputs: scalar indicating reward from belief xk at experiment k
        xk is belief at experiment k
        '''
        if k < self.T-1:
            return 0
        #
        #Ptest = experiment@prior
        #Posterior = np.divide(np.multiply(experiment, prior),Ptest.reshape(-1,1))
        return dKL(xk,self.prior)
    

    def posterior(self,prior,i,yi): 
        '''
        Computes posterior belief given the prior belief, the experiment i performed, and the observation number yi \in {0, ..., |S|-1}
        '''
        experiment = self.possible_experiments[i] #|S|x|Θ| matrix
        signals, N_world_states = experiment.nsignals, experiment.nstates
        Ptest = (experiment.M)@prior
        Posterior = np.divide(np.multiply(experiment.M, prior),Ptest.reshape(-1,1))
        return Posterior[yi] 

    def propagate_dynamics(self,posterior):
        '''
        given a posterior belief (after observation), propagates belief using the RW dynamics 
        '''
        M = self.RW.get_M()
        F = (M.T)@posterior
        return F

    def get_NN_index(self,dist):
        '''
        Takes probability distribution dist and returns the index of self.samples that is closest
        '''
        return self.NNTree.query(dist)[1]

    def value_iter(self): 
        '''
        Performs value iteration to learn optimal values at k^th experiment given belief xk
        '''
        curr_vals = self.values
        best_policy = np.zeros((self.S,self.T)) # keeps track of the best measurements

        for k in tqdm(reversed(range(self.T)), total = self.T):
            for s in range(self.S): # Iterating over the sampled belief states 
                max_di = None
                max_val = 0 # max value over all possible states to measure here
                sample = self.samples[s]
                for i in range(self.N): # Iterating over all possible experiments we could perform 
                    #In the future, generalize this to more than two outcomes                  
                    post_1 = self.posterior(sample,i,1) #We probably don't have to do this calculation for all times?
                    post_0 = self.posterior(sample,i,0)
                    exp_val = None
                    if k == self.T - 1:
                        exp_val = sample[i]*(self.reward(k,post_1)) + (1-sample[i])*(self.reward(k,post_0))
                    else:
                        # Find NN of the posteriors to pair correct future value with each posterior
                        NN_1 = self.get_NN_index(self.propagate_dynamics(post_1))
                        NN_0 = self.get_NN_index(self.propagate_dynamics(post_0))
                        # print('i: ', i)
                        # print('k: ', k)
                        # print('NN_1: ', NN_1)
                        # print('NN_0: ', NN_0)
                        exp_val = sample[i]*(self.reward(k,post_1)+curr_vals[NN_1,k+1]) + (1-sample[i])*(self.reward(k,post_0)+curr_vals[NN_0,k+1])
                    if exp_val> max_val:
                        max_val = exp_val
                        max_di = i
                # Update values
                curr_vals[s,k] = max_val
                best_policy[s,k] = max_di
        self.values = curr_vals
        return curr_vals, best_policy
        
