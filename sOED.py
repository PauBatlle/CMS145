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

def F(prior, posterior, betas = None):
    """
    Function F associated with Cost of Information
    """
    N = prior.shape[0]
    if betas is None: 
        betas = np.ones((N,N))
    
    return sum([betas[i,j]*posterior[i]/prior[i]*np.log(posterior[i]/prior[j]) for i in range(N) for j in range(N)])

#TODO: write tests for F and CoI
def CoI(prior,posteriors, post_probs, betas = None): 
    """
    Computes the cost of information
    prior: prior belief
    posteriors: list of possible posteriors that experiment would lead to
    post_probs: probability of getting each posterior
    """

    N = prior.shape[0]
    return sum([(F(posteriors[i],prior)-F(prior,prior))*post_probs[i] for i in range(N)])

def Random_walk_dynamics(G):
    aux = np.array(nx.adjacency_matrix(G).todense())
    return aux/aux.sum(axis=1, keepdims=True)
# END HELPER FUNCTIONS 

class sOED:
    def __init__(self,N,T,p,RW,experiments,S,samples = None):
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

    def reward(self,k, xk, alpha = 0, cost = 0, y=None, bonus = 0): 
        '''
        *** for now we are assuming 0 stage reward so the reward does not depend on the observation directly
        prior: vector over m states
        outputs: scalar indicating reward from belief xk at experiment k
        xk is belief at experiment k
        alpha: weight between 0 and 1 of how much to put value on intermediate reward

        '''
        if cost == 0: # Regular DKL reward
            if k < self.T-1:
                return alpha*dKL(xk,self.prior)
        
            #
            #Ptest = experiment@prior
            #Posterior = np.divide(np.multiply(experiment, prior),Ptest.reshape(-1,1))
            return dKL(xk,self.prior)
    
        if cost == 1: # Cost of information
            if k< self.T -1: 
                pass
        
        if cost == 2: # Reward for actually finding it
            find  = 0
            if y ==1 : find = bonus
            if k < self.T-1:
                return alpha*dKL(xk,self.prior) + bonus
            else: return dKL(xk,self.prior) + bonus

    def posterior(self,prior,i,yi): 
        '''
        Computes posterior belief given the prior belief, the experiment i performed, and the observation number yi \in {0,1}
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

    def value_iter(self, alpha = 0, cost = 0, bonus = 0): 
        '''
        Performs value iteration to learn optimal values at k^th experiment given belief xk
        '''
        curr_vals = self.values
        best_policy = np.zeros((self.S,self.T)) # keeps track of the best measurements
        all_vals = np.zeros((self.S,self.T,self.N))
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
                        exp_val = sample[i]*(self.reward(k,post_1,alpha = alpha)) + (1-sample[i])*(self.reward(k,post_0,alpha = alpha))
                    else:
                        # Find NN of the posteriors to pair correct future value with each posterior
                        NN_1 = self.get_NN_index(self.propagate_dynamics(post_1))
                        NN_0 = self.get_NN_index(self.propagate_dynamics(post_0))
                        # print('i: ', i)
                        # print('k: ', k)
                        # print('NN_1: ', NN_1)
                        # print('NN_0: ', NN_0)
                        exp_val = sample[i]*(self.reward(k,post_1, alpha = alpha, cost = cost, y = 1, bonus = bonus)+curr_vals[NN_1,k+1]) + (1-sample[i])*(self.reward(k,post_0, alpha = alpha, cost = cost,y=0, bonus = bonus)+curr_vals[NN_0,k+1])
                    all_vals[s,k,i] = exp_val
                    if exp_val> max_val:
                        max_val = exp_val
                        max_di = i
                # Update values
                curr_vals[s,k] = max_val
                best_policy[s,k] = max_di
        self.values = curr_vals
        return curr_vals, best_policy, all_vals
        
