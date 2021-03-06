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
def sample_simplex(dimension, n_points):
    return np.random.dirichlet([1]*dimension, size = n_points)


def dKL(prior, posterior):
    return entropy(posterior, prior, base = 2)

def Random_walk_dynamics(G):
    aux = np.array(nx.adjacency_matrix(G).todense())
    return aux/aux.sum(axis=1, keepdims=True)
# END HELPER FUNCTIONS 

class RandomWalk:
    def __init__(self,N,M=None,p = None, s= None):
        """
        Initializes a random walk with the following variables: 
            N: number of nodes
            M: transition matrix (if not entered, then generated randomly)
            p: prior distribution over nodes
            s: initial state (if not given, randomly chosen according to p)
        """
        self.N = N
        if M is None: 
            M = random.rand(N,N)
            A = random.randint(2,size = (N,N))
            M = np.multiply(M,A) # Elementwise multiplication: makes some of M zero
            M = M + np.diag(0.05*np.ones(N))
            sum_rows = M.sum(axis = 1)
            normed_M = M /sum_rows[:,np.newaxis]
            self.M = normed_M
        else: self.M = M

        if p is None: 
            p = random.rand(N)
            p = p/sum(p)
            self.p = p
        else: self.p = p 

        if s is None: 
            self.s = random.choice(N,1,p=self.p)
        else: self.s = s

        self.state = self.s # Current state
        self.t = 0 # Time step 
    
    def get_M(self):
        return self.M
    
    def update(self):
        """
        Take one step and update attributes
        """
        q = self.M[self.state,:]
        self.state = random.choice(self.N,1,p = q)
    
    def set_state(self,s):
        """
        Sets the state of the RW
        """
        self.state = s

    def observe(self,i):
        """
        Returns observation of state i
        """
    
        if i == self.state: return 1
        else: return 0
    
    # def get_phi(self,k,i,x_k):
    #     '''
    #     Takes in index k for experiment number, a function taking index k for the experiment number, i for the basis index, and current belief x_k to evaluate and returns the result of the 
    #             basis function phi_k,i(x_k)
    #     ********** Possibly should be in examples.py ****************
    #     '''
    #     ### Is this problem-specific? Should it be in the RandomWalk class? Yes
    #     pass

class sOED:
    def __init__(self,N,L,T,p,RW,S = 50):
        """
        Initializes a sequential optimal experiment design with the following variables: 
        N: number of states (nodes)
        get_phi: a function taking index k for the experiment number, i for the basis index, and current belief x_k to evaluate and returns the result of the 
        basis function phi_k,i(x_k)
        L: number of value/policy updates
        T: number of experiments (discrete)
        p: prior beliefs over states
        pi_explore: exploration policy 
        S: number of belief states in discretization
        """
    
        self.N = N
        self.L = L
        self.T = T
        self.prior = p
        self.S = S
        self.values = random.rand(S,T)
        self.samples = sample_simplex(self.N,S)
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

        # Ptest = experiment@prior
        # Posterior = np.divide(np.multiply(experiment, prior),Ptest.reshape(-1,1))
        return dKL(self.prior,xk)
    
    def posterior(self,prior,i,yi): 
        '''
        Computes posterior belief given the prior belief, the state i measured, and the observation yi
        '''
        epsilon = 0.0001
        if yi == 1: 
            p = np.ones(self.N)*epsilon
            p[i] = 1
            return p/np.linalg.norm(p,1)
        else: 
            #TODO?
            # How to update to posterior? Rn just zeroing out zero observation and renormalizing
            p = prior.copy()
            p[i] = 0
            p = p/np.linalg.norm(p,1)
            return p

    def propagate_belief(self,posterior):
        '''
        given a posterior belief (after observation), propagates belief using the RW dynamics 
        '''
        M = self.RW.get_M()
        F = M.T@posterior
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
        #TODO: why don't we need to iterate in rounds? what am I interpreting wrong here? seems deterministic given samples and prior
        # for l in range(L): # Number of rounds to perform
        curr_vals = self.values
        best_policy = np.zeros((self.S,self.T)) # keeps track of the best measurements

        for k in reversed(range(self.T)):
            for s in range(self.S): # Iterating over the sampled belief states 
                max_di = None
                max_val = 0 # max value over all possible states to measure here
                sample = self.samples[s]
                for i in range(self.N): # Iterating over all possible states we could measure 
                    post_1 = self.posterior(sample,i,1)
                    post_0 = self.posterior(sample,i,0)
                    exp_val = None
                    if k == self.T - 1:
                        exp_val = sample[i]*(self.reward(k,post_1)) + (1-sample[i])*(self.reward(k,post_0))
                    else:
                        # Find NN of the posteriors to pair correct future value with each posterior
                        NN_1 = self.get_NN_index(self.propagate_belief(post_1))
                        NN_0 = self.get_NN_index(self.propagate_belief(post_0))
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
        return curr_vals
        

                        

# Example here: 
#   
N_nodes = 10
T = 6
G = nx.erdos_renyi_graph(N_nodes, .2)
while not nx.is_connected(G):
    G = nx.erdos_renyi_graph(N_nodes, .2)     
trans_matrix = Random_walk_dynamics(G)
prior = np.ones(N_nodes)/N_nodes # Uniform prior
RW = RandomWalk(N_nodes,M = trans_matrix,p= prior)
OED = sOED(N_nodes,1,T,prior,RW,S = 10)
print(OED.value_iter())
nx.draw(G)
plt.show()


        
# To optimize: 
"""
We have a value function at each timestep based on the belief
Value iteration/iterating Bellman's equation until convergence
ow good everything is at each time step and indexed by number o f states and time, and our state space is discretized, 
BIG Matrix: with all the discretized states and how good they are at each time: iterating/dynamic programming on this matrix
Q: How to discretize the probability simplex: have to discretize st finding nearest neighbor is cheap 

To discretize the probability simplex, we should sample instead (and maybe have them move apart? Lenard Jones potential)

Alternately, policy iteration: see stackoverflow post that was shared

"""