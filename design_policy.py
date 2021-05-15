import numpy as np 
import math 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx
from tqdm import tqdm 
from numpy import random

#This is a comment for a commit
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
        else: self.p = p 

        if s is None: 
            self.s = np.choice(N,1,p=self.p)
        else: self.s = s

        self.state = self.s # Current state
        self.t = 0 # Time step 
    
    def get_M(self):
        return self.M
    
    def update(self):
        '''
        Take one step and update attributes
        '''
        #TODO
        pass
    
    def observe(self,i):
        '''
        Returns observation of state i
        '''
        #TODO
        pass
    
    def get_phi(self,k,i,x_k):
        '''
        Takes in index k for experiment number, a function taking index k for the experiment number, i for the basis index, and current belief x_k to evaluate and returns the result of the 
                basis function phi_k,i(x_k)
        ********** Possibly should be in examples.py ****************
        '''
        #TODO ### Is this problem-specific? Should it be in the RandomWalk class? Yes
        pass

class sOED:
    def __init__(self,N,pi_explore,L,R,T,get_phi,m,p):
        '''
        Initializes a sequential optimal experiment design with the following variables: 
            N: number of experiments
            get_phi: a function taking index k for the experiment number, i for the basis index, and current belief x_k to evaluate and returns the result of the 
                basis function phi_k,i(x_k)
            L: number of policy updates
            R: number of exploration trajectories
            T: number of exploitation trajectories 
            m: number of states (discrete)
            p: prior beliefs over states
            pi_explore: exploration policy 
        '''
        self.N = N
        self.L = L
        self.R = R
        self.T = T
        self.m = m
        self.p = p
        self.xb = p # Current beliefs about the world
        self.pi_expore = pi_explore #TODO: what is pi_explore practically????
        


        
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