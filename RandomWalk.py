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


def dKL(posterior, prior):
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
        q = self.M[self.state,:][0]
        self.state = random.choice(self.N,1,p = q)
        return self.state
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
    