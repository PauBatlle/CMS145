import numpy as np 
import math 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm 
from scipy.stats import entropy
from design_policy import RandomWalk

# Helper functions
def transition_matrix(A):
    """
    Generates a random transition matrix for the associated adjacency matrix A
    """
    pass

def sample_simplex(dimension, n_points):
    return np.random.dirichlet([1]*dimension, size = n_points)


def dKL(prior, posterior):
    return entropy(posterior, prior, base = 2)

def reward(time, prior, experiment): #This should be a class and the example here just one instance
    #Experiment is a |S|x|Θ| matrix
    #Prior a |Θ| vector
    #Time = number of experiments done *before* this
    if time < tsteps-1:
        return 0
    Ptest = experiment@prior
    Posterior = np.divide(np.multiply(experiment, prior),Ptest.reshape(-1,1))
    return sum([Ptest[i]*dKL(prior, Posterior[i,:]) for i in range(experiment.shape[0])])

# Example parameters (specific)
N_nodes = 10 #Number of node networks
world_size = 10 #Number of possible states of the world (in this case = N_nodes)
exp_av = 10 #Number of possible experiments at each timestep (in this case = N_nodes)
tsteps = 10 #Number of timesteps, we count them as 0...n-1
mesh_points = 1000 #Number of samples of the probability simplex
sensitivity = 0.99 #Probability you find the RW in the node you are looking for, provided that is is there. 
specificity = 0.97 #Probability you don't find the RW in the node you are looking for, provided it is not there



rw1 = RandomWalk(5)
print(rw1.get_M())
