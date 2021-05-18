import numpy as np 
import math 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from tqdm import tqdm 
from scipy.spatial import KDTree
from numpy import random
from scipy.stats import entropy

class Experiment(): #Not needed as a class if you don't like it
    def __init__(self, M):
        #Experiment is a |S|x|Θ| matrix
        self.M = M
        self.nsignals = M.shape[0]
        self.nstates = M.shape[1]


def build_experiments_random_walk(N_nodes, sensitivity, specificity ):

    #sensitivity = Probability you find the RW in the node you are looking for, provided that is is there. 
    #specificity = Probability you don't find the RW in the node you are looking for, provided it is not there
    exps = []
    for i in range(N_nodes):
        M = np.zeros((2,N_nodes))
        for k in range(N_nodes):
            M[:,k] = [1-sensitivity, sensitivity] if k == i else [specificity, 1-specificity]
        exps.append(Experiment(M))
    return exps 