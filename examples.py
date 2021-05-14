import numpy as np 
import math 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm 
from design_policy import RandomWalk

def transition_matrix(A):
    """
    Generates a random transition matrix for the associated adjacency matrix A
    """
    pass

rw1 = RandomWalk(5)
print(rw1.get_M())
