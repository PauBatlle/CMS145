from sOED import *
from RandomWalk import *
from Experiment import *
"""
- Fix posterior function
- Write tests
    - Each function in sOED/RW
    - t = 1 example that we have solution to
- Make experiments matrices in sOED
- Make value iteration actually value iteration (maybe in future)
- Visualizations of choices/beliefs
"""
# EXPERIMENT TESTS

#Example test, non-pytest version
ex =  build_experiments_random_walk(3,sensitivity=0.7, specificity=0.8)
#print(ex[0].M)
#print(ex[1].M)
#print(ex[2].M)
assert np.allclose(ex[1].M, np.array([[0.8, 0.3, 0.8],[0.2, 0.7, 0.2]]))

# RANDOMWALK TESTS
p1 = np.array([0.9,0.1])
p2 = np.array([0.3,0.7])
assert np.abs(dKL(p1,p2)-1.14573)< 0.00001, "KL divergence incorrect"

N = 3
random_RW = RandomWalk(N) # Initializes random RW with N nodes
assert (sum(random_RW.get_M()) - np.ones(N)).all(), "M not stochastic" 

# M_test = np.array([[0.3, 0.1, 0.6],[0.9,0,0.1],[0.3,0.6,0.1]])
# RW_test = RandomWalk(M_test.shape[0],M = M_test,p=np.array([0.1,0.4,0.5]))
# assert np.allclose(RW_test.update(),np.array([0.54,0.31,0.15])), "RW Update Incorrect"

# SOED tests
M_test = np.array([[0.3, 0.1, 0.6],[0.9,0,0.1],[0.3,0.6,0.1]])
RW_test = RandomWalk(M_test.shape[0],M = M_test,p=np.array([0.1,0.4,0.5]))
N = M_test.shape[0]
experiments = build_experiments_random_walk(N,1,1)
S = 3
T = 2
samples = np.array([[1,0,0],[0,1,0],[0,0,1]])
sOED_test = sOED(N,1,T,np.ones(N)/N,RW_test,experiments,S = 3, samples = samples)
xk = np.array([0.8,0.1,0.1])
assert sOED_test.reward(0,xk) == 0, "sOED intermediate reward incorrect"
# TODO: final reward is dLK(prior || posterior) or vv? 
assert np.allclose(sOED_test.reward(T-1,xk),0.66303), "sOED Reward Incorrect"
assert sOED_test.get_NN_index(xk) == 0, "NN function not working "

# TODO: write value iteration test (dependent on prior)





