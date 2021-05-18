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
#Example test, non-pytest version
ex =  build_experiments_random_walk(3,sensitivity=0.7, specificity=0.8)
#print(ex[0].M)
#print(ex[1].M)
#print(ex[2].M)
assert np.allclose(ex[1].M, np.array([[0.8, 0.3, 0.8],[0.2, 0.7, 0.2]]))

