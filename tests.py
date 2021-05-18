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
ex =  build_experiments_random_walk(2, 0.2, 0.9)
assert np.allclose(ex[0].M,np.array([[0.2, 0.1],[0.8, 0.9]]))
assert np.allclose(ex[1].M, np.array([[0.1, 0.2], [0.9, 0.8]]))

