from sOED import *
from Experiment import *
from RandomWalk import *


N_nodes = 10
T = 6
G = nx.erdos_renyi_graph(N_nodes, .2)
while not nx.is_connected(G):
    G = nx.erdos_renyi_graph(N_nodes, .2)     
trans_matrix = Random_walk_dynamics(G)
prior = np.ones(N_nodes)/N_nodes # Uniform prior
RW = RandomWalk(N_nodes,M = trans_matrix,p= prior)
#Build possible experiments
specificity = 0.99
sensitivity = 0.99
Experiment_list = build_experiments_random_walk(N_nodes, sensitivity, specificity)
OED = sOED(N_nodes,T,prior,RW,experiments = Experiment_list, S = 100)
#print(OED.posterior(np.array([1,1,1,1,1,1,1,1,1,1]), 2, 3))
print(OED.value_iter())
#nx.draw(G)
#plt.show()
