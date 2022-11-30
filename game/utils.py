import pickle 
from graph import Graph 

def save_graph_state(game_graph = Graph(nodes=50)):
    """
    for the RL setting, we run the agent on one graph 
    initialized with different starting states. 

    verified by professor since we need to compute the 
    optimal values with the bellman equation and then
    use a function approximation model on that. 
    """
    file_reader = open("gamegraph.obj", "wb")
    pickle.dump(game_graph, file_reader)

def open_graph_state():
    """
    retrieves and returns the graph from pickled object
    """
    file_parser = open("gamegraph.obj", "rb")
    return pickle.load(file_parser)

# save_graph_state()

# VERIFY THAT THE SAVE FUNCTIONALITY WORKS
"""
g1 = Graph(50)
print(g1.nbrs)
save_graph_state(g1)
g2 = open_graph_state()
print(g2.nbrs)
"""
