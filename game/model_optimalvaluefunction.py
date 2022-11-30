import os, json, pickle 
from graph import Graph 
from copy import deepcopy 

"""
Step 0: Retrieves the graph on which to run value iteration. 
This functionality is copied from utils.py due to import dependencies/errors. 
"""
def retrieve_graph(filename="sample.json"):
    def keysStrToInt(d):
        if isinstance(d, dict):
            return {int(k): v for k, v in d.items()}
        return d
    dirname = "graphs/"
    filepath = dirname + filename
    if os.path.exists(filepath):
        with open(filepath, "r") as fp:
            nbrs = json.load(fp, object_hook=keysStrToInt)
    return nbrs

GAME_GRAPH = Graph(nbrs=retrieve_graph())
print(GAME_GRAPH.nbrs)

"""
Step 1: Initialize Starting Distribution of State Values

Terminal States: 
     agent_loc == pred_loc --> V(s) = -infty 
     agent_loc == prey_loc --> V(s) = 0 
     
Non-Terminal States: 
    Heuristic Starting Distribution V(s) = bfs(agent_loc, prey_loc) * -1
    This initial seed heuristic enables value iteration to converge quickly
"""
def bfs(graph, source, goal):
    # create queue and enqueue source
    queue = [source]

    # create a dist hashmap to store distance between nodes
    dist = {}
    dist[source] = 0

    # create prev hashmap to maintain a directed shortest path
    prev = {}
    prev[source] = None

    # loop until queue is empty
    while len(queue) > 0:
        node = queue.pop(0)
        nbrs = graph.get_node_neighbors(node)
        for nbr in nbrs:
            if nbr not in dist:
                dist[nbr], prev[nbr] = dist[node] + 1, node
                if goal == nbr:
                    return dist[nbr]
                queue.append(nbr)
    return -1

def init_state_values():
    def heuristic_dists_agent_to_pred():
        agent_to_pred_dists = dict() 
        for agent_loc in range(1,51):
            for prey_loc in range(1,51):
                agent_to_pred_dists[(agent_loc, prey_loc)] = bfs(GAME_GRAPH, agent_loc, prey_loc)
        return agent_to_pred_dists
                
    u0 = dict() 
    u0_heuristic = heuristic_dists_agent_to_pred()

    for agent_loc in range(1,51):
        for prey_loc in range(1,51):
            for pred_loc in range(1,51):
                state = (agent_loc, prey_loc, pred_loc)
                if agent_loc == prey_loc: 
                    u0[state] = 0 
                elif agent_loc == pred_loc:
                    u0[state] = -float("inf")
                else: 
                    u0[state] =  u0_heuristic[(agent_loc, prey_loc)] * -1 




# Step 2: Until k sweeps of convergence, update the values