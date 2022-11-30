"""
THIS FILE COMPUTES U*(s), THE OPTIMAL VALUE FUNCTION, GIVEN AS INPUT A GRAPH. 
IT COMPUTES THE OPTIMAL VALUE FUNCTION ITERATIVELY, USING VALUE ITERATION AND DP.

Note:
I just copied the retrieve graph functionality from utils.py.
I keep running into errors with relative errors and so on / so forth.
If you can fix that'd be great for cleanliness, but ultimately not necessary. 
"""

import json
import pickle 
import os

def retrieve_graph(filename="sample.json"):
    def keysStrToInt(d):
        if isinstance(d, dict):
            return {int(k): v for k, v in d.items()}
        return d

    nbrs = None 
    dirname = "graphs/"
    filepath = dirname + filename
    if os.path.exists(filepath):
        with open(filepath, "r") as fp:
            nbrs = json.load(fp, object_hook=keysStrToInt)
    return nbrs

from graph import Graph
from copy import deepcopy

hashmap_previous_probs = dict() 

# loads into memory the graph we are optimizing for
GAME_GRAPH = Graph(nbrs=retrieve_graph())

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

def optimal_pred_moves(agent_idx, pred_idx):
    pred_next_states = GAME_GRAPH.nbrs[pred_idx] + [pred_idx]

    distances = dict()
    for neighbor in pred_next_states:

        if (neighbor, agent_idx) in hashmap_previous_probs:
            distances[neighbor] = hashmap_previous_probs[(neighbor, agent_idx)]
        else: 
            distances[neighbor] = bfs(GAME_GRAPH, neighbor, agent_idx)
            hashmap_previous_probs[neighbor] = distances[neighbor]

    # get shortest distance from current location to agent
    shortest_distance = min(distances.values())

    # get all neighbors that result in the shortest path
    potential_moves = []
    for key, value in distances.items():
        if value == shortest_distance:
            potential_moves.append(key)
    return potential_moves

def transition_dynamics(agent_idx, prey_idx, pred_idx):

    new_states = dict() 

    prey_next_states = GAME_GRAPH.nbrs[prey_idx] + [prey_idx]
    pred_next_states = GAME_GRAPH.nbrs[pred_idx] + [pred_idx]
    pred_optimal_next_states = set(optimal_pred_moves(agent_idx, pred_idx))

    num_prey_moves = len(prey_next_states)
    num_pred_moves = len(pred_next_states)
    num_pred_moves_optimal = len(pred_optimal_next_states)

    for prey_next_state in prey_next_states:
        for pred_next_state in pred_next_states:
            next_state = (agent_idx, prey_next_state, pred_next_state)
            if pred_next_state in pred_optimal_next_states: 
                new_states[next_state] = (1 / num_prey_moves) * (0.4 / num_pred_moves + 0.6 / num_pred_moves_optimal)
            else: 
                new_states[next_state] = (1 / num_prey_moves) * (0.4 / num_pred_moves)

    return new_states 




# initializes the state space of the model with initial distribution and values. 
def initialize_state_space():
    states = dict() 
    for agent_idx in range(1, 51):
        for prey_idx in range(1,51):
            for pred_idx in range(1,51):

                state = (agent_idx, prey_idx, pred_idx)
                
                # recall round order: agent -> prey -> predator 
                # if agent_idx = prey_idx, then agent will kill prey in turn 
                # elif agent_idx = pred_idx, then pred will kill agent next turn 
                # otherwise, initialize all other values randomly, or some arbitrary fixed value
                # note, in expectation, all the non-terminal states will converge
                # to steady state due to the bellman optimality equations.

                value = 0 
                if agent_idx == prey_idx: value = 0  
                elif agent_idx == pred_idx: value = -99999
                else: value = -1

                # add the state to the dictionary
                states[state] = value 

    return states 

# defines minimum level of error until convergence
EPSILON = 0.2
BETA = 0.90 
ksweeps = 0 
converged = False 

# defines the previous values at t and at t + 1 
value_k0 = initialize_state_space()
value_k1 = {} 

# runs the value iteration algorithm until it converges
while converged == False: 
    print(f"{ksweeps}th iteration")
    for agent_idx in range(1, 51):
        print(f"{agent_idx} set completed.")
        for prey_idx in range(1,51):
            for pred_idx in range(1,51):
                state = (agent_idx, prey_idx, pred_idx)
                #print(state)
                value_k1[state] = value_k0[state]

                # if its a terminal state keep the old value 
                # if its a new value, compute expected sum of discounted rewards
                if agent_idx != prey_idx or agent_idx != pred_idx:
                    agent_neighbors = GAME_GRAPH.nbrs[agent_idx]
                    agent_next_states = GAME_GRAPH.nbrs[agent_idx]+ [agent_idx]

                    value_k1[state] = -10000000

                    # find the transition probabilities and dynamics for new states s' given an action
                    for action in agent_next_states: 
                        next_states = transition_dynamics(agent_idx, prey_idx, pred_idx)
                        sum_of_future_rewards = 0 
                        for new_state in next_states.keys():
                            sum_of_future_rewards += value_k0[new_state] * next_states[new_state]
                        new_value = -1 + BETA * sum_of_future_rewards
                    value_k1[state] = max(value_k1[state], new_value)
    

    ksweeps += 1
    if ksweeps == 30 : 
        print(ksweeps)
        converged=True 

        with open('value_t-1.pickle', 'wb') as handle:
            pickle.dump(value_k0, handle)
        
        with open('value_t.pickle', 'wb') as handle:
            pickle.dump(value_k1, handle)

    value_k0 = deepcopy(value_k1)
    value_k1 = dict()

print(value_k0)


                    







