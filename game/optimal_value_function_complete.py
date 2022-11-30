"""
THIS FILE COMPUTES U*(s), THE OPTIMAL VALUE FUNCTION, GIVEN AS INPUT A GRAPH. 
IT COMPUTES THE OPTIMAL VALUE FUNCTION ITERATIVELY, USING VALUE ITERATION AND DP.

Note:
I just copied the retrieve graph functionality from utils.py.
I keep running into errors with relative errors and so on / so forth.
If you can fix that'd be great for cleanliness, but ultimately not necessary. 
"""

import json
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

# loads into memory the graph we are optimizing for
graph = Graph(nbrs=retrieve_graph())

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
                elif agent_idx == pred_idx: value = -float("inf")
                else: value = -1

                # add the state to the dictionary
                states[state] = value 

    return states 

# defines minimum level of error until convergence
ksweeps = 0 
epsilon = 0.2
converged = False 

# defines the previous values at t and at t + 1 
value_k0 = initialize_state_space()
value_k1 = {} 

# runs the value iteration algorithm until it converges
while converged == False: 
    for agent_idx in range(1, 51):
        for prey_idx in range(1,51):
            for pred_idx in range(1,51):
                state = (agent_idx, prey_idx, pred_idx)
                
                value = 0
                if agent_idx == prey_idx: value = 0  
                elif agent_idx == pred_idx: value = -float("inf")
                else:

                    




