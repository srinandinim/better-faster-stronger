import os, json, pickle 
from itertools import islice 
from graph import Graph 
from copy import deepcopy 

"""
Step 0: Retrieves the graph on which to run value iteration. 

This functionality is copied from utils.py due to import dependencies/errors. 
Note: do not remove GAME_GRAPH.json unless you can figure out how to retrieve it from graphs directory. 
"""
def retrieve_json(filename="GAME_GRAPH.json"):
    def keysStrToInt(d):
        if isinstance(d, dict):
            return {int(k): v for k, v in d.items()}
        return d
    if os.path.exists(filename):
        with open(filename, "r") as fp:
            nbrs = json.load(fp, object_hook=keysStrToInt)
    return nbrs

GAME_GRAPH = Graph(nbrs=retrieve_json())

"""
Step 1: Initialize Starting Distribution of State Values

Terminal States: 
     agent_loc == pred_loc --> V(s) = -9999 
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
    return u0 

"""
Step 2: Until convergence or a steady state, update non-terminal state values with Bellman Equations using Value Iteration.
- Initialize Hyperparameters: beta=0.9, eps=0.25, ksweeps=0, converged=False
- Initialize v_t and v_t+1 so that we can do synchronous updates with value iteration
- IF s is a terminal state: u_t+1(s) = u_t(s) 
- IF s is a non terminal state: u_t+1(s) = max of all actions in action space (-1 + beta * sum over all states(p(s'|s) * previous u_t(s')))
"""
BETA_DISCOUNTFACTOR, EPSILON, ksweeps, converged = 0.90, 0.10, 0, False 
u0, u1 = init_state_values(), dict() 

# store the optimal pred locations cached
optimal_pred_moves_cached = dict() 

def sanity_check_value_updates(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def optimal_pred_moves(agent_loc, pred_loc):
    """
    Returns all optimal locations pred can go to. 
    """
    pred_next_states = GAME_GRAPH.nbrs[pred_loc] + [pred_loc]

    # stores the distances of all actions in the action space 
    distances = dict() 

    for pred_action in pred_next_states:

        # compute -bfs(pred, agent) and cache that value for reuse
        if (pred_action, agent_loc) not in optimal_pred_moves_cached:
            distances[pred_action] = bfs(GAME_GRAPH, pred_action, agent_loc)
            optimal_pred_moves_cached[(pred_action, agent_loc)] = distances[pred_action]

        # retrieve cached value if it exists 
        else: distances[pred_action] = optimal_pred_moves_cached[(pred_action, agent_loc)]
    
    # finds the shortest distance for the predator 
    shortest_distance = min(distances.values())

    # gets all neighbors that result in shortest path 
    potential_moves = [] 
    for key, value in distances.items():
        if value == shortest_distance:
            potential_moves.append(key)
    return potential_moves 

def transition_dynamics(agent_loc, prey_loc, pred_loc):
    """
    Returns next possible states for an action and their probabilities
    """

    # transition dynamics: {s' : p(s' | s)}
    new_states = dict() 

    # retrieves all possible prey, pred_ed, pred_opt moves
    prey_next = GAME_GRAPH.nbrs[prey_loc] + [prey_loc]
    pred_next = GAME_GRAPH.nbrs[pred_loc] + [pred_loc]
    pred_optimal_next = set(optimal_pred_moves(agent_loc, pred_loc))

    # iterate through all state transitions given an agent action
    for prey_next_state in prey_next: 
        for pred_next_state in pred_next: 
            next_state = (agent_loc, prey_next_state, pred_next_state)
            if pred_next_state in pred_optimal_next:
                new_states[next_state] = (1 / len(prey_next)) * (0.4 / len(pred_next) + 0.6 / len(pred_optimal_next))
            else: 
                new_states[next_state] = (1 / len(prey_next)) * (0.4 / len(pred_next))
    return new_states 

# RUNS THE VALUE ITERATION ALGORITHM UNTIL CONVERGENCE
while converged == False: 
    print(f"{ksweeps}th iteration")

    # iterate through all possible states
    for agent_loc in range(1,51):
        print(f"{agent_loc} set completed.")
        for prey_loc in range(1,51):
            for pred_loc in range(1,51):

                # determines the state we're currently at
                state = (agent_loc, prey_loc, pred_loc)
                
                # retrieve old values for terminal states
                if agent_loc == prey_loc or agent_loc == pred_loc: 
                    u1[state] = u0[state]

                # compute new values for non-terminal states
                else: 
                    agent_actions = GAME_GRAPH.nbrs[agent_loc] + [agent_loc]

                    # worst case is -inf 
                    max_action_value = -9999

                    # iterate through all agent actions 
                    for action in agent_actions:

                        # iterate through the transition
                        new_states = transition_dynamics(agent_loc, prey_loc, pred_loc)
                        future_reward = 0 
                        for sprime in new_states.keys():
                            if u0[sprime] == -float("inf"):
                                future_reward == -float("inf")
                                break
                            future_reward += new_states[sprime] * u0[sprime]
    
                        action_value = -1 + 1 * future_reward
                        max_action_value = max(max_action_value, action_value)

                    u1[state] = max_action_value

    ksweeps += 1

    converged = True 
    for state in u0.keys():
        if abs(u1[state] - u0[state]) > EPSILON:
            converged = False 
            print("THE ERROR IS")
            print(abs(u1[state] - u0[state]))
            break  

    with open('u0.pickle', 'wb') as handle:
        pickle.dump(u0, handle)
    
    with open('u1.pickle', 'wb') as handle:
        pickle.dump(u1, handle)

    print(sanity_check_value_updates(300, u0.values()))
    print(sanity_check_value_updates(300, u1.values()))

    u0 = deepcopy(u1)
    u1 = dict()

print(u0)
print(ksweeps)