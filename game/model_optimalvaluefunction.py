import os
import json
import pickle
from itertools import islice
from graph import Graph
from copy import deepcopy


# HELPER, MISCELLANEOUS FUNCTIONS


def retrieve_json(filename="GAME_GRAPH.json"):
    ''' 
    Function to retrieve a json from a given file, intended to use on
    stored graphs. 
    @param:filename - name of the json file being loaded.  
    @return the dictionary loaded, intended to be a graph adjacency list
    '''

    def keysStrToInt(d):
        if isinstance(d, dict):
            return {int(k): v for k, v in d.items()}
        return d

    dirname = "../graphs/"
    filepath = dirname + filename
    if os.path.exists(filepath):
        with open(filepath, "r") as fp:
            nbrs = json.load(fp, object_hook=keysStrToInt)
    return nbrs


def clean_up(u0, u1, sanity_check):
    '''
    Function to pickle the u0 and u1 vectors as well as do a 
    sanity check on their contents
    @param u0 - a dictionary, keyed by a tuple and valued by a number
    @param u1 - a dictionary, keyed by a tuple and valued by a number
    @return void
    '''

    pickle_vector(u0, 'u0.pickle')
    pickle_vector(u1, 'u1.pickle')
    print(sanity_check_value_updates(sanity_check, u0.values()))
    print(sanity_check_value_updates(sanity_check, u1.values()))


def pickle_vector(vector, filename):
    '''
    Function to take a vector and pickle it into a file
    @param vector - array to pickle
    @param filename - filename of file to store pickle into
    '''
    dirname = "game/pickles/"
    if not os.path.exists(os.path.dirname(dirname)):
        os.makedirs(os.path.dirname(dirname))

    filepath = dirname + filename
    with open(filepath, 'wb') as handle:
        pickle.dump(vector, handle)


def sanity_check_value_updates(n, iterable):
    '''
    Function to return the first n items of an iterable as a list
    @param n - number of values to return
    @param iterable - an object that is iterable
    '''

    return list(islice(iterable, n))


# HELPER, GRAPH FUNCTIONS


def calculate_shortest_distances(graph, source, goals):
    '''
    Function to calculate all of the shortest distances from the source to a list of goals
    @param:graph - the graph object to operate on
    @param:source - the source node to calculae distances from
    @param:goals - a list of goal nodes
    @return the dictionary of shortest distances from the source to the goal; each key is 
    a tuple structured as (source, goal) and the value is the shortest distance from the
    source to the goal
    '''

    queue = [source]
    visited = set()
    shortest_distances = {}
    lengths = {source: 0}
    completed = []

    while not len(queue) == 0 and not len(goals) == len(completed):

        cur = queue.pop(0)

        if cur in visited:
            continue

        if cur in goals:
            shortest_distances[(source, cur)] = lengths[cur]
            completed.append(cur)

        nbrs = graph.get_node_neighbors(cur)

        for nbr in nbrs:
            if nbr not in visited:
                queue.append(nbr)
                if nbr not in lengths.keys():
                    lengths[nbr] = lengths[cur] + 1

        visited.add(cur)
    return shortest_distances


def heuristic_dists_agent_to_pred(graph):
    '''
    Function to calculate all of the shortest distances between every pair of nodes in the graph
    @param:graph - the graph object to operate on
    @return the dictionary of shortest distances from the source to the goal; each key is 
    a tuple structured as (node1, node2) and the value is the shortest distance from
    node1 to node2
    '''

    agent_to_pred_dists = dict()
    for i in range(1, graph.get_nodes() + 1):
        agent_to_pred_dists.update(calculate_shortest_distances(
            graph, i, list(graph.get_neighbors().keys())))

    return agent_to_pred_dists


def optimal_pred_moves(graph, agent_loc, pred_loc, shortest_distances):
    """
    Function to return a list containing the best moves the predator can make to reach the agent
    the quickest. 
    @param graph - the graph the function operates on
    @param agent_loc - the location of the agent
    @param pred_loc - the location of the predator
    @param shortesT_distances - a dictionary containing the shortest distances between every pair of nodes in 
    the graph
    @return a list containing the predator's neighbors with the shortest distances to the agent
    """

    pred_nbrs_to_agent = {nbr: shortest_distances[(
        nbr, agent_loc)] for nbr in graph.get_node_neighbors(pred_loc)}
    smallest = min(pred_nbrs_to_agent.values())
    return [nbr for nbr in pred_nbrs_to_agent.keys() if pred_nbrs_to_agent[nbr] == smallest]


# HELPER, BELLMAN EQUATION COMPUTATION


def init_state_values(graph, shortest_distances):
    '''
    Function to initizalize the values of the U0 vector, used in computing
    the Bellman equation. Terminal states are when the agent is in the same spot as
    the prey or the predator, which is then valued at 0 and negative infintiy 
    respectively. Other values are estimated as the shortest distance between the
    agent and the prey. 
    @param graph: the graph the function operates on
    @param shortest_distances: a dictionary containing the shortest distance between every pair
    or nodes
    @return the U0 vector initialized in the function 
    '''

    graph_size = graph.get_nodes() + 1
    u0 = dict()

    for agent_loc in range(1, graph_size):
        for prey_loc in range(1, graph_size):
            for pred_loc in range(1, graph_size):
                state = (agent_loc, prey_loc, pred_loc)
                if agent_loc == prey_loc:
                    u0[state] = 0
                elif agent_loc == pred_loc:
                    u0[state] = -float("inf")
                else:
                    u0[state] = shortest_distances[(agent_loc, prey_loc)] * -1
    return u0


def transition_dynamics(graph, agent_loc, prey_loc, pred_loc, shortest_distances):
    """
    Function to return 'sister states' of the current state and their probabilities of occuring, 
    each one with a variation in the prey's and the predator's location. 
    @param:graph - the graph this function operates on
    @param:agent_loc - the location of the agent
    @param:prey_loc - the location of the prey
    @param:pred_loc - the location of the predator
    @param:shortest_distances - a dictionary containing the shortest distances between every pair of nodes
    @return a dictionary of sister states, where each key is a tuple structured as (agent location, prey location,
    predator location) and where each value is the probability of the state happening. 
    """

    new_states = dict()
    prey_next = graph.nbrs[prey_loc] + [prey_loc]
    pred_next = graph.nbrs[pred_loc]
    pred_optimal_next = set(optimal_pred_moves(
        graph, agent_loc, pred_loc, shortest_distances))

    for prey_next_state in prey_next:
        for pred_next_state in pred_next:
            next_state = (agent_loc, prey_next_state, pred_next_state)
            if pred_next_state in pred_optimal_next:
                new_states[next_state] = (
                    1 / len(prey_next)) * (0.4 / len(pred_next) + 0.6 / len(pred_optimal_next))
            else:
                new_states[next_state] = (
                    1 / len(prey_next)) * (0.4 / len(pred_next))

    return new_states


def calculate_next_iteration(new_states, u0, u1_cur_state):
    '''
    Function to calculate the utility of taking a specific action
    @param:new_states - the sister states of the current state this function is evaluating in
    @param:u0 - a dictionary containing the old utility values
    @param:u1_cur_state - a number representing the utility of the curent state, before taking this action
    @return - the maximum of the utility of the current state and the utility of taking a specific action
    '''

    future_reward = 0
    for sprime in new_states.keys():
        if u0[sprime] == -float("inf"):
            future_reward == -float("inf")
            break
        future_reward += new_states[sprime] * u0[sprime]

    action_value = -1 + future_reward
    return max(u1_cur_state, action_value)


# MAIN BELLMAN COMPUTATION


def calculate_optimal_values(graph, shortest_distances, convergence_factor):
    '''
    Function to use value iteration to compute the Bellman Equation
    @param graph - the graph this function operates on
    @param shortest_distances - a dictionary containing the shortest distances between any two nodes
    @param convergence-factor - the threshold that describes how close values need to be for convergence. 
    @return the number of rounds or 'sweeps' the function needs to converge and U0 (a vector containing the optimal 
    utilities)
    '''

    graph_size, ksweeps, converged = graph.get_nodes() + 1, 0, False
    u0, u1 = init_state_values(graph, shortest_distances), dict()

    while converged == False:

        print(f"{ksweeps}th iteration")
        converged = True
        for agent_loc in range(1, graph_size):
            for prey_loc in range(1, graph_size):
                for pred_loc in range(1, graph_size):

                    # determines the state we're currently at
                    state = (agent_loc, prey_loc, pred_loc)

                    # retrieve old values for terminal states
                    if agent_loc == prey_loc or agent_loc == pred_loc:
                        u1[state] = u0[state]
                        continue

                    # compute new values for non-terminal states
                    agent_actions = graph.nbrs[agent_loc] + [agent_loc]

                    # worst case is -inf
                    u1[state] = -9999

                    # iterate through all agent actions
                    for action in agent_actions:

                        # iterate through the transition
                        new_states = transition_dynamics(
                            graph, action, prey_loc, pred_loc, shortest_distances)
                        u1[state] = calculate_next_iteration(
                            new_states, u0, u1[state])

                    if converged and abs(u1[state] - u0[state]) > convergence_factor:
                        converged = False
                        print("THE ERROR IS")
                        print(abs(u1[state] - u0[state]))

        ksweeps += 1
        u0 = deepcopy(u1)
        u1 = dict()

    clean_up(u0, u1, 300)
    return ksweeps, u0


GAME_GRAPH = Graph(nbrs=retrieve_json())
shortest_distances = heuristic_dists_agent_to_pred(GAME_GRAPH)
ksweeps, u0 = calculate_optimal_values(GAME_GRAPH, shortest_distances, 0.001)

print(u0)
print(ksweeps)


"""
Step 0: Retrieves the graph on which to run value iteration. 

This functionality is copied from utils.py due to import dependencies/errors. 
Note: do not remove GAME_GRAPH.json unless you can figure out how to retrieve it from graphs directory. 
"""


"""
Step 1: Initialize Starting Distribution of State Values

Terminal States: 
     agent_loc == pred_loc --> V(s) = -9999 
     agent_loc == prey_loc --> V(s) = 0 
Non-Terminal States: 
    Heuristic Starting Distribution V(s) = bfs(agent_loc, prey_loc) * -1
    This initial seed heuristic enables value iteration to converge quickly
"""


"""
Step 2: Until convergence or a steady state, update non-terminal state values with Bellman Equations using Value Iteration.
- Initialize Hyperparameters: beta=0.9, eps=0.25, ksweeps=0, converged=False
- Initialize v_t and v_t+1 so that we can do synchronous updates with value iteration
- IF s is a terminal state: u_t+1(s) = u_t(s) 
- IF s is a non terminal state: u_t+1(s) = max of all actions in action space (-1 + beta * sum over all states(p(s'|s) * previous u_t(s')))
"""

# store the optimal pred locations cached
# optimal_pred_moves_cached = dict()


# def optimal_pred_moves(agent_loc, pred_loc):
#     """
#     Returns all optimal locations pred can go to.
#     """
#     pred_next_states = GAME_GRAPH.nbrs[pred_loc]

#     # stores the distances of all actions in the action space
#     distances = dict()

#     for pred_action in pred_next_states:

#         # compute -bfs(pred, agent) and cache that value for reuse
#         if (pred_action, agent_loc) not in optimal_pred_moves_cached:
#             distances[pred_action] = bfs(GAME_GRAPH, pred_action, agent_loc)
#             optimal_pred_moves_cached[(pred_action, agent_loc)] = distances[pred_action]

#         # retrieve cached value if it exists
#         else: distances[pred_action] = optimal_pred_moves_cached[(pred_action, agent_loc)]

#     # finds the shortest distance for the predator
#     shortest_distance = min(distances.values())

#     # gets all neighbors that result in shortest path
#     potential_moves = []
#     for key, value in distances.items():
#         if value == shortest_distance:
#             potential_moves.append(key)
#     return potential_moves


# RUNS THE VALUE ITERATION ALGORITHM UNTIL CONVERGENCE
# while converged == False:
#     print(f"{ksweeps}th iteration")

#     # iterate through all possible states
#     converged = True
#     for agent_loc in range(1,51):
#         print(f"{agent_loc} set completed.")
#         for prey_loc in range(1,51):
#             for pred_loc in range(1,51):

#                 # determines the state we're currently at
#                 state = (agent_loc, prey_loc, pred_loc)

#                 # retrieve old values for terminal states
#                 if agent_loc == prey_loc or agent_loc == pred_loc:
#                     u1[state] = u0[state]

#                 # compute new values for non-terminal states
#                 else:
#                     agent_actions = GAME_GRAPH.nbrs[agent_loc] + [agent_loc]

#                     # worst case is -inf
#                     u1[state] = -9999

#                     # iterate through all agent actions
#                     for action in agent_actions:

#                         # iterate through the transition
#                         new_states = transition_dynamics(agent_loc, prey_loc, pred_loc)
#                         future_reward = 0
#                         for sprime in new_states.keys():
#                             if u0[sprime] == -float("inf"):
#                                 future_reward == -float("inf")
#                                 break
#                             future_reward += new_states[sprime] * u0[sprime]

#                         action_value = -1 + 1 * future_reward
#                         u1[state] = max(u1[state], action_value)

#                     if convegered and abs(u1[state] - u0[state]) > EPSILON:
#                         converged = False
#                         print("THE ERROR IS")
#                         print(abs(u1[state] - u0[state]))


#     ksweeps += 1

# converged = True
# for state in u0.keys():
#     if abs(u1[state] - u0[state]) > EPSILON:
#         converged = False
#         print("THE ERROR IS")
#         print(abs(u1[state] - u0[state]))
#         break

# pickle_vector(u0, 'u0.pickle')
# pickle_vector(u1, 'u1.pickle')

# print(sanity_check_value_updates(300, u0.values()))
# print(sanity_check_value_updates(300, u1.values()))

# u0 = deepcopy(u1)
# u1 = dict()

# for agent_loc in range(1,51):
#     for prey_loc in range(1,51):
#         agent_to_pred_dists[(agent_loc, prey_loc)] = bfs(GAME_GRAPH, agent_loc, prey_loc)
# return agent_to_pred_dists

# # create prev hashmap to maintain a directed shortest path
# dist = {}
# dist[source] = 0
# prev = {}
# prev[source] = None

# # loop until queue is empty
# while len(queue) > 0:
#     node = queue.pop(0)
#     nbrs = graph.get_node_neighbors(node)
#     for nbr in nbrs:
#         if nbr not in dist:
#             dist[nbr], prev[nbr] = dist[node] + 1, node
#             if goal == nbr:
#                 return dist[nbr]
#             queue.append(nbr)
# return -1
