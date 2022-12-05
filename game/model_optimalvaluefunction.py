import json
import os
import pickle
from copy import deepcopy
from itertools import islice
from graph import Graph

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

    dirname = "/graphs/"
    filepath = dirname + filename
    parent_directory = os.path.split(os.path.dirname(__file__))[0]
    filepath = parent_directory + dirname + filename
    if os.path.exists(filepath):
        print("in here")
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


def agent_to_pred_distances(graph):
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
def init_state_values(graph):
    '''
    Function to initizalize the values of the U0 vector, used in computing
    the Bellman equation. Terminal states are when the agent is in the same spot as
    the prey or the predator, which is then valued at 0 and negative infintiy 
    respectively. Other values are estimated as the shortest distance between the
    agent and the prey. 
    @param graph: the graph the function operates on
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
                    u0[state] = -1
    return u0


def get_future_reward(graph, agent_loc, prey_loc, pred_loc, shortest_distances, u0):
    """
    Function to return the future reward of the current state by considering all of the 'sister states',
    or variations in the prey's and predator's location
    @param:graph - the graph this function operates on
    @param:agent_loc - the location of the agent
    @param:prey_loc - the location of the prey
    @param:pred_loc - the location of the predator
    @param:shortest_distances - a dictionary containing the shortest distances between every pair of nodes
    @param:u0 - a vector containing the utilities of each state from the previous sweep
    @return the future reward from being in this state
    """

    prey_next = graph.nbrs[prey_loc] + [prey_loc]
    pred_next = graph.nbrs[pred_loc]
    pred_optimal_next = set(optimal_pred_moves(
        graph, agent_loc, pred_loc, shortest_distances))

    future_reward = 0
    for prey_next_state in prey_next:
        for pred_next_state in pred_next:
            next_state = (agent_loc, prey_next_state, pred_next_state)
            if u0[next_state] == -float("inf"):
                return -float("inf")

            gamma = 0.6 / \
                len(pred_optimal_next) if pred_next_state in pred_optimal_next else 0
            future_reward += u0[next_state] * \
                ((1 / len(prey_next)) * (0.4 / len(pred_next) + gamma))

    return future_reward


def get_current_reward(graph, agent_loc, prey_loc, pred_loc, shortest_distances):
    """
    Function to return the current reward of ending in the given state
    @param:graph - the graph this function operates on
    @param:agent_loc - the location of the agent
    @param:prey_loc - the location of the prey
    @param:pred_loc - the location of the predator
    @param:shortest_distances - a dictionary containing the shortest distances between every pair of nodes
    @param:u0 - a vector containing the utilities of each state from the previous sweep
    @return the current reward for ending in the current state
    """
    return -1 if shortest_distances[(agent_loc, pred_loc)] > 1 else -float("inf")


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
    u0, u1 = init_state_values(graph), dict()

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
                    agent_actions = graph.nbrs[agent_loc]

                    # worst case is -inf
                    u1[state] = -float("inf")

                    # iterate through all agent actions
                    for action in agent_actions:
                        # iterate through the transition
                        u1[state] = max(u1[state], get_current_reward(graph, action, prey_loc, pred_loc, shortest_distances) +
                                        get_future_reward(graph, action, prey_loc, pred_loc, shortest_distances, u0))

                    if converged and abs(u1[state] - u0[state]) > convergence_factor:
                        converged = False
                        print(f'ERROR: {abs(u1[state] - u0[state])}')

        ksweeps += 1
        u0 = deepcopy(u1)
        u1 = dict()

    clean_up(u0, u1, 300)
    return ksweeps, u0


GAME_GRAPH = Graph(nbrs=retrieve_json())
shortest_distances = agent_to_pred_distances(GAME_GRAPH)
print(shortest_distances)
ksweeps, u0 = calculate_optimal_values(GAME_GRAPH, shortest_distances, 0.001)

print(u0)
print(ksweeps)
