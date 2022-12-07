import pickle
import random
from copy import deepcopy
from game.models.optimalvaluefunction import (agent_to_pred_distances,
                                              get_future_reward)
from .agent import Agent


class Agent3RL(Agent):
    def __init__(self, graph, location):
        # initialize agent location
        super().__init__(graph, location)

        self.utility = pickle.load(
            open("game/pickles/OPTIMAL_U*.pickle", "rb"))
        self.shortest_distances = agent_to_pred_distances(graph)

        # store the graph
        self.graph = graph

        # initialize the belief dict
        self.beliefs = dict()

        # initialize agent belief prob dist
        self.init_probs_step1()

        # list of all prey prev locations
        self.prev_prey_locations = []

    def move(self, graph, prey, predator):
        """
        updates location based on assignment specifications given
        """
        return len(self.prev_prey_locations), None

    def move_debug(self, graph, prey, predator):
        """
        updates location based on assignment specifications given
        """

        return None, None

    def get_highest_prob_nodes(self):
        """
        HELPER:
        RETURNS LIST OF ALL NODES OF EQUIVALENT HIGHEST PROBABILITY. 
        """
        PROB, nodes = max(self.beliefs.values()), []
        for node, prob in self.beliefs.items():
            if prob == PROB:
                nodes.append(node)
        return nodes

    def survey_node(self, prey):
        """
        HELPER:
        RETURNS (SIGNAL=T/F, NODE_SURVEYED=n_i)
        Indicates node surveyed and whether or not prey is there. 
        """
        signal = False
        node = random.choice(self.get_highest_prob_nodes())
        if prey.location == node:
            signal = True
            self.prev_prey_locations.append(node)
        return signal, node

    def init_probs_step1(self):
        """
        CORE: INITIALIZING INITIAL PROBABILITY.

        BELIEF UPDATE STEP 1: 
        P(n_i) = 1 / (n-1) for every node not containing agent 
        P(n_k) = 0 for the kth node containing the agent
        """
        for i in range(1, self.graph.get_nodes() + 1):
            if i == self.location:
                self.beliefs[i] = 0
            else:
                self.beliefs[i] = 1 / (self.graph.get_nodes() - 1)

    def init_probs_step2(self, surveyed_node):
        """
        CORE: SURVEYED NODE BUT THE PREY IS NOT THERE AND WE HAVEN'T FOUND PREY BEFORE. 

        BELIEF UPDATE STEP 2: 
        P(n_i) = 1 / (n-2) for every node not agent's current location or surveyed_node
        P(n_k) = P(n_surveyed) = 0 for the kth node containing the agent and the surveyed node
        """
        for node, _ in self.beliefs.items():
            if node == self.location or node == surveyed_node:
                self.beliefs[node] = 0
            else:
                self.beliefs[node] = 1 / (self.graph.get_nodes() - 2)

    def init_probs_step3(self, surveyed_node):
        """
        CORE: SURVEYED NODE CONTAINS PREY!

        BELIEF UPDATE STEP 3: 
        P(n_surveyed) = 1
        P(n_i) = 0 for all i != n_surveyed
        """
        for node, _ in self.beliefs.items():
            if node == surveyed_node:
                self.beliefs[node] = 1
            else:
                self.beliefs[node] = 0

        # SETS UP FRONTIER OF UNIQUE STATES VISITED FOR BELIEF UPDATE 4
        self.frontier = set()
        self.frontier.add(surveyed_node)

    def init_probs_step4(self, surveyed_node):
        """
        CORE: SURVEYED NODE DOESN'T CONTAIN PREY BUT WE FOUND A PREY BEFORE!

        BELIEF UPDATE STEP 4: 
        - Given frontier F_{t-1} at t-1, determine frontier F_{t} at t, and compute # of ways to get to each element in F_{t}
        - Remove the number of ways to get to current agent location if exists in set or current surveyed node if it exists in set
        - Update beliefs based on the number of ways to get to each place in a particular state
        """

        # RETRIEVES THE FREQUENCY EACH STATE CAN BE VISITED FROM FRONTIER
        counts = dict()
        for node in self.frontier:
            counts[node] = counts.get(node, 0) + 1
            for nbr in self.graph.nbrs[node]:
                counts[nbr] = counts.get(nbr, 0) + 1

        # UDPATES FRONTIER, ALL POSSIBLE STATES AGENT CAN BE IN
        self.frontier = set(counts.keys())

        # WE COMPUTE THE PROBABILITIES BASED ON FREQUENCY
        probability_mass = deepcopy(counts)
        probability_mass[self.location] = 0
        probability_mass[surveyed_node] = 0
        denominator = sum(probability_mass.values())

        # UPDATE THE BELIEFS BASED ON FREQUENCIES
        for key in probability_mass.keys():
            self.beliefs[key] = probability_mass[key] / denominator
        for key in self.beliefs.keys():
            if key not in probability_mass:
                self.beliefs[key] = 0

    def normalize_beliefs(self):
        """
        ENSURES THAT ALL PROBABILITIES SUM TO 1
        """
        values_sum = sum(self.beliefs.values())
        for node, probability in self.beliefs.items():
            self.beliefs[node] = probability/values_sum
