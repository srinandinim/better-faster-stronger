import random
from copy import deepcopy
from game.models.optimalvaluefunction import (
    agent_to_pred_distances, get_future_reward_prediction_partial_prey)
from neuralnetworks.utils import load_model_for_agent
from .agent import Agent


class Agent3RLNN(Agent):
    def __init__(self, graph, location):
        # initialize agent location
        super().__init__(location)

        self.vpartial_model = load_model_for_agent(
            filename="OPTIMAL_VPARTIAL_MODEL.pkl")
        self.shortest_distances = agent_to_pred_distances(graph)

        # store the graph
        self.graph = graph

        # initialize agent belief prob dist
        self.beliefs = dict()
        self.init_probs_step1()

        # list of all prey prev locations
        self.prev_prey_locations = []

    def move(self, graph, prey, predator):
        """
        surveys the node with the highest probability of containing the prey
        updates the beliefs
        * if signal is false and we have previously not found prey, reinitialize beliefs to 1/(n - 2) for all nodes other than surveyed and agent current location
        * if signal is false and we have previously found prey, update beliefs based on probability that the prey could be in each position
        * if signal is true, beliefs is a one-hot vector

        calculates the predicted utility of each action in the agent's action space
        moves to the action with the greatest predicted utility
        """
        signal, surveyed_node = self.survey_node(prey)
        if len(self.prev_prey_locations) == 0:
            self.init_probs_step2(surveyed_node)
        elif signal and len(self.prev_prey_locations) > 0:
            self.init_probs_step3(surveyed_node)
        elif not signal and len(self.prev_prey_locations) > 0:
            self.init_probs_step4(surveyed_node)
        self.normalize_beliefs()

        action_space = graph.get_node_neighbors(
            self.location) + [self.location]

        best_action = None
        best_reward = -float("inf")
        for action in action_space:
            if action == predator.location:
                current_reward = -float("inf")
            else:
                current_reward = -1 + get_future_reward_prediction_partial_prey(
                    graph, action, self.beliefs, predator.location, self.shortest_distances, self.vpartial_model)
            if current_reward >= best_reward:
                best_reward = current_reward
                best_action = action

        self.location = best_action
        return len(self.prev_prey_locations), None

    def move_debug(self, graph, prey, predator):
        """
        updates location based on assignment specifications given
        """
        signal, surveyed_node = self.survey_node(prey)
        if len(self.prev_prey_locations) == 0:
            self.init_probs_step2(surveyed_node)
        elif signal and len(self.prev_prey_locations) > 0:
            self.init_probs_step3(surveyed_node)
        elif not signal and len(self.prev_prey_locations) > 0:
            self.init_probs_step4(surveyed_node)
        self.normalize_beliefs()

        action_space = graph.get_node_neighbors(
            self.location) + [self.location]

        best_action = None
        best_reward = -float("inf")
        for action in action_space:
            if action == predator.location:
                current_reward = -float("inf")
            else:
                current_reward = -1 + self.partial_utility(action, predator)
            if current_reward >= best_reward:
                best_reward = current_reward
                best_action = action

        self.location = best_action
        print(f"The action was to do {self.location}")

        return len(self.prev_prey_locations), None

    def survey_node(self, prey):
        """
        HELPER:
        RETURNS (SIGNAL=T/F, NODE_SURVEYED=n_i)
        Indicates node surveyed and whether or not prey is there. 
        """

        def get_highest_prob_nodes():
            """
            RETURNS LIST OF ALL NODES OF EQUIVALENT HIGHEST PROBABILITY. 
            """
            PROB, nodes = max(self.beliefs.values()), []
            for node, prob in self.beliefs.items():
                if prob == PROB:
                    nodes.append(node)
            return nodes

        signal = False
        node = random.choice(get_highest_prob_nodes())
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
