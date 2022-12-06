import pickle
from game.models.optimalvaluefunction import (agent_to_pred_distances,
                                              get_future_reward)
from .agent import Agent
import game.neuralnetworks.nn as nn


class Agent1RLNN(Agent):
    def __init__(self, graph, location):
        # initialize agent location
        super().__init__(location)
        self.utility = pickle.load(open("game/pickles/u0.pickle", "rb"))
        self.shortest_distances = agent_to_pred_distances(graph)
        self.vcomplete_model = nn.load_model(filename="OPTIMAL_VCOMPLETE_MODEL.pkl")

    def move(self, graph, prey, predator):
        """
        updates location based on assignment specifications given
        """
        action_space = graph.get_node_neighbors(
            self.location) + [self.location]

        best_action = None
        best_reward = -float("inf")
        for action in action_space:
            if action == prey.location:
                current_reward = -1
            elif action == predator.location:
                current_reward = -float("inf")
            else:
                current_reward = -1 + get_future_reward(
                    graph, action, prey.location, predator.location, self.shortest_distances, self.utility)
            if current_reward >= best_reward:
                best_reward = current_reward
                best_action = action

        self.location = best_action
        return None, None

    def move_debug(self, graph, prey, predator):
        """
        updates location based on assignment specifications given
        """
        action_space = graph.get_node_neighbors(
            self.location) + [self.location]

        best_action = None
        best_reward = -float("inf")
        for action in action_space:
            if action == prey.location:
                current_reward = -1
            elif action == predator.location:
                current_reward = -float("inf")
            else:
                current_reward = -1 + get_future_reward(
                    graph, action, prey.location, predator.location, self.shortest_distances, self.utility)
            if current_reward >= best_reward:
                best_reward = current_reward
                best_action = action

        self.location = best_action
        print(f"The action was to do {self.location}")

        return None, None
