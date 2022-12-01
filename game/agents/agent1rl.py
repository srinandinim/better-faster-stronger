import pickle
import random
from .agent import Agent


class Agent1RL(Agent):
    def __init__(self, location):
        # initialize agent location
        super().__init__(location)
        self.utility = pickle.load(open("game/pickles/u0.pickle", "rb"))

    def move(self, graph, prey, predator):
        """
        updates location based on assignment specifications given
        """
        action_space = graph.get_node_neighbors(self.location) + [self.location]
        best_utility = max([self.utility[(action, prey.location, predator.location)] for action in action_space])
        best_actions = [action for action in action_space if self.utility[(action, prey.location, predator.location)] == best_utility]
        self.location = random.choice(best_actions)

        return None, None 

    def move_debug(self, graph, prey, predator):
        return None, None

