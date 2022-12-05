import pickle
import random
from .agent import Agent


class Agent3RL(Agent):
    def __init__(self, location):
        # initialize agent location
        super().__init__(location)
        self.utility = pickle.load(open("game/pickles/u0.pickle", "rb"))

    def current_reward(self, new_location, predator):
        return -1 if abs(new_location - predator.location) % 50 > 1 else -float("inf")

    def partial_utility(self, graph, new_location, predator):
        partial_reward = 0
        for prey_location in range(1, graph.get_nodes() + 1):
            if new_location != prey_location:
                partial_reward += 1/49 * \
                    self.utility[(new_location, prey_location,
                                  predator.location)]

        return partial_reward

    def move(self, graph, prey, predator):
        """
        updates location based on assignment specifications given
        """
        action_space = graph.get_node_neighbors(
            self.location)
        best_utility = max([self.current_reward(
            action, predator) + self.partial_utility(graph, action, predator) for action in action_space])
        best_actions = [action for action in action_space if self.current_reward(
            action, predator) + self.partial_utility(graph, action, predator) == best_utility]
        self.location = random.choice(best_actions)

        return None, None

    def move_debug(self, graph, prey, predator):
        """
        updates location based on assignment specifications given
        """
        action_space = graph.get_node_neighbors(
            self.location)
        for action in action_space:
            print(action, self.current_reward(action, predator) + self.utility[(
                action, prey.location, predator.location)])

        best_utility = max([self.current_reward(
            action, predator) + self.partial_utility(graph, action, predator) for action in action_space])
        best_actions = [action for action in action_space if self.current_reward(
            action, predator) + self.partial_utility(graph, action, predator) == best_utility]
        self.location = random.choice(best_actions)

        print(f"The action was to do {self.location}")

        return None, None
