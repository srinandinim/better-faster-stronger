import pickle
import random
import numpy as np 
import nn
import nn_engine
from .agent import Agent

class Agent1RLNN(Agent):
    def __init__(self, location):
        # initialize agent location
        super().__init__(location)

    def current_reward(self, new_location, predator):
        return -1 if abs(new_location - predator.location) % 50 > 1 else -float("inf")

    def move(self, graph, prey, predator):
        """
        updates location based on assignment specifications given
        """
        action_space = graph.get_node_neighbors(self.location)

        all_utilities = dict() 
        for action in action_space:
            # represent s as the approrpiate vector for the moel 
            x = np.asarray(nn.vectorize_state( (action, prey.location, predator.location) ), dtype="float32")
            x = x.reshape(1, x.shape[0])

            # make an output prediction for a particular input
            yhat = nn_engine.model_predict_completeinfo(x)

            # add the utility to dict
            all_utilities[action] = yhat 
        
        max_utility = max(all_utilities.values())
        best_action = self.location 
        for action, value in all_utilities.items():
            if value == max_utility:
                best_action = action 
        self.location = best_action
    
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

        best_utility = max([self.current_reward(action, predator) + self.utility[(
            action, prey.location, predator.location)] for action in action_space])
        best_actions = [action for action in action_space if self.current_reward(action, predator) + self.utility[(
            action, prey.location, predator.location)] == best_utility]
        self.location = random.choice(best_actions)

        print(f"The action was to do {self.location}")

        return None, None


