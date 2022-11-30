import pickle 
from .agent import Agent
import random

class Agent1RL(Agent):
    def __init__(self, location):
        # initialize agent location
        super().__init__(location)
        self.utility = pickle.load(open("game/agents/u0.pickle", "rb"))

    def move(self, graph, prey, predator):
        """
        updates location based on assignment specifications given
        """
        action_value = dict() 
        action_space = graph.get_node_neighbors(self.location) + [self.location]
        best_utility = max([self.utility[(action, prey.location, predator.location)] for action in action_space])
        best_actions = [action for action in action_space if self.utility[(action, prey.location, predator.location)] == best_utility]
        self.location = random.choice(best_actions)
        
        # for action in action_space: 
        #     action_value[action] = optimal_value_function[(action, prey.location, predator.location)]
        
        # #print(action_value)
        
        # max_value = max(action_value.values())
        
        # agent_action = self.location 
        # for action in action_space: 
        #     if action_value[action] == max_value:
        #         agent_action = action 
        
        # self.location = agent_action 
        return None, None 

    def move_debug(self, graph, prey, predator):
        return None, None

