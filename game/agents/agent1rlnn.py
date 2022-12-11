from game.models.optimalvaluefunction import (agent_to_pred_distances,
                                              get_future_reward_prediction)
from neuralnetworks.utils import load_model_for_agent
from .agent import Agent


class Agent1RLNN(Agent):
    def __init__(self, graph, location):
        # initialize agent location
        super().__init__(location)
        self.vcomplete_model = load_model_for_agent(
            filename="OPTIMAL_VCOMPLETE_MODEL.pkl")
        self.shortest_distances = agent_to_pred_distances(graph)

    def move(self, graph, prey, predator):
        """
        moves based on the action in the agent's action space that is predicted to have the greatest utility
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
                current_reward = -1 + get_future_reward_prediction(
                    graph, action, prey.location, predator.location, self.shortest_distances, self.vcomplete_model)

            if current_reward >= best_reward:
                best_reward = current_reward
                best_action = action

        self.location = best_action
        return None, None

    def move_debug(self, graph, prey, predator):
        """
        debug version of move
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
                current_reward = -1 + get_future_reward_prediction(
                    graph, action, prey.location, predator.location, self.shortest_distances, self.vcomplete_model)
            if current_reward >= best_reward:
                best_reward = current_reward
                best_action = action

        self.location = best_action
        print(f"The action was to do {self.location}")

        return None, None
