import random
import matplotlib.pyplot as plt
import networkx as nx
import game.utils as utils
from .agents.agent1 import Agent1
from .agents.agent1rl import Agent1RL
from .agents.agent1rlnn import Agent1RLNN
from .agents.agent3rl import Agent3RL
from .graph import Graph
from .predator import Predator
from .prey import Prey


class Game:
    def __init__(self, nodes=50, timeout=1000):
        # initializes the graph on which agents/prey/predator play
        nbrs = utils.retrieve_graph()
        self.graph = Graph(nbrs=nbrs)

        # initializes prey location to be random from nodes 1...50
        self.prey = Prey(random.randint(1, self.graph.get_nodes()))

        # determines the predator location which will be used to create the specific predator
        self.predator = None
        self.predator_location = random.randint(1, self.graph.get_nodes())

        # agent initializes randomly to any spot that is not occupied by predator/prey
        occupied_s = min(self.prey.location, self.predator_location)
        occupied_l = max(self.prey.location, self.predator_location)
        agent_location_options = list(range(1, occupied_s)) + list(range(
            occupied_s+1, occupied_l)) + list(range(occupied_l+1, self.graph.get_nodes() + 1))
        self.agent_starting_location = random.choice(agent_location_options)

        # initializes an agent which allows us to call the relevant agent.
        self.agent = None

        # stores the trajectories of the agent/predator/prey
        self.agent_trajectories = [self.agent_starting_location]
        self.prey_trajectories = [self.prey.location]
        self.predator_trajectories = [self.predator_location]

        # initializes the number of steps before timing out
        self.timeout = timeout

        # initializes the number of steps the agent took
        self.steps = 0

    def step_return_values(self, status, found_prey, found_pred):
        """
        returns status & found_prey/found_pred as a percentage of total moves
        """
        if found_prey is not None and found_pred is not None:
            return status, found_prey/self.steps * 100, found_pred/self.steps * 100
        elif found_prey is not None:
            return status, found_prey/self.steps * 100, found_pred
        elif found_pred is not None:
            return status, found_prey, found_pred/self.steps * 100
        else:
            return status, found_prey, found_pred

    def step(self):
        """
        moves the agent, prey, and predator one step

        returns
        * 1 if agent wins
        * 0 if game in progress
        * -1 if agent looses 

        also returns number of times agent knew the exact location of the prey and the pred in the partial information settings
        """
        self.steps = self.steps + 1

        found_prey, found_pred = self.agent.move(
            self.graph, self.prey, self.predator)
        self.agent_trajectories.append(self.agent.location)
        if self.agent.location == self.prey.location:
            return self.step_return_values(1, found_prey, found_pred)
        if self.agent.location == self.predator.location:
            return self.step_return_values(-1, found_prey, found_pred)

        self.prey.move(self.graph)
        self.prey_trajectories.append(self.prey.location)
        if self.agent.location == self.prey.location:
            return self.step_return_values(1, found_prey, found_pred)

        self.predator.move(self.graph, self.agent)
        self.predator_trajectories.append(self.predator.location)
        if self.agent.location == self.predator.location:
            return self.step_return_values(-1, found_prey, found_pred)

        return self.step_return_values(0, found_prey, found_pred)

    def step_debug(self):
        """
        -- debug method --
        moves the agent, prey, and predator one step

        returns
        * 1 if agent wins
        * 0 if game in progress
        * -1 if agent looses 
        """
        # print(f"THE NEIGHBORS ARE{self.graph.nbrs}")
        print("\nNEW RUN")
        print(f"prey is at {self.prey.location}")
        print(f"predator is at {self.predator.location}")
        print(f"agent is at {self.agent.location}")
        found_prey, found_pred = self.agent.move_debug(
            self.graph, self.prey, self.predator)
        self.agent_trajectories.append(self.agent.location)
        if self.agent.location == self.prey.location:
            return 1, found_prey, found_pred
        if self.agent.location == self.predator.location:
            return -1, found_prey, found_pred

        self.prey.move(self.graph)
        self.prey_trajectories.append(self.prey.location)
        if self.agent.location == self.prey.location:
            return 1, found_prey, found_pred

        self.predator.move(self.graph, self.agent)
        self.predator_trajectories.append(self.predator.location)
        if self.agent.location == self.predator.location:
            return -1, found_prey, found_pred

        return 0, found_prey, found_pred

    def run_agent_1_rl(self):
        self.predator = Predator(self.predator_location)
        self.agent = Agent1RL(self.graph, self.agent_starting_location)

        status = 0
        step_count = 0

        while status == 0 and step_count < self.timeout:
            status, _, _ = self.step()
            step_count = step_count + 1

        # agent timed out
        if status == 0:
            status = -2

        return status

    def run_agent_1_rl_debug(self):
        self.predator = Predator(self.predator_location)
        self.agent = Agent1RL(self.graph, self.agent_starting_location)
        self.visualize_graph()

        status = 0
        step_count = 0

        while status == 0 and step_count < self.timeout:
            status, _, _ = self.step_debug()
            step_count = step_count + 1
            self.visualize_graph()

        # self.visualize_graph_video()

        # agent timed out
        if status == 0:
            status = -2

        return status

    def run_agent_1_rl_nn(self):
        self.predator = Predator(self.predator_location)
        self.agent = Agent1RLNN(self.graph, self.agent_starting_location)

        status = 0
        step_count = 0

        while status == 0 and step_count < self.timeout:
            status, _, _ = self.step()
            step_count = step_count + 1

        # agent timed out
        if status == 0:
            status = -2

        return status

    def run_agent_1_rl_nn_debug(self):
        self.predator = Predator(self.predator_location)
        self.agent = Agent1RLNN(self.graph, self.agent_starting_location)
        self.visualize_graph()

        status = 0
        step_count = 0

        while status == 0 and step_count < self.timeout:
            status, _, _ = self.step_debug()
            step_count = step_count + 1
            self.visualize_graph()

        self.visualize_graph_video()

        # agent timed out
        if status == 0:
            status = -2

        return status

    def run_agent_3_rl(self):
        self.predator = Predator(self.predator_location)
        self.agent = Agent3RL(self.agent_starting_location)

        status = 0
        step_count = 0

        while status == 0 and step_count < self.timeout:
            status, _, _ = self.step()
            step_count = step_count + 1

        # agent timed out
        if status == 0:
            status = -2

        return status

    def run_agent_3_rl_debug(self):
        self.predator = Predator(self.predator_location)
        self.agent = Agent3RL(self.agent_starting_location)
        self.visualize_graph()

        status = 0
        step_count = 0

        while status == 0 and step_count < self.timeout:
            status, _, _ = self.step_debug()
            step_count = step_count + 1
            self.visualize_graph()

        self.visualize_graph_video()

        # agent timed out
        if status == 0:
            status = -2

        return status

    def visualize_graph_color_map(self):
        """
        grey: unoccupied node
        green: node of prey 
        yellow: node of agent 
        pink: node of prey 
        """
        color_map = ["grey" for _ in self.graph.get_neighbors()]
        color_map[self.prey.location - 1] = "yellowgreen"
        color_map[self.predator.location - 1] = "lightcoral"

        if self.agent is not None:
            color_map[self.agent.location - 1] = "gold"

        return color_map

    def visualize_graph(self, fn='environment.png'):
        """visualizes nodes and their edges with labels in non-circular layout"""
        plt.rcParams['figure.figsize'] = [8, 6]
        G = nx.from_dict_of_lists(self.graph.get_neighbors())
        my_pos = nx.spring_layout(G, seed=100)
        nx.draw(G, pos=my_pos,
                node_color=self.visualize_graph_color_map(), with_labels=True)

        figure_text = "Agent: {}, Prey: {}, Predator: {}".format(
            self.agent.location, self.prey.location, self.predator.location)
        plt.figtext(0.5, 0.05, figure_text, ha="center", fontsize=10)

        trajectories = f"Agent: {self.agent_trajectories}\nPrey: {self.prey_trajectories}\nPredator: {self.predator_trajectories}"
        plt.figtext(0.1, 0.1, trajectories, ha="left", fontsize=8)

        plt.show()

    def visualize_graph_video(self, fn='sample.mp4'):
        """visualizes nodes and their edges with labels in non-circular layout as a video"""
        import os
        plt.rcParams['figure.figsize'] = [16, 10]

        dirname = "videos/"

        if not os.path.exists(os.path.dirname(dirname)):
            os.makedirs(os.path.dirname(dirname))

        filepath = dirname + fn
        if os.path.exists(filepath):
            os.remove(filepath)

        G = nx.from_dict_of_lists(self.graph.get_neighbors())
        my_pos = nx.spring_layout(G, seed=100)

        for i in range(len(self.agent_trajectories)):
            plt.clf()  # make sure we clear any old stuff
            agent_location = self.agent_trajectories[min(
                i, len(self.agent_trajectories)-1)]
            prey_location = self.prey_trajectories[min(
                i, len(self.prey_trajectories)-1)]
            predator_location = self.predator_trajectories[min(
                i, len(self.predator_trajectories)-1)]

            color_map = ["grey" for _ in self.graph.get_neighbors()]
            color_map[prey_location - 1] = "yellowgreen"
            color_map[predator_location - 1] = "lightcoral"
            color_map[agent_location - 1] = "gold"

            nx.draw(G, pos=my_pos, node_color=color_map, with_labels=True)

            figure_text = "Agent: {}, Prey: {}, Predator: {}".format(
                agent_location, prey_location, predator_location)
            plt.figtext(0.5, 0.05, figure_text, ha="center", fontsize=10)

            plt.savefig('figure' + str(i) + '.png')  # save this figure to disk

        # now combine all of the figures into a video
        os.system('ffmpeg -r 3 -i figure%d.png -vcodec mpeg4 -y '+filepath)
        print("A video showing the agent's traversal is ready to view. Opening...")
        os.system('open '+filepath)

        # clean up the environment a bit
        for i in range(len(self.agent_trajectories)):
            os.remove('figure' + str(i) + '.png')

    def visualize_graph_circle(self):
        """visualizes nodes and their edges with labels in a circular layout"""
        nx.draw_networkx(nx.Graph(self.graph.get_neighbors()), pos=nx.circular_layout(
            nx.Graph(self.graph.get_neighbors())), node_color=self.visualize_graph_color_map(), node_size=50, with_labels=True)
        plt.show()
