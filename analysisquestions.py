import pickle
from game.game import Game
from game.graph import Graph
from game.utils import retrieve_graph
import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(graph_filename="GAME_GRAPH.json"):
    """visualizes nodes and their edges with labels in non-circular layout"""
    graph = Graph(nbrs=retrieve_graph(filename=graph_filename))

    plt.rcParams['figure.figsize'] = [8, 6]
    G = nx.from_dict_of_lists(graph.get_neighbors())
    my_pos = nx.spring_layout(G, seed=100)
    nx.draw(G, pos=my_pos,
            node_color=["grey" for _ in graph.get_neighbors()], with_labels=True)

    figname = "graphs/" + graph_filename.split(".")[0] + ".png"
    plt.savefig(figname)

    plt.show()

if __name__ == "__main__":
    # find the state with the largest possible finite value of u*
    # load the pickle file with the optimal utilities
    OPTIMAL_UTILITIES_USTAR = pickle.load(open("game/pickles/OPTIMAL_U*.pickle", "rb"))

    min_ustar_value = 0
    for utility in OPTIMAL_UTILITIES_USTAR.values():
        if utility != -float("inf"):
            min_ustar_value = min(utility, min_ustar_value)

    largest_states = dict()
    for state, utility in OPTIMAL_UTILITIES_USTAR.items():
        if utility == min_ustar_value:
            largest_states[state] = utility

    # prints largest finite state and value
    print(largest_states)

    # generates image of largest possible finite state and value
    g1 = Game()
    g1.run_agent_1_rl_debug()

    # for graph generation
    visualize_graph()
