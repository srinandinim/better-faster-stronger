import json
import os

import game.models.optimalvaluefunction as optimalvaluefunction
import game.utils as utils
import simulation_statistics as simulation_statistics
from game.game import Game
from game.graph import Graph


def get_overall_simulation_statistics(wins, losses, timeouts, success_rates, found_prey=None, found_pred=None):
    average_wins = round(sum(wins) / len(wins), 2)
    average_losses = round(sum(losses) / len(losses), 2)
    average_timeouts = round(sum(timeouts) / len(timeouts), 2)
    average_success = round(sum(success_rates) / len(success_rates), 2)

    average_found_prey = round(
        sum(found_prey) / len(found_prey), 2) if found_prey != None else None
    average_found_pred = round(
        sum(found_pred) / len(found_pred), 2) if found_pred != None else None

    statistics = {"avg-wins": average_wins, "avg-losses": average_losses,
                  "avg-timeouts": average_timeouts, "avg-found-prey": average_found_prey,
                  "avg-found-pred": average_found_pred, "avg-success-rates": average_success}
    return {"overall": statistics, "success-rates": success_rates}


def save_simulation_statistics(setting, agent, agent_data):
    """
    stores overall statistics to a json file depending on agent setting
    settings:   "complete", "partial-prey", "partial-pred", "combined-partial", 
                "combined-partial-defective", "combined-partial-defective-updated"
    """
    dirname = "data/"
    if not os.path.exists(os.path.dirname(dirname)):
        os.makedirs(os.path.dirname(dirname))

    filename = f'simulation_statistics_{setting}.json'
    filepath = dirname + filename
    if os.path.exists(filepath):
        with open(filepath, "r") as fp:
            data = json.load(fp)
    else:
        data = {}

    data[agent] = agent_data

    with open(filepath, "w") as fp:
        json.dump(data, fp)


def labreport_simulation_statistics_agent1():
    """
    runs 100 simulations 30 times and returns the average 
    """
    wins = []
    losses = []
    timeouts = []
    success_rates = []

    for _ in range(30):
        simulation_wins, simulation_losses, simulation_timeouts, simulation_success = simulation_statistics.agent1(
            100, 50)

        wins.append(simulation_wins)
        losses.append(simulation_losses)
        timeouts.append(simulation_timeouts)
        success_rates.append(simulation_success)

    agent_data = get_overall_simulation_statistics(
        wins, losses, timeouts, success_rates)
    save_simulation_statistics("complete", "agent1", agent_data)

    print(
        f"Agent1: Overall Success Rate: {round(sum(success_rates) / len(success_rates),2)}%")


def labreport_simulation_statistics_agent1_rl():
    """
    runs 100 simulations 30 times and returns the average 
    """
    wins = []
    losses = []
    timeouts = []
    success_rates = []

    for _ in range(30):
        simulation_wins, simulation_losses, simulation_timeouts, simulation_success = simulation_statistics.agent1rl(
            100, 50)

        wins.append(simulation_wins)
        losses.append(simulation_losses)
        timeouts.append(simulation_timeouts)
        success_rates.append(simulation_success)

    agent_data = get_overall_simulation_statistics(
        wins, losses, timeouts, success_rates)
    save_simulation_statistics("complete", "agent1rl", agent_data)

    print(
        f"Agent1RL: Overall Success Rate: {round(sum(success_rates) / len(success_rates),2)}%")


def labreport_simulation_statistics_agent3_rl():
    """
    runs 100 simulations 30 times and returns the average 
    """
    wins = []
    losses = []
    timeouts = []
    success_rates = []

    for _ in range(30):
        simulation_wins, simulation_losses, simulation_timeouts, simulation_success = simulation_statistics.agent3rl(
            100, 50)

        wins.append(simulation_wins)
        losses.append(simulation_losses)
        timeouts.append(simulation_timeouts)
        success_rates.append(simulation_success)

    agent_data = get_overall_simulation_statistics(
        wins, losses, timeouts, success_rates)
    save_simulation_statistics("partial-prey", "agent3rl", agent_data)

    print(
        f"Agent3RL: Overall Success Rate: {round(sum(success_rates) / len(success_rates),2)}%")


def calculate_utility_values(filename="GAME_GRAPH.json"):
    game_graph = Graph(nbrs=utils.retrieve_graph(filename))
    shortest_distances = optimalvaluefunction.agent_to_pred_distances(game_graph)
    print(shortest_distances)
    ksweeps, u0 = optimalvaluefunction.calculate_optimal_values(game_graph, shortest_distances, 0.001)

    print(u0)
    print(ksweeps)


if __name__ == "__main__":
    # utils.save_graph(graph=Graph(nodes=6), filename="SMALL_GRAPH.json")
    # calculate_utility_values(filename="SMALL_GRAPH.json")
    
    # labreport_simulation_statistics_agent1()
    labreport_simulation_statistics_agent1_rl()
    # labreport_simulation_statistics_agent3_rl()

    # game = Game()
    # game_success = game.run_agent_1_rl_debug()

    
