import json
import os
import game.models.optimalvaluefunction as optimalvaluefunction
import game.utils as utils
import simulation_statistics
import visualize_statistics
from game.graph import Graph


def get_overall_simulation_statistics(wins, losses, timeouts, success_rates, step_counts, found_prey=None, found_pred=None):
    """saves everything to json the relevant information"""

    average_wins = round(sum(wins) / len(wins), 2)
    average_losses = round(sum(losses) / len(losses), 2)
    average_timeouts = round(sum(timeouts) / len(timeouts), 2)
    average_success = round(sum(success_rates) / len(success_rates), 2)
    average_steps = round(sum(step_counts) / len(step_counts),
                          2)

    average_found_prey = round(
        sum(found_prey) / len(found_prey), 2) if found_prey != None else None
    average_found_pred = round(
        sum(found_pred) / len(found_pred), 2) if found_pred != None else None

    statistics = {"avg-wins": average_wins, "avg-losses": average_losses,
                  "avg-timeouts": average_timeouts, "avg-found-prey": average_found_prey,
                  "avg-found-pred": average_found_pred, "avg-success-rates": average_success,
                  "avg-step-counts": average_steps}
    return {"overall": statistics, "success-rates": success_rates, "steps": step_counts}


def save_simulation_statistics(setting, agent, agent_data):
    """
    stores overall statistics to a json file depending on agent setting
    settings:   "complete", "utility_complete", "partial", "utility_partial"
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
    step_counts = []

    for _ in range(30):
        simulation_wins, simulation_losses, simulation_timeouts, simulation_success, simulation_steps = simulation_statistics.agent1(
            100, 50)

        wins.append(simulation_wins)
        losses.append(simulation_losses)
        timeouts.append(simulation_timeouts)
        success_rates.append(simulation_success)
        step_counts.append(simulation_steps)

    agent_data = get_overall_simulation_statistics(
        wins, losses, timeouts, success_rates, step_counts)
    save_simulation_statistics("complete", "agent1", agent_data)

    print(
        f"Agent1: Overall Success Rate: {round(sum(success_rates) / len(success_rates),2)}%")


def labreport_simulation_statistics_agent2():
    """
    runs 100 simulations 30 times and returns the average 
    """
    wins = []
    losses = []
    timeouts = []
    success_rates = []
    step_counts = []

    for _ in range(30):
        simulation_wins, simulation_losses, simulation_timeouts, simulation_success, simulation_steps = simulation_statistics.agent2(
            100, 50)

        wins.append(simulation_wins)
        losses.append(simulation_losses)
        timeouts.append(simulation_timeouts)
        success_rates.append(simulation_success)
        step_counts.append(simulation_steps)

    agent_data = get_overall_simulation_statistics(
        wins, losses, timeouts, success_rates, step_counts)
    save_simulation_statistics("complete", "agent2", agent_data)

    print(
        f"Agent2: Overall Success Rate: {round(sum(success_rates) / len(success_rates),2)}%")


def labreport_simulation_statistics_agent3():
    """
    runs 100 simulations 30 times and returns the average 
    """
    wins = []
    losses = []
    timeouts = []
    success_rates = []
    found_prey = []
    step_counts = []

    for _ in range(30):
        simulation_wins, simulation_losses, simulation_timeouts, simulation_success, simulation_found_prey, simulation_steps = simulation_statistics.agent3(
            100, 50)

        wins.append(simulation_wins)
        losses.append(simulation_losses)
        timeouts.append(simulation_timeouts)
        success_rates.append(simulation_success)
        found_prey.append(simulation_found_prey)
        step_counts.append(simulation_steps)

    agent_data = get_overall_simulation_statistics(
        wins, losses, timeouts, success_rates, step_counts, found_prey=found_prey)
    save_simulation_statistics("partial", "agent3", agent_data)

    print(
        f"Agent3: Overall Success Rate: {round(sum(success_rates) / len(success_rates),2)}%")


def labreport_simulation_statistics_agent4():
    """
    runs 100 simulations 30 times and returns the average 
    """
    wins = []
    losses = []
    timeouts = []
    success_rates = []
    found_prey = []
    step_counts = []

    for _ in range(30):
        simulation_wins, simulation_losses, simulation_timeouts, simulation_success, simulation_found_prey, simulation_steps = simulation_statistics.agent4(
            100, 50)

        wins.append(simulation_wins)
        losses.append(simulation_losses)
        timeouts.append(simulation_timeouts)
        success_rates.append(simulation_success)
        found_prey.append(simulation_found_prey)
        step_counts.append(simulation_steps)

    agent_data = get_overall_simulation_statistics(
        wins, losses, timeouts, success_rates, step_counts, found_prey=found_prey)
    save_simulation_statistics("partial", "agent4", agent_data)

    print(
        f"Agent4: Overall Success Rate: {round(sum(success_rates) / len(success_rates),2)}%")


def labreport_simulation_statistics_agent1_rl():
    """
    runs 100 simulations 30 times and returns the average 
    """
    wins = []
    losses = []
    timeouts = []
    success_rates = []
    step_counts = []

    for _ in range(30):
        simulation_wins, simulation_losses, simulation_timeouts, simulation_success, simulation_steps = simulation_statistics.agent1rl(
            100, 50)

        wins.append(simulation_wins)
        losses.append(simulation_losses)
        timeouts.append(simulation_timeouts)
        success_rates.append(simulation_success)
        step_counts.append(simulation_steps)

    agent_data = get_overall_simulation_statistics(
        wins, losses, timeouts, success_rates, step_counts)
    save_simulation_statistics("complete", "agent1rl", agent_data)
    save_simulation_statistics("utility_complete", "agent1rl", agent_data)

    print(
        f"Agent1RL: Overall Success Rate: {round(sum(success_rates) / len(success_rates),2)}%")


def labreport_simulation_statistics_agent1_rl_nn():
    """
    runs 100 simulations 30 times and returns the average 
    """
    wins = []
    losses = []
    timeouts = []
    success_rates = []
    step_counts = []

    for _ in range(30):
        simulation_wins, simulation_losses, simulation_timeouts, simulation_success, simulation_steps = simulation_statistics.agent1rlnn(
            100, 50)

        wins.append(simulation_wins)
        losses.append(simulation_losses)
        timeouts.append(simulation_timeouts)
        success_rates.append(simulation_success)
        step_counts.append(simulation_steps)

    agent_data = get_overall_simulation_statistics(
        wins, losses, timeouts, success_rates, step_counts)
    save_simulation_statistics("utility_complete", "agent1rlnn", agent_data)

    print(
        f"Agent1RLNN: Overall Success Rate: {round(sum(success_rates) / len(success_rates),2)}%")


def labreport_simulation_statistics_agent3_rl():
    """
    runs 100 simulations 30 times and returns the average 
    """
    wins = []
    losses = []
    timeouts = []
    success_rates = []
    step_counts = []

    for _ in range(30):
        simulation_wins, simulation_losses, simulation_timeouts, simulation_success, simulation_steps = simulation_statistics.agent3rl(
            100, 50)

        wins.append(simulation_wins)
        losses.append(simulation_losses)
        timeouts.append(simulation_timeouts)
        success_rates.append(simulation_success)
        step_counts.append(simulation_steps)

    agent_data = get_overall_simulation_statistics(
        wins, losses, timeouts, success_rates, step_counts)
    save_simulation_statistics("partial", "agent3rl", agent_data)
    save_simulation_statistics("utility_partial", "agent3rl", agent_data)

    print(
        f"Agent3RL: Overall Success Rate: {round(sum(success_rates) / len(success_rates),2)}%")


def labreport_simulation_statistics_agent3_rl_nn():
    """
    runs 100 simulations 30 times and returns the average 
    """
    wins = []
    losses = []
    timeouts = []
    success_rates = []
    step_counts = []

    for _ in range(30):
        simulation_wins, simulation_losses, simulation_timeouts, simulation_success, simulation_steps = simulation_statistics.agent3rlnn(
            100, 50)

        wins.append(simulation_wins)
        losses.append(simulation_losses)
        timeouts.append(simulation_timeouts)
        success_rates.append(simulation_success)
        step_counts.append(simulation_steps)

    agent_data = get_overall_simulation_statistics(
        wins, losses, timeouts, success_rates, step_counts)
    save_simulation_statistics("utility_partial", "agent3rlnn", agent_data)

    print(
        f"Agent3RLNN: Overall Success Rate: {round(sum(success_rates) / len(success_rates),2)}%")


def calculate_utility_values(filename="GAME_GRAPH.json"):
    game_graph = Graph(nbrs=utils.retrieve_graph(filename))
    shortest_distances = optimalvaluefunction.agent_to_pred_distances(
        game_graph)
    print(shortest_distances)
    ksweeps, u0 = optimalvaluefunction.calculate_optimal_values(
        game_graph, shortest_distances, 0.001)

    print(u0)
    print(ksweeps)


if __name__ == "__main__":
    # calculate_utility_values()

    # labreport_simulation_statistics_agent1()
    # labreport_simulation_statistics_agent2()
    # labreport_simulation_statistics_agent1_rl()
    visualize_statistics.visualize_success_rates(
        "data/", "simulation_statistics_complete.json")
    visualize_statistics.visualize_step_counts(
        "data/", "simulation_statistics_complete.json")

    # labreport_simulation_statistics_agent1_rl_nn()
    visualize_statistics.visualize_success_rates(
        "data/", "simulation_statistics_utility_complete.json")
    visualize_statistics.visualize_step_counts(
        "data/", "simulation_statistics_utility_complete.json")

    # labreport_simulation_statistics_agent3()
    # labreport_simulation_statistics_agent4()
    # labreport_simulation_statistics_agent3_rl()
    visualize_statistics.visualize_success_rates(
        "data/", "simulation_statistics_partial.json")
    visualize_statistics.visualize_step_counts(
        "data/", "simulation_statistics_partial.json")

    # labreport_simulation_statistics_agent3_rl_nn()
    visualize_statistics.visualize_success_rates(
        "data/", "simulation_statistics_utility_partial.json")
    visualize_statistics.visualize_step_counts(
        "data/", "simulation_statistics_utility_partial.json")
