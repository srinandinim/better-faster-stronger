import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np


def visualize_success_rates(dirname, filename):
    """
    plot the simulation success rates and error bars
    """
    # get the agent setting
    setting = filename[:-5].split("_")[-1]

    # read the file
    filepath = dirname + filename
    with open(filepath, 'r') as fp:
        data = json.load(fp)

    # get the mean and standard deviation of each agent
    agents = []
    means = []
    stds = []
    for agent, value in data.items():
        agent_split = re.split(r'(\d+)', agent)
        agents.append(
            f'{agent_split[0].capitalize()} {agent_split[1]} {agent_split[2].upper()}')

        success_rates = value['success-rates']
        np_success_rates = np.array(success_rates)
        means.append(np.mean(np_success_rates))
        stds.append(2 * np.std(np_success_rates))  # 2 standard deviations

    # create the bar graph with error bars
    x_pos = np.arange(len(agents))
    colors = ["cornflowerblue", "mediumaquamarine",
              "cadetblue", "lightslategrey"]

    _, axes = plt.subplots()
    axes.bar(x_pos, means, color=colors, yerr=stds,
             align='center', ecolor='black', capsize=10)

    plt.title(f'{setting.title()} Agents\' Average Success Rates')
    axes.set_xticks(x_pos)
    axes.set_xticklabels(agents)
    plt.gca().set_ylim(bottom=0, top=105)
    plt.ylabel('Success Rate (%)')

    # save the bar graph
    if not os.path.exists(os.path.dirname(dirname)):
        os.makedirs(os.path.dirname(dirname))

    # get the filename
    settings = filename[:-5].split("_")
    saved_filename = '_'.join(settings[2:])

    plot_name = "{}visualize_success_statistics_{}.png".format(dirname, saved_filename)
    plt.savefig(plot_name, bbox_inches='tight')

    # show the bar graph
    # plt.show()

    return 1

def visualize_step_counts(dirname, filename):
    """
    plot the simulation success rates and error bars
    """
    # get the agent setting
    setting = filename[:-5].split("_")[-1]

    # read the file
    filepath = dirname + filename
    with open(filepath, 'r') as fp:
        data = json.load(fp)

    # get the mean and standard deviation of each agent
    agents = []
    means = []
    stds = []
    for agent, value in data.items():
        agent_split = re.split(r'(\d+)', agent)
        agents.append(
            f'{agent_split[0].capitalize()} {agent_split[1]} {agent_split[2].upper()}')

        steps = value['steps']
        np_steps = np.array(steps)
        means.append(np.mean(np_steps))
        stds.append(2 * np.std(np_steps))  # 2 standard deviations

    # create the bar graph with error bars
    x_pos = np.arange(len(agents))
    colors = ["cornflowerblue", "mediumaquamarine",
              "cadetblue", "lightslategrey"]

    _, axes = plt.subplots()
    axes.bar(x_pos, means, color=colors,
             align='center', ecolor='black', capsize=10)

    plt.title(f'{setting.title()} Agents\' Average Steps to Catch Prey')
    axes.set_xticks(x_pos)
    axes.set_xticklabels(agents)
    # plt.gca().set_ylim(bottom=0, top=105)
    plt.ylabel('Average Steps')

    # save the bar graph
    if not os.path.exists(os.path.dirname(dirname)):
        os.makedirs(os.path.dirname(dirname))

    # get the filename
    settings = filename[:-5].split("_")
    saved_filename = '_'.join(settings[2:])

    plot_name = "{}visualize_steps_statistics_{}.png".format(dirname, saved_filename)
    plt.savefig(plot_name, bbox_inches='tight')

    # show the bar graph
    # plt.show()

    return 1
