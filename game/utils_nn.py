import pickle
import random
import numpy as np


# takes pickled binary and generates training data CSV
def create_supervised_training_data(utilities, graph_size=50):
    """
    Given the utilities and the state, we create a supervised learning training dataset. 
    Observe that we shuffle the dataset so that its more suitable for training. 

    Dataset Format: y, x1, x2, ..., x150
    """
    yx = []
    for agent_loc in range(1, graph_size+1):
        for prey_loc in range(1, graph_size+1):
            for pred_loc in range(1, graph_size+1):
                state = (agent_loc, prey_loc, pred_loc)
                state_utility = utilities[state]
                yx.append([state_utility] + vectorize_state(state))
    random.shuffle(yx)
    return yx


def create_training_dataset_csv(utilities):
    """
    Dump the information from the input and output targets into a CSV file.
    """
    dataset = create_supervised_training_data(utilities=utilities)
    csv = np.asarray(dataset)
    np.savetxt("models/data/supervised_dataset_1hotvector_vcomplete.csv",
               csv, delimiter=",", fmt='%.6f')

# takes pickled binary and generates training data CSV


def create_supervised_training_data_normalstates(utilities, graph_size=50):
    """
    Given the utilities and the state, we create a supervised learning training dataset. 
    Observe that we shuffle the dataset so that its more suitable for training. 

    Dataset Format: y, x1, x2, ..., x150
    """
    yx = []
    for agent_loc in range(1, graph_size+1):
        for prey_loc in range(1, graph_size+1):
            for pred_loc in range(1, graph_size+1):
                state = (agent_loc, prey_loc, pred_loc)
                state_utility = utilities[state]
                yx.append([state_utility, agent_loc, prey_loc, pred_loc])
    random.shuffle(yx)
    return yx


def create_training_dataset_normal_csv(utilities):
    """
    Dump the information from the input and output targets into a CSV file.
    """
    dataset = create_supervised_training_data_normalstates(utilities=utilities)
    csv = np.asarray(dataset)
    np.savetxt("models/data/supervised_dataset_normalstates_vcomplete.csv",
               csv, delimiter=",", fmt='%.6f')


if __name__ == "__main__":
    OPTIMAL_COMPLETE_UTILITIES = pickle.load(
        open("game/pickles/OPTIMAL_U*.pickle", "rb"))
    create_training_dataset_csv(OPTIMAL_COMPLETE_UTILITIES)
    # create_training_dataset_normal_csv(OPTIMAL_COMPLETE_UTILITIES)
