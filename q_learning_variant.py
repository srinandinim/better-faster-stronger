import random
import pickle 
import numpy as np
from copy import deepcopy
from game.game import *
from neuralnetworks.nn import *

class Game_State:
	def __init__(self, agent, prey, predator, beliefs, graph, game, P, found_prey):
		self.agent = agent
		self.prey = prey
		self.predator = predator
		self.beliefs = beliefs
		self.graph = graph
		self.game = game
		self.P = P
		self.found_prey = found_prey

	def retrieve_game(self):
		return self.agent, self.prey, self.predator, self.beliefs, self.graph, self.game, self.P, self.found_prey

def init_new_game():
	game = Game()
	agent_location, prey, predator, graph, _ = game.setup_q_learning()
	beliefs, P = belief_system_init(graph, agent_location)
	return Game_State(agent_location, prey, predator, beliefs, graph, game, P, False)

# HELPER FUNCTIONS ##############################

def save_model(model, error, filename=f"vpartial_dqn_"):
    dirname = "neuralnetworks/trainedmodels/"
    filepath = os.path.dirname(__file__) + dirname + filename
    with open(filepath + str(error) + ".pkl", "wb") as file:
        pickle.dump(model, file)

def load_model(filename="vpartial_dqn_last.pkl"):
    dirname = "neuralnetworks/trainedmodels/"
    filepath = os.path.dirname(__file__) + dirname + filename
    with open(filepath, "rb") as file:
        # print("opening the file")
        model = pickle.load(file)
        # print("model successfully deserialized")
    return model

def index_transform(action):
	'''
	Function to transform the action into an index 
	'''

	return action - 1

def normalize_probs(vector):
	'''
	Function to normalize a vector of probabilities
	'''
	s = sum(vector)
	vector = list(map(lambda x: x / s, vector))
	return vector

def check_prob_sum(su):
	'''
	Function to check if a vector of probabilities adds up to 1
	'''
	if abs(1 - su) < 0.00000000000001:  # 0.000000000000001
		return
	print("BELIEF SYSTEM FAULTY: " + str(su))
	exit()

def normalize_and_check(vector):
	'''
	Routine function to normalize a vector of probabilites and check if sums to 1
	'''
	vector = normalize_probs(vector)
	check_prob_sum(sum(vector))
	return vector

def pick_most_probable_spot(graph, vector):
	'''
	Function to pick the most probable spot in the graph, given a corresponding array
	of probabilities
	'''
	max_prob = max(vector)
	return random.choice([i for i in graph.get_neighbors().keys() if vector[index_transform(i)] == max_prob])

# BELIEF SYSTEM FUNCTIONS ##############################

def belief_system_init(graph, agent_location):
	'''
	Function to initialize a belief system. It returns an array of probabilities
	describing where the prey is and a probability transition matrix. 
	'''

	beliefs = [1/(graph.get_nodes() - 1) for _ in range(0, graph.get_nodes())]
	beliefs[index_transform(agent_location)] = 0 

	# set up transition matrix
	P = [[0 for _ in range(graph.get_nodes())] for _ in range(graph.get_nodes())]
	nbrs = graph.get_neighbors()
	for i in nbrs.keys():
		P[index_transform(i)][index_transform(i)] = 1 / (len(nbrs[i]) + 1)
		for j in nbrs[i]:
			P[index_transform(i)][index_transform(j)] = 1 / (len(nbrs[j]) + 1)
	
	return beliefs, P

def survey_function(graph, beliefs, prey):
	'''
	A function to survey a spot on the graph based on the current belief system. 
	'''

	survey_spot = index_transform(pick_most_probable_spot(graph, beliefs))
	found_prey = False
	if survey_spot == index_transform(prey.location):
		beliefs = [0 for _ in range(graph.get_nodes())]
		beliefs[survey_spot] = 1
		found_prey = True
	else:
		old_survey_spot_prob = beliefs[survey_spot]
		beliefs[survey_spot] = 0
		beliefs = list(map(lambda x: x / (1 - old_survey_spot_prob), beliefs))

	return normalize_and_check(beliefs), found_prey

def update_belief_system_after_agent_moves(beliefs, agent_location, graph, found_prey):
	'''
	Function to update the belief system after the agent moves. If the prey is not found,
	it reinitializes the belief system. If the prey has been found, then it uses Bayes' 
	Law to update. 
	'''

	if not found_prey:
		beliefs = [1/(graph.get_nodes() - 1) for _ in range(0, graph.get_nodes())]
	else:
		old_agent_pos_prob = beliefs[index_transform(agent_location)]
		beliefs = list(map(lambda x: x / (1 - old_agent_pos_prob), beliefs))
	beliefs[index_transform(agent_location)] = 0
	
	return normalize_and_check(beliefs)

def update_belief_system_after_prey_moves(beliefs, P):
	'''
	Function that uses Markov Chain Processes to update the belief system about where
	the prey could have gone. 
	'''
	
	beliefs = list(np.dot(P, beliefs))
	beliefs = normalize_probs(beliefs)

	return normalize_and_check(beliefs)

### DEEP LEARNING FUNCTIONS ###
def init_q_function():
	"""initializes q network for prediction"""
	q_function = NeuralNetwork()
	q_function.add(DenseLinear(150, 150))
	q_function.add(NonLinearity(tanh, tanh_prime))
	q_function.add(DenseLinear(150, 150))
	q_function.add(NonLinearity(tanh, tanh_prime))
	q_function.add(DenseLinear(150, 1))
	q_function.choose_error(mse, mse_prime)
	return q_function

def init_hyperparameters():
	"""initializes hyperparameters in 1 function"""
	alpha, epsilon, delta = 0.0750, 0.0001, 0.1000
	num_games, num_states, batch_size = 256, 128, 128
	return alpha, epsilon, delta, num_games, num_states, batch_size

def init_games(num_games=256):
	""""""
	game_vector = []
	for _ in range(0, num_games):
		game_vector.append(init_new_game())
	return game_vector

def train():

	# set up training NN and target NN
	q_function = init_q_function() 
	q_target = deepcopy(q_function)

	# set up hyperparameters of training
	alpha, epsilon, delta, num_games, num_states, batch_size = init_hyperparameters()
	
	# gets a vector of the games
	game_vector = init_games(num_games)

	# sets up trainig
	avg_loss, batches = float("inf"), 0
	
	








