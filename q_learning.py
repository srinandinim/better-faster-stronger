import pickle 
from neuralnetworks.nn import *
from game.game import *
from game.predator import *
from game.prey import *
import random
import numpy as np
from game.utils import *

class Game_State:
	def __init__(self, agent, prey, predator, beliefs, found_prey):
		self.agent = agent
		self.prey = prey
		self.predator = predator
		self.beliefs = beliefs
		self.found_prey = found_prey

	def retrieve_game(self):
		return self.agent, self.prey, self.predator, self.beliefs, self.found_prey

def init_new_game(graph):
	# initializes prey location to be random from nodes 1...50
	prey = Prey(random.randint(1, graph.get_nodes()))

	# determines the predator location which will be used to create the specific predator
	predator = Predator(random.randint(1, graph.get_nodes()))

	# agent initializes randomly to any spot that is not occupied by predator/prey
	occupied_s = min(prey.location, predator.location)
	occupied_l = max(prey.location, predator.location)
	agent_location_options = list(range(1, occupied_s)) + list(range(
		occupied_s+1, occupied_l)) + list(range(occupied_l+1, graph.get_nodes() + 1))
	agent_location = random.choice(agent_location_options)

	beliefs = belief_system_init(graph, agent_location)
	return Game_State(agent_location, prey, predator, beliefs, False)




# HELPER FUNCTIONS ##############################

def load_model(filename="OPTIMAL_VPARTIAL_MODEL.pkl"):
    dirname = "neuralnetworks/trainedmodels/"
    filepath = os.path.dirname(__file__) + dirname + filename
    with open(filepath, "rb") as file:
        # print("opening the file")
        model = pickle.load(file)
        # print("model successfully deserialized")
    return model

def calculate_shortest_distances(graph, source, goals):
	'''
	Function to calculate all of the shortest distances from the source to a list of goals
	@param:graph - the graph object to operate on
	@param:source - the source node to calculae distances from
	@param:goals - a list of goal nodes
	@return the dictionary of shortest distances from the source to the goal; each key is 
	a tuple structured as (source, goal) and the value is the shortest distance from the
	source to the goal
	'''

	queue = [source]
	visited = set()
	shortest_distances = {}
	lengths = {source: 0}
	completed = []

	while not len(queue) == 0 and not len(goals) == len(completed):

		cur = queue.pop(0)

		if cur in visited:
			continue

		if cur in goals:
			shortest_distances[(source, cur)] = lengths[cur]
			completed.append(cur)

		nbrs = graph.get_node_neighbors(cur)

		for nbr in nbrs:
			if nbr not in visited:
				queue.append(nbr)
				if nbr not in lengths.keys():
					lengths[nbr] = lengths[cur] + 1

		visited.add(cur)
	return shortest_distances


def agent_to_pred_distances(graph):
	'''
	Function to calculate all of the shortest distances between every pair of nodes in the graph
	@param:graph - the graph object to operate on
	@return the dictionary of shortest distances from the source to the goal; each key is 
	a tuple structured as (node1, node2) and the value is the shortest distance from
	node1 to node2
	'''

	agent_to_pred_dists = dict()
	for i in range(1, graph.get_nodes() + 1):
		agent_to_pred_dists.update(calculate_shortest_distances(
			graph, i, list(graph.get_neighbors().keys())))
	return agent_to_pred_dists


def optimal_pred_moves(graph, agent_loc, pred_loc, shortest_distances):
	"""
	Function to return a list containing the best moves the predator can make to reach the agent
	the quickest. 
	@param graph - the graph the function operates on
	@param agent_loc - the location of the agent
	@param pred_loc - the location of the predator
	@param shortesT_distances - a dictionary containing the shortest distances between every pair of nodes in 
	the graph
	@return a list containing the predator's neighbors with the shortest distances to the agent
	"""
	pred_nbrs_to_agent = {nbr: shortest_distances[(
		nbr, agent_loc)] for nbr in graph.get_node_neighbors(pred_loc)}
	smallest = min(pred_nbrs_to_agent.values())
	return [nbr for nbr in pred_nbrs_to_agent.keys() if pred_nbrs_to_agent[nbr] == smallest]

def save_model(model, error, filename=f"vpartial_dqn_"):
	dirname = "neuralnetworks/trainedmodels/"
	filepath = os.path.dirname(__file__) + dirname + filename
	with open(filepath + str(error) + ".pkl", "wb") as file:
		pickle.dump(model, file)

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

def get_transition_matrix(graph):
	# set up transition matrix
	P = [[0 for _ in range(graph.get_nodes())] for _ in range(graph.get_nodes())]
	nbrs = graph.get_neighbors()
	for i in nbrs.keys():
		P[index_transform(i)][index_transform(i)] = 1 / (len(nbrs[i]) + 1)
		for j in nbrs[i]:
			P[index_transform(i)][index_transform(j)] = 1 / (len(nbrs[j]) + 1)
	return P

def nn_compute_wrapper(nn, agent_loc, beliefs, p_loc):
	return np.asarray(nn.compute_output(get_state_vector(agent_loc, beliefs, p_loc)), dtype="float32")

def belief_system_init(graph, agent_location):
	'''
	Function to initialize a belief system. It returns an array of probabilities
	describing where the prey is and a probability transition matrix. 
	'''

	beliefs = [1/(graph.get_nodes() - 1) for _ in range(0, graph.get_nodes())]
	beliefs[index_transform(agent_location)] = 0 
	
	return beliefs

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

# NEURAL NETWORK FUNCTIONS

def init_q_function():
	'''
	Function to initialize the neural network. 
	'''

	q_function = NeuralNetwork()
	q_function.add(DenseLinear(150, 150))
	q_function.add(NonLinearity(tanh, tanh_prime))
	q_function.add(DenseLinear(150, 150))
	q_function.add(NonLinearity(tanh, tanh_prime))
	q_function.add(DenseLinear(150, 1))
	q_function.choose_error(mse, mse_prime)

	return q_function

def get_state_vector(a_loc, beliefs, p_loc):
	'''
	Function to develop a vector describing the state space
	'''

	state_vector = [0 for _ in range(0, 150)]
	state_vector[index_transform(a_loc)] = 1
	state_vector[50:100] = beliefs
	state_vector[100 + index_transform(p_loc)] = 1
	state_vector = np.asarray(state_vector, dtype="float32")
	return state_vector.reshape( (1, state_vector.shape[0]) )

def compute_convergence_condition(avg_loss, delta):
	return avg_loss < delta

def copy_neural_network(original):
	copy = NeuralNetwork()
	for layer in original.layers:
		if isinstance(layer, (DenseLinear)):
			temp = DenseLinear(layer.input_size, layer.output_size)
			temp.w = np.copy(layer.w)
			temp.b = np.copy(layer.b)
			copy.add(temp)
		else:
			copy.add(NonLinearity(layer.activation, layer.activation_derivative))

	copy.choose_error(mse, mse_prime)

	return copy

def get_future_reward(graph, agent_loc, beliefs, pred_loc, shortest_distances, q_function):
	"""
	Function to return the future reward of the current state by considering all of the 'sister states',
	or variations in the prey's and predator's location
	@param:graph - the graph this function operates on
	@param:agent_loc - the location of the agent
	@param:prey_loc - the location of the prey
	@param:pred_loc - the location of the predator
	@param:shortest_distances - a dictionary containing the shortest distances between every pair of nodes
	@param:u0 - a vector containing the utilities of each state from the previous sweep
	@return the future reward from being in this state
	"""
	# print("running")
	pred_next = graph.nbrs[pred_loc]
	pred_optimal_next = set(optimal_pred_moves(graph, agent_loc, pred_loc, shortest_distances))

	future_reward = 0
	for pred_next_state in pred_next:
		if agent_loc == pred_next_state:
			return -50
		gamma = 0.6 / len(pred_optimal_next) if pred_next_state in pred_optimal_next else 0
		future_reward += nn_compute_wrapper(q_function, agent_loc, beliefs, pred_next_state)[0] * (0.4 / len(pred_next) + gamma)
	return future_reward

def train():
	

	# q_function = init_q_function()
	q_function = load_model()
	target_network = copy_neural_network(q_function)
	
	graph = Graph(nbrs=utils.retrieve_graph())
	P = get_transition_matrix(graph)
	shortest_distances = agent_to_pred_distances(graph)
	
	epsilon, alpha, delta, beta = 0.10, 0.001, 0.1, 0.99
	
	game_vector = []
	
	number_of_games = 1000

	# gen a bunch of games
	print("Initializing games....")
	for _ in range(0, number_of_games):
		game_vector.append(init_new_game(graph))

	number_of_states_to_process = 20
	batch_size = 128
	avg_loss = float("inf")
	batches = 0

	print("Beginning training....")

	while not compute_convergence_condition(avg_loss, delta):
		print("Batch ..." + str(batches))
		batch_loss = 0
		
		for j in range(0, batch_size):
			processed = set()
			outputs, correct = [], []
			print("Running... iteration " + str(batches*5 + j))
			
			for i in range(0, number_of_states_to_process): # random number of games:

				# retrive a game
				if len(processed) == number_of_games:
					processed = set()
				
				random_game_index = random.randint(0, number_of_games - 1)
				while random_game_index in processed:
					random_game_index = random.randint(0, number_of_games - 1)
				# print("Game Selected")
				processed.add(random_game_index)

				cur_game_state = game_vector[random_game_index]
				agent_location, prey, predator, beliefs, found_prey = cur_game_state.retrieve_game()
				game_over = False
					
				# 	#survey a node
				beliefs, temp_found_prey = survey_function(graph, beliefs, prey)
				found_prey = found_prey or temp_found_prey

				# print("Node surveyed")
				# initial evaluation of the q_function on the state_vector
				outputs.append(nn_compute_wrapper(q_function, agent_location, beliefs, predator.location))
				# process_nn_output(initial_evaluation, graph, agent_location, epsilon) 
				# print("NN Evaluated")

				# calculate value iteration from and loss for each action we can take from here
				action_space = graph.get_node_neighbors(agent_location) + [agent_location] # this is Q(s,a)
				next_move = random.choice(action_space) # put in e - greedy

				# need to calculate the maximum future reward
				answer = 0 
				if agent_location == prey.location: 
					answer =0
				elif agent_location == predator.location:
					answer = -50
				else:
					actions_and_utilities = dict()
					# print(action_space)
					for action in action_space:
						actions_and_utilities[action] = get_future_reward(graph, action, beliefs, predator.location, shortest_distances, target_network)
						# print(actions_and_utilities[action])
					best = max(actions_and_utilities.values())
					answer = -1 + beta*best
					if random.random() >= epsilon:
						next_move = random.choice([action for action in actions_and_utilities.keys() if actions_and_utilities[action] == best])
				
				correct.append(np.asarray(answer, dtype="float32"))

				# move
				agent_location = next_move
				if agent_location == prey.location or agent_location == predator.location:
					game_over = True
				
				# update belief system for agent moving
				if not game_over:
					beliefs = update_belief_system_after_agent_moves(beliefs, agent_location, graph, found_prey)

				prey.move(graph)
				
				if agent_location == prey.location:
					game_over = True

				predator.move(graph, None, aloc=agent_location)
				
				if agent_location == predator.location:
					game_over = True

				# update belief system for the prey moving
				if not game_over:
					beliefs = update_belief_system_after_prey_moves(beliefs, P)

				game_vector[random_game_index] = init_new_game(graph) if game_over else Game_State(agent_location, prey, predator, beliefs, found_prey)

				# step_count = step_count + 1

			# back propagate loss
			loss_sum = 0
			for i in range(len(correct)):
				loss_sum = loss_sum + np.absolute(correct[i] - outputs[i])
				q_function.back_propagate(correct[i], outputs[i], alpha) # back propage the loss
			avg_loss = loss_sum / len(correct)
			batch_loss += avg_loss
			print("Average Loss: " , avg_loss)
		
		# copy over main weights to target network
		batches = batches + 1
		target_network = copy_neural_network(q_function)

		# checkpoint the model every 25 epochs
		if batches % 5 == 0:
			save_model(q_function, batch_loss/batch_size)
		
		if batches % 100 == 0: 
			epsilon = epsilon / 2
		
		if batch_loss / batch_size < 7.5:
			alpha = 0.0005
		
	print("Finished training....")

train()




# def process_nn_output(output, graph, agent_location, epsilon):
# 	'''
# 	Function that takes the output of the neural network and chooses a random action
# 	with a chance of epsilon and chooses the best action otherwise
# 	'''

# 	action_space = graph.get_node_neighbors(agent_location) + [agent_location]
# 	return random.choice(action_space)
# 	# if random.random() < epsilon:
# 	# 	best_action = random.choice(action_space)
# 	# else:
		
# 	# 	possible_actions = { i : output[0][index_transform(i)] for i in action_space }
# 	# 	best_utility = max(possible_actions.values())
# 	# 	lst = [i for i in possible_actions.keys() if possible_actions[i] == best_utility]
# 	# 	if len(lst) == 0:
# 	# 		print(output)
# 	# 		print(possible_actions)
# 	# 		print(best_utility)
# 	# 	best_action = random.choice([i for i in possible_actions.keys() if possible_actions[i] == best_utility])

# 	# return best_action


# for action in action_space:
				# 	q_s_prime_a = 0
				# 	if action == prey.location:
				# 		q_s_prime_a = 0
				# 	elif action == predator.location:
				# 		q_s_prime_a = -50
				# 	else:
				# 		new_state_vector = get_state_vector(action, beliefs, predator.location) # the new vector describing s_{t + 1} from taking an action
				# 		future_evaluation = np.asarray(target_network.compute_output(state_vector), dtype="float32") # a set of vectors describing Q(s_{t+1}, a)
				# 		optimal_future_action = process_nn_output(future_evaluation, graph, action, 0) # maximum action
				# 		q_s_prime_a = -1 + future_evaluation[0][index_transform(optimal_future_action)] # maximum Q(s_{t+1}, a)
				# 	answers[0][index_transform(action)] = q_s_prime_a
				
				# outputs.append(initial_evaluation)
				# correct.append(answers)