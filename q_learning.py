from neuralnetworks.nn import *
from game.game import *
import random
import numpy as np
 

class Game_State:
	def __init__(self, agent, prey, predator, beliefs, graph, game, P):
		self.agent = agent
		self.prey = prey
		self.predator = predator
		self.beliefs = beliefs
		self.graph = graph
		self.game = game
		self.P = P

	def retrieve_game(self):
		return self.agent, self.prey, self.predator, self.beliefs, self.graph, self.game, self.P

def init_new_game():
	game = Game()
	agent_location, prey, predator, graph, _ = game.setup_q_learning()
	beliefs, P = belief_system_init(graph, agent_location)
	return Game_State(agent_location, prey, predator, beliefs, graph, game, P)


# HELPER FUNCTIONS ##############################

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
	return random.choice([i for i in graph.get_neighbors().keys() if vector[i] == max_prob])


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

	survey_spot = pick_most_probable_spot(graph, beliefs)
	found_prey = False
	if survey_spot == prey.location:
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
	q_function.add(DenseLinear(150, 256))
	q_function.add(NonLinearity(tanh, tanh_prime))
	q_function.add(DenseLinear(256, 256))
	q_function.add(NonLinearity(tanh, tanh_prime))
	q_function.add(DenseLinear(256, 256))
	q_function.add(NonLinearity(tanh, tanh_prime))
	q_function.add(DenseLinear(256, 50))
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
	return state_vector.reshape(state_vector.shape[0])

def process_nn_output(output, graph, agent_location, epsilon):
	'''
	Function that takes the output of the neural network and chooses a random action
	with a chance of epsilon and chooses the best action otherwise
	'''

	action_space = graph.get_node_neighbors(agent_location) + [agent_location]
	if random.random() < epsilon:
		best_action = random.choice(action_space)
	else:
		
		possible_actions = { i : output[index_transform(i)] for i in action_space }
		best_utility = max(possible_actions.values())
		best_action = random.choice([i for i in possible_actions.keys() if possible_actions[i] == best_utility])

	return best_action

def compute_convergence_condition(avg_loss, delta):
	return avg_loss < delta


def train():

	q_function = init_q_function()
	epsilon, alpha, delta = 0.1, 0.001, 0.1
	game_vector = []
	number_of_games = 100
	# gen a bunch of games
	for _ in range(0, number_of_games):
		game_vector.append(init_new_game())

	number_of_states_to_process = 100
	avg_loss = float("inf")
	i = 1
	while not compute_convergence_condition(avg_loss, delta):
		loss_sum = 0
		processed = set()
		outputs, predicted = [], []
		
		print("Running... iteration " + str(i))
		for i in range(0, number_of_states_to_process): # random number of games:

			# retrive a game
			random_game_index = random.randint(0, number_of_games)
			while random_game_index not in processed:
				random_game_index = random.randint(0, number_of_games)
				 
			processed.add(random_game_index)
			cur_game_state = game_vector[random_game_index]
			agent_location, prey, predator, beliefs, graph, game, P = cur_game_state.retrieve_game()
			game_over = False
				
			# 	#survey a node
			beliefs, temp_found_prey = survey_function(graph, beliefs, prey)
			found_prey = found_prey or temp_found_prey


			# initial evaluation of the q_function on the state_vector
			state_vector = get_state_vector(agent_location, beliefs, predator.location)
			initial_evaluation = np.asarray(q_function.compute_output(state_vector), dtype="float32")
			next_move = process_nn_output(initial_evaluation, graph, agent_location, epsilon) 


				# calculate value iteration from and loss for each action we can take from here
			action_space = graph.get_node_neighbors(agent_location) + [agent_location] # this is Q(s,a)
			for action in action_space:
				q_s_prime_a = 0
				if action == prey.location:
					q_s_prime_a = 0
				elif action == predator.location:
					q_s_prime_a = -float("inf")
				else:
					new_state_vector = get_state_vector(action, beliefs, predator.location) # the new vector describing s_{t + 1} from taking an action
					future_evaluation = np.asarray(q_function.compute_output(state_vector), dtype="float32") # a set of vectors describing Q(s_{t+1}, a)
					optimal_future_action = process_nn_output(future_evaluation, graph, action, 0) # maximum action
					q_s_prime_a = -1 + future_evaluation[index_transform(optimal_future_action)] # maximum Q(s_{t+1}, a)
				
				outputs.append(initial_evaluation[index_transform(action)])
				predicted.append(q_s_prime_a)


			# move
			agent_location = next_move
			# update belief system for agent moving
			beliefs = update_belief_system_after_agent_moves(beliefs, agent_location, graph, found_prey)

			if agent_location == prey.location:
				game_over = True
			if agent_location == predator.location:
				game_over = True
			prey.move(graph)
			if agent_location == prey.location:
				game_over = True

			predator.move(graph, aloc=agent_location)
			
			if agent_location == predator.location:
				game_over = True

			# update belief system for the prey moving

			beliefs = update_belief_system_after_prey_moves(beliefs, P)

			game_vector[random_game_index] = init_new_game() if game_over else Game_State(agent_location, prey, predator, beliefs, graph, game, P)

			# step_count = step_count + 1

		# back propagate loss
		loss_sum = 0
		for i in range(len(predicted)):
			loss_sum = loss_sum + (predicted[i] - outputs[i])
			q_function.back_propagate(predicted[i], outputs[i], alpha) # back propage the loss
		avg_loss = loss_sum / len(predicted)
		print("Average Loss: " , avg_loss)
		i = i + 1


train()
