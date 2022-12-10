import pickle 
from neuralnetworks.nn import *
from game.game import *
import random
import numpy as np
 

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
	return state_vector.reshape( (1, state_vector.shape[0]) )

def process_nn_output(output, graph, agent_location, epsilon):
	'''
	Function that takes the output of the neural network and chooses a random action
	with a chance of epsilon and chooses the best action otherwise
	'''

	action_space = graph.get_node_neighbors(agent_location) + [agent_location]
	if random.random() < epsilon:
		best_action = random.choice(action_space)
	else:
		
		possible_actions = { i : output[0][index_transform(i)] for i in action_space }
		best_utility = max(possible_actions.values())
		lst = [i for i in possible_actions.keys() if possible_actions[i] == best_utility]
		if len(lst) == 0:
			print(output)
			print(possible_actions)
			print(best_utility)
		best_action = random.choice([i for i in possible_actions.keys() if possible_actions[i] == best_utility])

	return best_action

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

def train():
	
	q_function = init_q_function()
	target_network = copy_neural_network(q_function)
	epsilon, alpha, delta = 0.10, 0.001, 0.1
	game_vector = []
	number_of_games = 1000
	# gen a bunch of games
	print("Initializing games....")
	for _ in range(0, number_of_games):
		game_vector.append(init_new_game())

	number_of_states_to_process = 128
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
				# print("Game running")
				# retrive a game
				random_game_index = random.randint(0, number_of_games - 1)
				while random_game_index in processed:
					random_game_index = random.randint(0, number_of_games - 1)
				# print("Game Selected")
				processed.add(random_game_index)

				cur_game_state = game_vector[random_game_index]
				agent_location, prey, predator, beliefs, graph, game, P, found_prey = cur_game_state.retrieve_game()
				game_over = False
					
				# 	#survey a node
				beliefs, temp_found_prey = survey_function(graph, beliefs, prey)
				found_prey = found_prey or temp_found_prey

				# print("Node surveyed")
				# initial evaluation of the q_function on the state_vector
				state_vector = get_state_vector(agent_location, beliefs, predator.location)
				initial_evaluation = np.asarray(q_function.compute_output(state_vector), dtype="float32")
				next_move = process_nn_output(initial_evaluation, graph, agent_location, epsilon) 
				# print("NN Evaluated")

					# calculate value iteration from and loss for each action we can take from here
				action_space = graph.get_node_neighbors(agent_location) + [agent_location] # this is Q(s,a)
				answers = np.copy(initial_evaluation)
				for action in action_space:
					q_s_prime_a = 0
					if action == prey.location:
						q_s_prime_a = 0
					elif action == predator.location:
						q_s_prime_a = -50
					else:
						new_state_vector = get_state_vector(action, beliefs, predator.location) # the new vector describing s_{t + 1} from taking an action
						future_evaluation = np.asarray(target_network.compute_output(state_vector), dtype="float32") # a set of vectors describing Q(s_{t+1}, a)
						optimal_future_action = process_nn_output(future_evaluation, graph, action, 0) # maximum action
						q_s_prime_a = -1 + future_evaluation[0][index_transform(optimal_future_action)] # maximum Q(s_{t+1}, a)
					answers[0][index_transform(action)] = q_s_prime_a
				
				outputs.append(initial_evaluation)
				correct.append(answers)


				# move
				agent_location = next_move
				if agent_location == prey.location:
					game_over = True
				if agent_location == predator.location:
					game_over = True
				
				# update belief system for agent moving
				if not game_over:
					if beliefs[index_transform(agent_location)] == 1:
						print(agent_location)
						print(prey.location)
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

				game_vector[random_game_index] = init_new_game() if game_over else Game_State(agent_location, prey, predator, beliefs, graph, game, P, found_prey)

				# step_count = step_count + 1

			# back propagate loss
			loss_sum = 0
			for i in range(len(correct)):
				loss_sum = loss_sum + (np.sum(np.absolute(np.subtract(correct[i], outputs[i]))))
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