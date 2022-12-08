
from neuralnetworks.nn import *
import random

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
    return choice([i for i in graph.get_neighbors().keys() if vector[i] == max_prob])


# BELIEF SYSTEM FUNCTIONS ##############################

def belief_system_init(graph):
	'''
	Function to initialize a belief system. It returns an array of probabilities
	describing where the prey is and a probability transition matrix. 
	'''

	beliefs = [1/(graph.get_nodes() - 1) for _ in range(0, graph.get_nodes())]
    beliefs[index_transform(agent.location)] = 0 

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

def update_belief_system_after_agent_moves(beliefs, agent, graph, found_prey):
    '''
    Function to update the belief system after the agent moves. If the prey is not found,
    it reinitializes the belief system. If the prey has been found, then it uses Bayes' 
    Law to update. 
    '''

    if not found_prey:
        beliefs = [1/(graph.get_nodes() - 1) for _ in range(0, graph.get_nodes())]
    else:
        old_agent_pos_prob = beliefs[index_transform(agent.location)]
        beliefs = list(map(lambda x: x / (1 - old_agent_pos_prob), beliefs))
    beliefs[index_transform(agent.location)] = 0
    
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

	q_function = Network()
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
	return state_vector

def process_nn_output(output, graph, agent_location, epsilon):
	'''
	Function that takes the output of the neural network and chooses a random action
	with a chance of epsilon and chooses the best action otherwise
	'''

	action_space = graph.get_node_neighbors(agent_location) + [agent_location]
	if random.random() < epsilon:
		best_action = choice(action_space)
	else:
		possible_actions = [i:output[index_transform(i)] for i in action_space]
		best_utility = max(possible_actions.values())
		best_action = random.choice([i in possible_actions.keys() if possible_actions[i] == best_utility])

	return best_action


def train():

	q_function = init_q_function()
	epsilon, alpha = 0.1, 0.001
	for i in range(0, 1000): # random number of games:

		# initialize a game

		game = Game(nodes=nodes)
		agent, prey, predator, graph, timeout = game.setup_q_learning()
		status = 0
        step_count = 0

        # INIT BELIEF SYSTEM
        beliefs, P = belief_system_init()

        found_prey = False

        while status == 0 and step_count < timeout:
        	
			#survey a node
			beliefs, temp_found_prey = survey_function(graph, beliefs, prey)
			found_prey = found_prey or temp_found_prey


			# initial evaluation of the q_function on the state_vector
			state_vector = get_state_vector(agent.location, beliefs, predator.location)
			initial_evaluation = q_function.compute_output(state_vector)
			next_move = process_nn_output(initial_evaluation, graph, agent.location, epsilon) 


			# calculate value iteration from and loss for each action we can take from here
			action_space = graph.get_node_neighbors(agent.location) + [agent.location] # this is Q(s,a)
			for action in action_space:
				new_state_vector = get_state_vector(action, beliefs, predator.location) # the new vector describing s_{t + 1} from taking an action
				future_evaluation = q_function.compute_output(state_vector) # a set of vectors describing Q(s_{t+1}, a)
				optimal_future_action = process_nn_output(future_evaluation, graph, action, 0) # maximum action
				q_s_prime_a = -1 + future_evaluation[index_transform(optimal_future_action)] # maximum Q(s_{t+1}, a)
				q_function.back_propagate(q_s_prime_a, initial_evaluation[index_transform(action)], alpha) # back propage the loss


			# move
			agent.location = next_move
			# update belief system for agent moving
			beliefs = update_belief_system_after_moving(beliefs, agent, graph, found_prey)

			if agent.location == prey.location:
				status = 1
			if agent.location == predator.location:
				status = -1

			prey.move(graph)
			if agent.location == prey.location:
				status = 1

			predator.move(graph, agent)
			
			if agent.location == predator.location:
				status = -1 

			# update belief system for the prey moving

			beliefs = update_belief_system_after_prey_moves(beliefs, P)

            step_count = step_count + 1
		
		# agent timed out
		if status == 0:
        	status = -2
