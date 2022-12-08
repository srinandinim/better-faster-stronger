import pickle 
from game.game import Game 


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







