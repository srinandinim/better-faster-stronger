from nn import *

# LOAD THE DATA INTO MEMORY
y, x = get_training_data(start_idx=0, end_idx=125000) 

# BUILD OUT THE NEURAL NETWORK & LOSS FUNCTION
dnn_v_complete = NeuralNetwork()
dnn_v_complete.add(DenseLinear(150, 150))
dnn_v_complete.add(NonLinearity(tanh, tanh_prime))
dnn_v_complete.add(DenseLinear(150, 150))
dnn_v_complete.add(NonLinearity(tanh, tanh_prime))
dnn_v_complete.add(DenseLinear(150, 1))
dnn_v_complete.choose_error(mse, mse_prime)

# TRAIN THE MODEL WITH RESPECT TO THE DATAPOINTS
train(dnn_v_complete, x, y, 100, 0.001)