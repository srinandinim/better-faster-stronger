import time 
import pickle
import numpy as np 
from utils_models import * 
from fastneuralnet import *

#  load X, Y targets into memory
Y,X = get_training_data()
print("Y, X are loaded into memory")

# build out the neural network
dnn_v_complete = Network()
dnn_v_complete.add(DenseLinear(150, 1000))
dnn_v_complete.add(NonLinearity(tanh, tanh_prime))
dnn_v_complete.add(DenseLinear(1000, 1000))
dnn_v_complete.add(NonLinearity(tanh, tanh_prime))
dnn_v_complete.add(DenseLinear(1000, 1))

# choose the loss function
dnn_v_complete.choose_error(mse, mse_prime)

# train the model 
start = time.time()
dnn_v_complete.train(X[:20000], X[:20000], epochs=100, learning_rate=0.001)
print(f"training took {time.time()-start} seconds")

# serialize the pickle 
with open("dnn_v_complete.pkl", "wb") as file:
    pickle.dump(dnn_v_complete, file)
    print("model successfully serialized")

# deserialize the pickle 
with open("dnn_v_complete.pkl", "rb") as file:
    pickle.load(file)
    print("model successfully deserialized")
