import time 
import pickle
import numpy as np 
from utils_models import * 
from fastneuralnet import *

# load X, Y targets into memory
Y,X = get_training_data()
print(X.shape, Y.shape)
#print(Y[0], X[0])
X = np.array(X, dtype=np.float32)
X = X.reshape((X.shape[0], 1, 150))
print(X)

Y = np.array(Y, dtype=np.float32)
Y = Y.reshape((Y.shape[0], 1))
print(Y)
#print(Y[0], X[0])
print(X.shape, Y.shape)


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
dnn_v_complete.train(X, Y, epochs=100, learning_rate=0.001)
end = time.time()-start
print(f"training took {end} seconds")

# serialize the pickle 
with open("dnn_v_complete.pkl", "wb") as file:
    pickle.dump(dnn_v_complete, file)
    print("model successfully serialized")

# deserialize the pickle 
with open("dnn_v_complete.pkl", "rb") as file:
    pickle.load(file)
    print("model successfully deserialized")
