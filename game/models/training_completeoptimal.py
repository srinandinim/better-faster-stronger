import time 
import pickle
import numpy as np 
from utils_models import * 
from fastneuralnet import *

# load X, Y targets into memory
Y,X = get_training_data()
Y_val, X_val = get_evaluation_data()

X = np.array(X, dtype=np.float32)
X = X.reshape((X.shape[0], 1, 150))

Y = np.array(Y, dtype=np.float32)
Y = Y.reshape((Y.shape[0], 1))

X_val = np.array(X_val, dtype=np.float32)
X_val = X_val.reshape((X_val.shape[0], 1, 150))

Y_val = np.array(Y_val, dtype=np.float32)
Y_val = Y_val.reshape((Y_val.shape[0], 1))

print(X[0]) 
print(Y[0])



print("Y, X are loaded into memory")

# build out the neural network
dnn_v_complete = Network()
dnn_v_complete.add(DenseLinear(150, 256))
dnn_v_complete.add(NonLinearity(tanh, tanh_prime))
dnn_v_complete.add(DenseLinear(256, 256))
dnn_v_complete.add(NonLinearity(tanh, tanh_prime))
dnn_v_complete.add(DenseLinear(256, 256))
dnn_v_complete.add(NonLinearity(tanh, tanh_prime))
dnn_v_complete.add(DenseLinear(256, 1))

# choose the loss function
dnn_v_complete.choose_error(mse, mse_prime)

# train the model 
start = time.time()
dnn_v_complete.train(X, Y, X_val, Y_val, epochs=100, learning_rate=0.001)
end = time.time()-start
#print(f"training took {end} seconds")

# serialize the pickle 
with open("dnn_v_complete.pkl", "wb") as file:
    pickle.dump(dnn_v_complete, file)
    print("model successfully serialized")

# deserialize the pickle 
with open("dnn_v_complete.pkl", "rb") as file:
    dnn_v_complete = pickle.load(file)
    print("model successfully deserialized")


