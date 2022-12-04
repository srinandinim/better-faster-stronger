import pickle 
from dnn import *

neural_net = DNN(3, [3, 3,1])


import time 
from dnn import *
from utils_models import * 
import numpy as np 

X = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
Y = [2.0, -2.0, -3.0, 2.0]

print("Loaded training data into memory")
print(f"y1 is {Y[1]}")
print(f"x1 is {X[1]}")

# build a deep neural network
nn_complete_v = DNN(150, [20, 20, 1])
print("Built a neural network.")

# hyperparameters of neural network 
EPOCHS = 1000
MINIBATCH_SIZE = 4
LEARNING_RATE = 0.001

# train a deep neural network 
for epochs in range(EPOCHS):
    minibatch_number = 0 
    for mb in range(0, len(X), MINIBATCH_SIZE):
        start = time.time()
        
        # get a minibatch of data 
        location = mb + minibatch_number*MINIBATCH_SIZE
        X_mb = X[location: location + MINIBATCH_SIZE]
        Y_mb = Y[location: location + MINIBATCH_SIZE]

        #print(X_mb, Y_mb)

        # runs a forward pass of the minibatch on the neural network 
        ypreds = [neural_net(x) for x in X_mb]

        # computes the total loss on the minbatch 
        loss = compute_total_loss(compute_mse_loss_list(ypreds, Y_mb))
        if loss.data < 4: LEARNING_RATE = 0.0001

        # calculate the gradients for each parameter with backprop
        loss.backward()

        # adjust parameters based on the gradients computed
        for p in neural_net.parameters():
            p.data += -1 * LEARNING_RATE * p.grad
        
        # updates the loss
        neural_net.zero_grad()

        # update the minibatch
        minibatch_number += 1
        #print(f"this minibatch update took {time.time()-start} seconds")
        
        print("EPOCH {}: MINIBATCH {} \t LOSS {}".format(epochs, minibatch_number, loss.data))
    print("Epoch {} Completed".format(epochs))

#  for k epochs through datasets for minibatches of size m, run backprop


#  save the model into a binary after it has been trained 

ypreds = [neural_net(x) for x in xs]#
print(ypreds)

for i in range(400):

    ypreds = [neural_net(x) for x in xs]
    #print(ypreds)
    loss = compute_total_loss(compute_mse_loss_list(ypreds, ys))
    if loss.data < .01: break
    print(loss.data)
    loss.backward() 
    for p in neural_net.parameters():
        p.data += -0.001 * p.grad

ypreds = [neural_net(x) for x in xs]
print(ypreds)


## SERIALIZE THE TRAINED NEURAL NET WITH PICKLE 
def serialize(obj, file_name, verbose=True):
    if verbose:
        print("Serializing to:", file_name)
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)


## DESERIALIZE THE TRAINED NEURAL NET WITH PICKLE 
def deserialize(file_name, verbose=True):
    if verbose:
        print("Deserializing from:", file_name)
    with open(file_name, "rb") as f:
        return pickle.load(f)
        
serialize(neural_net, "trained-dnn-model.pkl")
#neural_net = deserialize("trained-dnn-model.pkl")