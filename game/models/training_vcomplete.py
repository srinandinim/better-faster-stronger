import time 
from dnn import *
from utils_models import * 
import numpy as np 

#  load X, Y targets into memory
Y,X = get_training_data()

print("Loaded training data into memory")
print(f"y1 is {Y[1]}")
print(f"x1 is {X[1]}")

# build a deep neural network
nn_complete_v = DNN(3, [50, 20, 10, 1])
print("Built a neural network.")

# hyperparameters of neural network 
EPOCHS = 2
MINIBATCH_SIZE = 500
LEARNING_RATE = 0.001

# train a deep neural network 
for epochs in range(EPOCHS):
    minibatch_number = 0 
    for mb in range(0, len(X), MINIBATCH_SIZE):
        start = time.time()
        
        # get a minibatch of data 
        location = mb + minibatch_number*MINIBATCH_SIZE
        X_mb = X[location: location + MINIBATCH_SIZE].astype(int).tolist()
        Y_mb = Y[location: location + MINIBATCH_SIZE].tolist()

        # if a value is negative infinity change it to -50 
        for i in range(len(Y_mb)):
            if Y_mb[i] == -float("inf"): 
                Y_mb[i] = -50 

        #print(X_mb)
        #print(Y_mb)

        # runs a forward pass of the minibatch on the neural network 
        ypreds = [nn_complete_v(x) for x in X_mb]

        # computes the total loss on the minbatch 
        loss = compute_total_loss(compute_mse_loss_list(ypreds, Y_mb))
        if loss.data < 4: LEARNING_RATE = 0.0001

        # calculate the gradients for each parameter with backprop
        loss.backward()

        # adjust parameters based on the gradients computed
        for p in nn_complete_v.parameters():
            p.data += -1 * LEARNING_RATE * p.grad

        # zero out the gradients
        for p in nn_complete_v.parameters():
            p.grad = 0 

        # update the minibatch
        minibatch_number += 1
        print(f"this minibatch update took {time.time()-start} seconds")
        
        print("EPOCH {}: MINIBATCH {} \t LOSS {}".format(epochs, minibatch_number, loss.data))
    print("Epoch {} Completed".format(epochs))



#  for k epochs through datasets for minibatches of size m, run backprop


#  save the model into a binary after it has been trained 