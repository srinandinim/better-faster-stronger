import os
import numpy as np
from nn import *
from utils import load_model_for_testing


def get_training_data(filename="upartial_data.csv", start_idx=0, end_idx=60000):
    """
    retrieves the start:end datapoints for the targets Y and input features X
    """
    dirname = "/trainingdata/"
    filepath = os.path.dirname(__file__) + dirname + filename
    data = np.loadtxt(filepath, delimiter=",")
    np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)

    # loads the CSV file from numpy into memory
    Y, X = data[start_idx:end_idx, 0], data[start_idx:end_idx, 1:]
    print(Y.shape, X.shape)

    # make all negative infinity values predict to -50
    Y[Y == -np.inf] = -50

    # reshape the data so that we can input the work onto neural net
    X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
    X, Y = X.reshape((X.shape[0], 1, X.shape[1])), Y.reshape((Y.shape[0], 1))

    # returns the
    return Y, X

def get_testing_data(filename="upartial_data.csv", start_idx=60000, end_idx=70000):
    """
    retrieves the start:end datapoints for the targets Y and input features X
    """
    dirname = "/trainingdata/"
    filepath = os.path.dirname(__file__) + dirname + filename
    data = np.loadtxt(filepath, delimiter=",")
    np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)


    # loads the CSV file from numpy into memory
    Y, X = data[start_idx:end_idx, 0], data[start_idx:end_idx, 1:]
    print(Y.shape, X.shape)

    # make all negative infinity values predict to -50
    Y[Y == -np.inf] = -50

    # reshape the data so that we can input the work onto neural net
    X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
    X, Y = X.reshape((X.shape[0], 1, X.shape[1])), Y.reshape((Y.shape[0], 1))

    # returns the
    return Y, X

def sanity_check_data(filename="upartial_data_sanitycheckfinal.csv", start_idx=0, end_idx=1000000):
    """
    retrieves the start:end datapoints for the targets Y and input features X
    """
    dirname = "/trainingdata/"
    filepath = os.path.dirname(__file__) + dirname + filename
    data = np.loadtxt(filepath, delimiter=",")

    # loads the CSV file from numpy into memory
    Y, X = data[start_idx:end_idx, 0], data[start_idx:end_idx, 1:]
    print(Y.shape, X.shape)

    # make all negative infinity values predict to -50
    Y[Y == -np.inf] = -50

    # reshape the data so that we can input the work onto neural net
    X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
    X, Y = X.reshape((X.shape[0], 1, X.shape[1])), Y.reshape((Y.shape[0], 1))

    # returns the
    return Y, X

def split_array(a):
  # get the first 50 elements of the array
  first_50 = a[:, :50]

  # get the middle 50 elements of the array
  middle_50 = a[:, 50:100]

  # get the last 50 elements of the array
  last_50 = a[:, 100:]

  return first_50, middle_50, last_50

def join_arrays(first_50, middle_50, last_50):
  # stack the arrays along the second dimension (columns)
  joined_array = np.hstack((first_50, middle_50, last_50))

  return joined_array

if __name__ == "__main__":

    #### DO NOT DELETE ANY OF THIS, ALL CRUCIAL FOR TRAINING/LOADING/USING MODELS ####
    
    # LOAD THE DATA INTO MEMORY
    # y_train, x_train = get_training_data()
    # y_test, x_test = get_testing_data()

    # print(y_train[0], x_train[0])
    # print(y_test[0], x_test[0])

    # BUILD OUT THE NEURAL NETWORK & LOSS FUNCTION
    #dnn_v_complete = NeuralNetwork()
    #dnn_v_complete.add(DenseLinear(150, 150))
    #dnn_v_complete.add(NonLinearity(tanh, tanh_prime))
    #dnn_v_complete.add(DenseLinear(150, 150))
    #dnn_v_complete.add(NonLinearity(tanh, tanh_prime))
    #dnn_v_complete.add(DenseLinear(150, 1))
    #dnn_v_complete.choose_error(mse, mse_prime)

    # LEVERAGE THE POWER OF FINE-TUNING FROM PRE-EXISTING TRAINED MODEL
    #dnn_vpartial = load_model_for_testing(filename="OPTIMAL_VCOMPLETE_MODEL.pkl")

    # TRAIN THE MODEL WITH RESPECT TO THE DATAPOINTS
    #train_vpartial(dnn_vpartial, x_train, y_train, x_test, y_test, 100, 0.001)

    # TRAIN THE MODEL WITH RESPECT TO THE DATAPOINTS
    # train(dnn_v_complete, x, y, 100, 0.001)

    
    # LOAD IN DATASET FOR TESTING THE MODEL
    Y_TEST, X_TEST = sanity_check_data()

    # TEST THE PERFORMANCE OF THE TRAINED MODEL ON A TESTING DATASET
    dnn_vpartial = load_model_for_testing(filename="OPTIMAL_VCOMPLETE_MODEL.pkl")

    """
    # FIND OUT THE TESTING ERROR
    total_mse_error = 0 
    for i in range(len(X_TEST)):
        # print(i)
        print(X_TEST[i])
        output = X_TEST[i]
        for layer in dnn_vpartial.layers:
            output = layer.forward(output)
        total_mse_error += dnn_vpartial.loss(Y_TEST[i], output)
    
    """
    
    # ANALYSIS QUESTION -- REPLACE U WITH V IN PARTIAL
    total_mse_error = 0 
    for i in range(len(X_TEST)):
        print(i)
        agent, prey, pred = split_array(X_TEST[i])
        prediction = 0 
        for j in range(0, 50):
            prey_one_hot_vector = np.zeros((1,50))
            prey_one_hot_vector[0, j] = 1 
            p = prey[0, j]

            if p != 0: 
                x = join_arrays(agent, prey_one_hot_vector, pred)
                output = x 
                for layer in dnn_vpartial.layers:
                    output = layer.forward(output)
                prediction += p * output 
        total_mse_error += p * dnn_vpartial.loss(Y_TEST[i], prediction)

    print(f"The total average MSE testing error on {len(X_TEST)} examples is {total_mse_error/len(X_TEST)}.")
    

