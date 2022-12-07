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


if __name__ == "__main__":
    # LOAD THE DATA INTO MEMORY
    y_train, x_train = get_training_data()
    y_test, x_test = get_testing_data()

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
    dnn_vpartial = load_model_for_testing(filename="OPTIMAL_VCOMPLETE_MODEL.pkl")

    # TRAIN THE MODEL WITH RESPECT TO THE DATAPOINTS
    train_vpartial(dnn_vpartial, x_train, y_train, x_test, y_test, 100, 0.001)

    # TRAIN THE MODEL WITH RESPECT TO THE DATAPOINTS
    # train(dnn_v_complete, x, y, 100, 0.001)
