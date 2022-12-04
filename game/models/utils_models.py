import numpy as np 

def get_training_data(data=np.loadtxt("data/supervised_dataset_normalstates_vcomplete.csv", delimiter=",")):
    """
    reads through dataset and returns the Y, and X values of the data 
    """
    Y, X = data[:,0], data[:,1:]
    return Y, X

if __name__ == "__main__":
    #Y, X = get_training_data()
    #print(f"y1 is {Y[1]}")
    #print(f"x1 is {X[1]}")
    #print(f"y2 is {Y[2]}")
    #print(f"x2 is {X[2]}")
    pass 
