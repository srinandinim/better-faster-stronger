import numpy as np 

def get_training_data(data=np.loadtxt("data/supervised_dataset_1hotvector_vcomplete.csv", delimiter=",")):
    """
    reads through dataset and returns the Y, and X values of the data 
    """
    Y, X = data[:500,0], data[:500,1:]
    Y[Y == -np.inf] = -50
    return Y, X

if __name__ == "__main__":

    
    Y, X = get_training_data()
    print(f"{Y.shape}, {X.shape}")
    print(f"y1 is {Y[1]}")
    print(f"x1 is {X[1]}")
    print(f"y2 is {Y[2]}")
    print(f"x2 is {X[2]}")
    
    X = np.array(
    [
        [ [0,0] ], 
        [ [0,1] ], 
        [ [1,0] ], 
        [ [1,1] ]
    ]
    )

    Y = np.array(
        [
            [[-9]], [[9]], [[8]], [[-27]]
        ]
    )

    print(f"{Y.shape}, {X.shape}")
    print(f"y1 is {Y[1]}")
    print(f"x1 is {X[1]}")
    print(f"y2 is {Y[2]}")
    print(f"x2 is {X[2]}")


