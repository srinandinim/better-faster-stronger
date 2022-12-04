import pickle 
from node import Node  
from dnn import * 

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
        
#serialize(neural_net, "trained-dnn-model.pkl")
neural_net = deserialize("trained-dnn-model.pkl")
print(neural_net([2.0, 3.0, -1.0]))