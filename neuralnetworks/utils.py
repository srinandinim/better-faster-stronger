import pickle
import os


def vectorize_coordinate(coordinate, length=50):
    """
    takes a coordinate point and converts it to one hot vector. 

    Example:
    input: coordinate=3
    output: [0, 0, 1, 0, 0, 0, ..., 0, 0]
    """

    vector = []
    for i in range(length):
        if i == (coordinate-1):
            vector.append(1)
        else:
            vector.append(0)
    return vector


def vectorize_state(state):
    """
    takes a state and converts it to one hot vector matrix. 

    Example:
    input: state=(1,2,3)
    output: 
           [1, 0, 0, 0, 0, 0, ..., 0, 0]
           [0, 1, 0, 0, 0, 0, ..., 0, 0]
           [0, 0, 1, 0, 0, 0, ..., 0, 0]
    """
    x, y, z = state
    return vectorize_coordinate(x) + vectorize_coordinate(y) + vectorize_coordinate(z)


def renamed_load(file_obj):
    class RenameUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            renamed_module = module
            if module == "nn":
                renamed_module = "neuralnetworks.nn"

            return super(RenameUnpickler, self).find_class(renamed_module, name)
    return RenameUnpickler(file_obj).load()


def save_model(model, error, filename=f"vcomplete_model"):
    dirname = "/trainedmodels/"
    filepath = os.path.dirname(__file__) + dirname + filename
    with open(filepath + str(error) + ".pkl", "wb") as file:
        pickle.dump(model, file)
        # print("model successfully serialized")


def load_model_for_testing(filename="OPTIMAL_VCOMPLETE_MODEL.pkl"):
    dirname = "/trainedmodels/"
    filepath = os.path.dirname(__file__) + dirname + filename
    with open(filepath, "rb") as file:
        # print("opening the file")
        model = pickle.load(file)
        # print("model successfully deserialized")
    return model


def load_model_for_agent(filename="OPTIMAL_VCOMPLETE_MODEL.pkl"):
    dirname = "/trainedmodels/"
    filepath = os.path.dirname(__file__) + dirname + filename
    with open(filepath, "rb") as file:
        # print("opening the file")
        model = renamed_load(file)
        # print("model successfully deserialized")
    return model