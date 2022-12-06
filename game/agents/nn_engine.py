import nn 
import numpy as np 

model_vcomplete = nn.load_model(filename="neuralnetworks/trainedmodels/vcomplete_model0.520227572692611.pkl")

def model_predict_completeinfo(x):
    """takes one hot vector of 150 and returns utility"""
    return np.array(model_vcomplete.predict(x)).item()
