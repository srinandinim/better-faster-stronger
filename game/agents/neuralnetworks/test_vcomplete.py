import numpy as np 
import nn 

# LOAD THE MODEL INTO MEMORY
model_vcomplete = nn.load_model(filename="trainedmodels/vcomplete_model0.520227572692611.pkl")

# LOAD DATA INTO MEMORY
y, x = nn.get_training_data(start_idx=0, end_idx=10000)

# DO INFERENCE ON THE MODEL WITH GIVEN DATA
print("TRAINING DATA PREDICTIONS")
print(model_vcomplete.predict(x[500:510]))
print(y[500:510])

print("TESTING DATA PREDICTIONS")
print(model_vcomplete.predict(x[8000:8010]))
print(y[8000:8010])

action=(1,2,3)
x = nn.vectorize_state(action)
print(x)

x = np.asarray(x, dtype="float32")
print(x)
print(x.shape)

x = x.reshape(1, x.shape[0])
print(x)
print(x.shape)

yhat = model_vcomplete.predict(x)
print(yhat)