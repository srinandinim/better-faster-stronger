import nn
import numpy as np
from train_vcomplete import get_training_data

# LOAD THE MODEL INTO MEMORY
model_vcomplete = nn.load_model(filename="test_model_pickle.pkl")

# LOAD DATA INTO MEMORY
y, x = get_training_data(start_idx=0, end_idx=10000)

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

yhat = np.array(model_vcomplete.predict(x)).item()
print(yhat)