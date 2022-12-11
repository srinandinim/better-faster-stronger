"""THIS FILE JUST SANITY CHECKS THE NN AND VARIOUS SERIALIZATION SCHEMES."""

import numpy as np
import utils
from train_vcomplete import get_training_data

# LOAD THE MODEL INTO MEMORY
print("loading")
model_vcomplete = utils.load_model_for_testing(filename="OPTIMAL_VCOMPLETE_MODEL.pkl")

# LOAD DATA INTO MEMORY
print("get training data!")
y, x = get_training_data(start_idx=0, end_idx=10000)

# DO INFERENCE ON THE MODEL WITH GIVEN DATA
print("TRAINING DATA PREDICTIONS")
print(model_vcomplete.predict(x[500:510]))
print(y[500:510])

print("TESTING DATA PREDICTIONS")
print(model_vcomplete.predict(x[8000:8010]))
print(y[8000:8010])

action = (1, 2, 3)
x = utils.vectorize_state(action)
print(x)

x = np.asarray(x, dtype="float32")
print(x)
print(x.shape)

x = x.reshape(1, x.shape[0])
print(x)
print(x.shape)

yhat = np.array(model_vcomplete.predict(x)).item()
print(yhat)
