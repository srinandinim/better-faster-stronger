import nn 

# LOAD THE MODEL INTO MEMORY
model_vcomplete = nn.load_model(filename="/Users/tejshah/Desktop/better-faster-stronger/neuralnetworks/trainedmodels/vcomplete_model4.645223311537296.pkl")

# LOAD DATA INTO MEMORY
y, x = nn.get_training_data(start_idx=0, end_idx=10000)

# DO INFERENCE ON THE MODEL WITH GIVEN DATA
print("TRAINING DATA PREDICTIONS")
print(model_vcomplete.predict(x[500:510]))
print(y[500:510])

print("TESTING DATA PREDICTIONS")
print(model_vcomplete.predict(x[8000:8010]))
print(y[8000:8010])