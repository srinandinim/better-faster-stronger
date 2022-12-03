from dnn import *

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

neural_net = DNN(3, [4, 4,1])

ypreds = [neural_net(x) for x in xs]
print(ypreds)

for i in range(10):
    ypreds = [neural_net(x) for x in xs]
    #print(ypreds)
    loss = compute_total_loss(compute_mse_loss_list(ypreds, ys))
    print(loss.data)
    loss.backward() 
    for p in neural_net.parameters():
        p.data += -0.1 * p.grad

ypreds = [neural_net(x) for x in xs]
print(ypreds)


"""
# arbitrary xi
x = [2.0, 3.0, -1.0]

# initialized DNN 
dnn = DNN(3, [4,4,1])

# initial output for arbitrary DNN
print("initial output for arbitrary DNN")
print(dnn(x))

# how many network parameters to optimize
print("how many network parameters to optimize")
print(len(dnn.parameters()))

# dataset for training
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [2.0, -2.0, -2.0, 2.0]

# predict outputs for every item in training data
print("predict outputs for every item in training data")
ypred=[dnn(x) for x in xs]
print(ypred)

# compute error between true and predicted targets
print("# compute error between true and predicted targets")
loss_list = compute_mse_loss(predictions, targets)
print(loss_list)
loss = compute_total_loss(loss_list)
print(loss)

# see the data and the gradients before
print("see the data and the gradients before backprop")
print(dnn.layers[0].neurons[0].w[0].data)
print(dnn.layers[0].neurons[0].w[0].grad)

# compute backprop on the layers
loss.backward()

# update the parameters of the network based on gradient descent
for p in dnn.parameters():
    p.data += -0.01 * p.grad

# see the data and the gradients
print("see the data and the gradients after backprop")
print(dnn.layers[0].neurons[0].w[0].data)
print(dnn.layers[0].neurons[0].w[0].grad)

# predict outputs for every item in training data
print("predict outputs for every item in training data")
ypred=[dnn(x) for x in xs]
print(ypred)

# compute error between true and predicted targets
print("# compute error between true and predicted targets")
loss_list = compute_mse_loss()
print(loss_list)
loss = compute_total_loss(loss_list)
print(loss)
"""
