import os
os.environ['TL_BACKEND'] = 'paddle'

import tensorlayerx as tlx
from tensorlayerx.nn import Linear
from tensorlayerx.nn import Sequential


tlx.set_seed(99999)  # set random set

# Single neuro

# matrix multiplication
x = tlx.convert_to_tensor([[1., 2., 3.]])
w = tlx.convert_to_tensor([[-0.5], [0.2], [0.1]])

print("X:\n", x, "\nShape:", x.shape)
print("W:\n", w, "\nShape", w.shape)

# Bias
b1 = tlx.convert_to_tensor(0.5)

z1 = tlx.matmul(x, w)+b1

print("Z:\n", z1, "\nShape", z1.shape)

# Two outputs

x = tlx.convert_to_tensor([[1., 2., 3.]])
w = tlx.convert_to_tensor([[-0.5, -0.3],
                           [0.2, 0.4],
                           [0.1, 0.15]])

print("X:\n", x, "\nShape:", x.shape)
print("W:\n", w, "\nShape", w.shape)

# Bias
b2 = tlx.convert_to_tensor([0.5, 0.4])

z2 = tlx.matmul(x, w)+b2

print("Z:\n", z2, "\nShape", z2.shape)


# Activation
# Reference: https://en.wikipedia.org/wiki/Activation_function
# Sigmoid function
sigmoid_tlx = tlx.nn.activation.Sigmoid()
a1 = sigmoid_tlx(z1)
print("Result tlx sigmoid:", a1)

# define your own activation function


class ActSigmoid(tlx.nn.Module):
    def forward(self, x):
        return 1 / (1 + tlx.exp(-x))


sigmoid_act = ActSigmoid()
a1_m = sigmoid_act(z1)

print("Result your own sigmoid:", a1_m)

# Softmax
a2 = tlx.softmax(z2, axis=-1)
print("Result tlx softmax:", a2)


class ActSoftmax(tlx.nn.Module):
    def forward(self, x, axis=-1):
        e = tlx.exp(x - tlx.reduce_max(x, axis=axis, keepdims=True))
        s = tlx.reduce_sum(e, axis=axis, keepdims=True)
        return e / s


softmax_act = ActSoftmax()
a2_m = softmax_act(z2, axis=-1)
print("Result your own softmax:", a2_m)


# MLP

layer_list = []
layer_list.append(Linear(out_features=3, act=tlx.ReLU,
                  in_features=3, name='linear1'))
layer_list.append(Linear(out_features=3, act=tlx.ReLU,
                  in_features=3, name='linear2'))
layer_list.append(Linear(out_features=3, act=tlx.ReLU,
                  in_features=3, name='linear3'))
layer_list.append(Linear(out_features=2, act=tlx.Softmax,
                  in_features=3, name='linear4'))
MLP = Sequential(layer_list)

out = MLP(x)
print("Neural network output: ", out.shape)

# Loss functions


def mae(output, target):
    return tlx.reduce_mean(tlx.abs(output-target), axis=-1)


def mse(output, target):
    return tlx.reduce_mean(tlx.square(output-target), axis=-1)


y1 = tlx.convert_to_tensor([1., 3., 5., 7.])
y2 = tlx.convert_to_tensor([2., 4., 6., 8.])

l_mae = mae(y1, y2)
l_mse = mse(y1, y2)

print("Loss MAE: {} \nLoss MSE: {}".format(l_mae, l_mse))

l_mse_tlx = tlx.losses.mean_squared_error(y1, y2)
print("Loss MSE by TLX: {}".format(l_mse_tlx.numpy()))


target_binary = tlx.convert_to_tensor([[1., 0.]])

print("Target value:{}".format(target_binary.numpy()))
print("Neural network output:{}".format(out.numpy()))

# Binary cross entropy
l_bce = tlx.losses.binary_cross_entropy(out, target_binary)
print("Loss binary cross entropy:{}".format(l_bce.numpy()))


# Error Back-Propagation
# Example
x = tlx.convert_to_tensor(1.)
x.stop_gradient = False
w = tlx.convert_to_tensor(0.5)
w.stop_gradient = False

t = x+w
z = t**2
z.backward()

print("Gradient of w is: {}".format(w.grad.numpy()))

# BP for the network
l_bce.backward()

print("Gradient of the network layer 1's weights is: \n{}".format(
    MLP.layer_list[0].weights.grad.numpy()))

#Optimization
opt = tlx.optimizers.SGD(lr=0.01, momentum=0.9)

x_in = tlx.convert_to_tensor([[1., 2., 3.]])
target = tlx.convert_to_tensor([[1., 0.]])

print("Before optimization, network layer 1's weights is: \n{}".format(
    MLP.layer_list[0].weights.numpy()))

out = MLP(x_in)
l_bce = tlx.losses.binary_cross_entropy(out, target)

grads = opt.gradient(l_bce, MLP.trainable_weights)
opt.apply_gradients(zip(grads, MLP.trainable_weights))
print("After optimization, network layer 1's weights is: \n{}".format(
    MLP.layer_list[0].weights.numpy()))


#Train your first neural network
#Load MNIST dataset
from tensorlayerx.dataflow import Dataset, DataLoader
X_train, y_train, X_val, y_val, X_test, y_test = tlx.files.load_mnist_dataset(
    shape=(-1, 784))

print("Train data shape:{}".format(X_train.shape))

#Prepare Dataset


class mnistdataset(Dataset):

    def __init__(self, data=X_train, label=y_train):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index].astype('float32')
        label = self.label[index].astype('int64')
        return data, label

    def __len__(self):
        return len(self.data)


#Build model
class CustomModel(tlx.nn.Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.linear1 = Linear(out_features=800, act=tlx.ReLU, in_features=784)
        self.linear2 = Linear(out_features=800, act=tlx.ReLU, in_features=800)
        self.linear3 = Linear(out_features=10, in_features=800)

    def forward(self, x):
        a = self.linear1(x)
        a = self.linear2(a)
        out = self.linear3(a)
        return out


MLP = CustomModel()

#Set training parameters
n_epoch = 50
batch_size = 128
print_freq = 1

#Set loss function and metric
loss_fn = tlx.losses.softmax_cross_entropy_with_logits
metric = tlx.metrics.Accuracy()

#Optimizer
optimizer = tlx.optimizers.Adam(lr=0.01)

train_dataset = mnistdataset(data=X_train, label=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#Use TensorlayerX high-level training API
#Warp training process
net_with_train = tlx.model.Model(
    network=MLP, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
print("Start training\n")
net_with_train.train(n_epoch=n_epoch, train_dataset=train_loader,
                     print_freq=print_freq, print_train_batch=False)
