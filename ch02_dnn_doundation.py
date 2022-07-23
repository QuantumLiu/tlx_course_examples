import tensorlayerx as tlx
# Single neuro

# matrix multiplication
x = tlx.constant([[1, 2, 3]])
w = tlx.constant([[-0.5], [0.2], [0.1]])

print("X:\n", x, "\nShape:", x.shape)
print("W:\n", w, "\nShape", w.shape)

# Bias
b1 = tlx.constant(0.5)

z1 = tlx.matmul(x, w)+b1

print("Z:\n", z1, "\nShape", z1.shape)

# Two outputs

x = tlx.constant([[1, 2, 3]])
w = tlx.constant([[-0.5, -0.3],
                  [0.2, 0.4],
                  [0.1, 0.15]])

print("X:\n", x, "\nShape:", x.shape)
print("W:\n", w, "\nShape", w.shape)

# Bias
b2 = tlx.constant([0.5,0.4])

z2 = tlx.matmul(x, w)+b2

print("Z:\n", z2, "\nShape", z2.shape)


# Activation
# Sigmoid function
a1 = tlx.nn.activation.Sigmoid()(z1)
print("Result tlx sigmoid:", a1)

# define your own activation function


class ActSigmoid(tlx.nn.Module):
    def forward(self, x):
        return 1 / (1 + tlx.exp(-x))


a1_m = ActSigmoid()(z1)

print("Result your own sigmoid:", a1_m)

# Softmax
a2 = tlx.softmax(z2, axis=-1)
print("Result tlx softmax:", a2)


class ActSoftmax(tlx.nn.Module):
    def forward(self, x, axis=-1):
        e = tlx.exp(x - tlx.reduce_max(x, axis=axis, keepdims=True))
        s = tlx.reduce_sum(e, axis=axis, keepdims=True)
        return e / s


a2_m = ActSoftmax()(z2, axis=-1)
print("Result your own softmax:", a2_m)
