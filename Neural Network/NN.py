import numpy as np
from math import log2

def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_derivative(z):
    # return np.multiply(sigmoid(z), 1-sigmoid(z))
    return z * (1 - z)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exps = np.exp(x - np.max(x))
    # return exps / exps.sum(axis=0)
    return exps / np.sum(exps)


def softmax_derivative(x):
    """ Compute derivative of softmax for each score in smax."""
    # s = smax.reshape(-1, 1)
    # return np.diagflat(s) - np.dot(s, s.T)
    exps = np.exp(x - np.max(x))
    return (exps * (np.sum(exps) - exps)) / np.sum(exps) ** 2

def cross_entropy_softmax_derivative(label, smax):
    """Compute derivative of softmax values for each sets of scores in x."""
    return np.subtract(smax, label)


def cross_entropy(label, smax):
    # ones = np.ones(len(label))
    # ce1 = np.multiply(label, np.log(smax))
    # print(ce1)
    # print(1 - label)
    # ce2 = np.multiply(1 - label, np.log(1 - smax))
    # print(ce2)
    # ce3 = np.add(ce1, ce2)
    # result = -sum(ce3)
    # return result
    return -np.sum(np.add(np.multiply(label, np.log(smax)), np.multiply(1 - label, np.log(1 - smax))))
    # return -sum(np.multiply(label, np.log(smax)))
    # x = np.multiply(label, np.log(smax))
    # print(x)
    # return -sum(x)


def cross_entropy_derivative(label, smax):
    return -np.add(np.divide(label, smax), np.divide(1-label, 1-smax))

class NeuralNetwork:
    def __init__(self, x, y):
        # self.input      = x
        # self.weights1   = np.random.rand(25, self.input.shape[0])
        # self.weights2   = np.random.rand(10, 25)
        self.bias = 1
        # self.y          = y
        # self.output     = y

    def feedforward_hidden_layer(self, input, weights):
        layer = sigmoid(np.dot(weights, input))
        return layer
        # print(self.weights1.shape , self.input.shape, self.weights2.shape)
        # self.layer1 = sigmoid(np.dot(self.weights1, self.input) + self.bias)
        # self.output = sigmoid(np.dot(self.weights2, self.layer1) + self.bias)

    def feedforward_output_layer(self, input, weights):
        layer = softmax(np.dot(weights, input))
        return layer

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    train_image = np.genfromtxt('unit_image.csv', delimiter=',')
    train_label = np.genfromtxt('unit_label.csv', delimiter=',')
    # Normalize the pixel value from [0,255] to [0,1]
    # train_image = (train_image / 255) - 0.5
    num_hidden_layers = 1
    num_neurons_input = 784
    num_neurons_output = 10
    num_neurons_layer = [800]
    weights = []

    ones = np.ones(1)
    for l in range(num_hidden_layers + 1):
        if l == num_hidden_layers:
            weights.append(np.random.rand(num_neurons_output, num_neurons_layer[l-1]))
        elif l == 0:
            weights.append(np.random.rand(num_neurons_layer[l], num_neurons_input))
        else:
            weights.append(np.random.rand(num_neurons_layer[l], num_neurons_layer[l-1]))
    results = list()
    for i in range(10):
        try:
            image = train_image[i]
            image = np.array(image, dtype='float')
            image.shape = (len(image), 1)

            lab = np.int32(train_label[i])
            label = np.zeros(num_neurons_output)
            label[lab] = 1
            label.shape = (len(label), 1)

            nn = NeuralNetwork(image, label)

            a = []
            z = []
            a.append(image)
            # a[0] = np.vstack((ones, image))
            # input = image
            # input = np.vstack((ones, image))

            # Continue with feedforward based on num of hidden layers
            # and num of neurons in each layer until output layer is reached.
            for l in range(num_hidden_layers + 1):
                if l == num_hidden_layers:
                    # weight = np.random.rand(num_neurons_output, input.shape[0])
                    # weight = np.zeros([num_neurons_output, input.shape[0]], dtype=float)
                    # output = nn.feedforward_output_layer(input, weights[l])
                    # input = np.hstack((ones, input))
                    z.append(np.dot(weights[l], a[l]))
                    o = softmax(z[l])
                    # o = nn.feedforward_output_layer(a[l], weights[l])
                else:
                    # weight = np.random.rand(num_neurons_layer[l], input.shape[0])
                    # weight = np.zeros([num_neurons_layer[l], input.shape[0]], dtype=float)
                    # input = nn.feedforward_hidden_layer(input, weights[l])
                    # input = np.vstack((ones, input))
                    # a.append(nn.feedforward_hidden_layer(a[l], weights[l]))
                    z.append(np.dot(weights[l], a[l]))
                    a.append(sigmoid(z[l]))
                    # a[l+1] = np.vstack((ones, a[l+1]))

            # print(o)

            # Calculate the cross entropy between actual and expected output
            # for i in range(len(label)):
            #     predicted = [1.0 - output[i], output[i]]
            #     expected = [1.0 - label[i], label[i]]
            #     ce = cross_entropy(expected, predicted)
            #     print("CE:", ce)
            #     results.append(ce)
            #     # cross_entropy = np.multiply(label, np.log2(output))

            ce = cross_entropy(label, o)
            # ce = -np.log(o[lab])
            print("CE_LOSS:", ce)
            results.append(ce)

            # ce_grad = - 1 / o[lab]
            ce_grad = cross_entropy_derivative(label, o)
            # print("CE_GRAD:", ce_grad)
            # output_grad = softmax_derivative(output)
            # o_grad = cross_entropy_softmax_derivative(label, o)
            o_grad = softmax_derivative(z[num_hidden_layers])
            # print("SMAX_GRAD:", o_grad)

            weights_grad = []


            weights_grad.append(np.dot(np.multiply(ce_grad, o_grad), a[num_hidden_layers].T))
            weights_grad.append(np.dot(np.multiply(np.dot(weights[num_hidden_layers].T, np.multiply(ce_grad, o_grad)), sigmoid_derivative(a[num_hidden_layers - 1])), a[num_hidden_layers - 1].T))

            weights_grad.reverse()
            for l in range(num_hidden_layers + 1):
                # print("WEIGHT_GRAD:", weights_grad[l].shape)
                weights[l] += weights_grad[l]
                # print("WEIGHTS:", weights[l].shape)

            # back_weights_layer = []
            # for l in range(num_hidden_layers, -1, -1):
            #     backprop_weights = []
            #     if l == num_hidden_layers:
            #         for i in range(num_neurons_output):
            #             for j in range(num_neurons_layer[l]):
            #                 if i == lab:
            #                     backprop_weights.append((o[i] - 1) * a[l][j])
            #                 else:
            #                     backprop_weights.append(o[i] * a[l][j])
            #         backprop_weights.reshape(num_neurons_output, num_neurons_layer[l])
            #     else:
            #
            #     weights[l] = np.add(weights[l], backprop_weights)
            #     back_weights_layer.append(backprop_weights)
            # print(output_grad)
            # nn.backprop()
            # pixels = image.reshape((28, 28))
            # plt.imshow(pixels)
            # plt.show()

        except Exception as ex:
            print(ex)
    cross_entropy_loss = np.mean(results)
    print("RESULT:", results)
    print("MEAN RESULT:", cross_entropy_loss)