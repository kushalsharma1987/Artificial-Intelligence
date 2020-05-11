import sys
import time
import numpy as np


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def sigmoid_derv(s):
    return s * (1 - s)


def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy_softmax_derivative(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res / n_samples


def cross_entropy_loss(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp) / n_samples
    return loss


def get_accuracy(images, labels):
    acc = 0
    for img, labl in zip(images, labels):
        s = model.predict(img)
        if s == np.argmax(labl):
            acc += 1
    return acc / len(images) * 100


def get_predictions(test_images):
    test_predictions = []
    for image in test_images:
        pred_label = model.predict(image)
        test_predictions.append(pred_label)
    return test_predictions


class NeuralNetwork:
    def __init__(self, x, y):
        # self.x = x
        # neurons = 128
        # neurons = 1568
        # neurons = 3136
        self.learning_rate = learn_rate
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.weight1 = np.random.randn(ip_dim, neurons)
        self.bias1 = np.zeros((1, neurons))
        self.weight2 = np.random.randn(neurons, neurons)
        self.bias2 = np.zeros((1, neurons))
        self.weight3 = np.random.randn(neurons, op_dim)
        self.bias3 = np.zeros((1, op_dim))
        # self.y = y

    def feedforward(self, x):
        z1 = np.dot(x, self.weight1) + self.bias1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.weight2) + self.bias2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.weight3) + self.bias3
        self.a3 = softmax(z3)

    def backprop(self, x, y):
        loss = cross_entropy_loss(self.a3, y)
        print('Error :', loss)
        a3_grad = cross_entropy_softmax_derivative(self.a3, y)  # weight3
        z2_grad = np.dot(a3_grad, self.weight3.T)
        a2_grad = z2_grad * sigmoid_derv(self.a2)  # weight2
        z1_grad = np.dot(a2_grad, self.weight2.T)
        a1_grad = z1_grad * sigmoid_derv(self.a1)  # weight1

        # if loss < 2.0:
        #     self.learning_rate = 0.5
        # elif loss < 1.0:
        #     self.learning_rate = 0.3
        # elif loss < 0.5:
        #     self.learning_rate = 0.2
        # elif loss < 0.1:
        #     self.learning_rate = 0.1

        if loss < .2:
            self.learning_rate = 0.5
        elif loss < 0.02:
            self.learning_rate = 0.3
        elif loss < 0.002:
            self.learning_rate = 0.2
        elif loss < 0.0002:
            self.learning_rate = 0.1

        self.weight3 -= self.learning_rate * np.dot(self.a2.T, a3_grad)
        self.bias3 -= self.learning_rate * np.sum(a3_grad, axis=0, keepdims=True)
        self.weight2 -= self.learning_rate * np.dot(self.a1.T, a2_grad)
        self.bias2 -= self.learning_rate * np.sum(a2_grad, axis=0)
        self.weight1 -= self.learning_rate * np.dot(x.T, a1_grad)
        self.bias1 -= self.learning_rate * np.sum(a1_grad, axis=0)
        return loss

    def predict(self, data):
        # self.x = data
        self.feedforward(data)
        return self.a3.argmax()


if __name__ == "__main__":
    start_time = time.time()
    len_arg = len(sys.argv)

    print("Arguments passed:", end=" ")
    for i in range(len_arg):
        print(sys.argv[i], end=" ")

    if len_arg > 1:
        train_image_file = sys.argv[1]
        train_label_file = sys.argv[2]
        test_image_file = sys.argv[3]
    elif len_arg == 1:
        train_image_file = 'train_image.csv'
        train_label_file = 'train_label.csv'
        test_image_file = 'test_image.csv'
        # test_label_file  = 'test_label.csv'

    tr_image = np.genfromtxt(train_image_file, delimiter=',')
    train_image = tr_image / 255.0
    tr_label = np.genfromtxt(train_label_file, delimiter=',', dtype=int)
    train_label = np.zeros((tr_label.size, 10))
    train_label[np.arange(tr_label.size), tr_label] = 1
    # print(train_label)
    te_image = np.genfromtxt(test_image_file, delimiter=',')
    test_image = te_image / 255.0
    # te_label = np.genfromtxt('test_label.csv', delimiter=',', dtype=int)
    # test_label = np.zeros((te_label.size, 10))
    # test_label[np.arange(te_label.size), te_label] = 1

    batch_size = 50
    neurons = 1568
    learn_rate = 0.7
    epochs = 20

    if len(train_image) < 15000:
        epochs = 40
    elif len(train_image) < 20000:
        epochs = 35
    elif len(train_image) < 30000:
        epochs = 30
    elif len(train_image) < 40000:
        epochs = 25

    print("Training Dataset Size:", len(train_image))
    print("Test Dataset Size:", len(test_image))
    print("Batch Size:", batch_size)
    print("Learning Rate:", learn_rate)
    print("Epochs:", epochs)
    print("Hidden Units:", neurons)

    model = NeuralNetwork(train_image, np.array(train_label))

    batch_image = []
    batch_label = []

    num_examples = len(train_image)

    for l in range(int(num_examples / batch_size)):
        batch_image.append(train_image[l * batch_size:(l + 1) * batch_size])
        batch_label.append(train_label[l * batch_size:(l + 1) * batch_size])
    if num_examples // batch_size != 0:
        batch_image.append(train_image[l * batch_size:])
        batch_label.append(train_label[l * batch_size:])

    loss_value = []
    runs = 0
    for x in range(epochs):
        for b in range(len(batch_image)):
            model.feedforward(batch_image[b])
            loss = model.backprop(batch_image[b], batch_label[b])
            runs += 1
            loss_value.append(loss)

    train_acc = get_accuracy(train_image, np.array(train_label))
    print("Training accuracy : ", train_acc)

    test_predictions = get_predictions(test_image)

    test_predictions_array = np.array(test_predictions)
    np.savetxt("test_predictions.csv", test_predictions_array, delimiter=",", fmt='%d')

    end_time = time.time()
    print('Processing time: {}s'.format(round(end_time - start_time, 7)))

    print("Total runs:", runs)
    # x = np.arange(runs)
    # plt.plot(x, loss_value, label='Loss Value')
    # # plt.plot(test_accuracy, y, label='test_acc')
    # plt.xlabel('Batches')
    # plt.ylabel('Loss')
    # plt.title('Loss Curve')
    # plt.legend
    # plt.show()
