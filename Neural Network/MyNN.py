# import pandas as pd
import time
import sys
import matplotlib.pyplot as plt
# import csv
import numpy as np
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split

# dig = load_digits()
# onehot_target = pd.get_dummies(dig.target)
# x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)




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


def get_acc(x, y):
    test_predictions = []
    acc = 0
    for xx, yy in zip(x, y):
        s = model.predict(xx)
        test_predictions.append(int(s))
        if s == np.argmax(yy):
            acc += 1
    return test_predictions, acc / len(x) * 100


def get_predictions(test_images):
    test_predictions = []
    for image in test_images:
        pred_label = model.predict(image)
        test_predictions.append(int(pred_label))
    return test_predictions


class MyNN:
    def __init__(self, x, y):
        # self.x = x
        # neurons = 128
        # neurons = 1568
        # neurons = 3136
        self.lr = learn_rate
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.w1 = np.random.randn(ip_dim, neurons)
        self.b1 = np.zeros((1, neurons))
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, op_dim)
        self.b3 = np.zeros((1, op_dim))
        # self.y = y

    def feedforward(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)

    def backprop(self, x, y):
        loss = cross_entropy_loss(self.a3, y)
        print('Error :', loss)
        a3_delta = cross_entropy_softmax_derivative(self.a3, y)  # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2)  # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1)  # w1

        if loss < 2:
            self.lr = 0.5
        elif loss < 0.2:
            self.lr = 0.3
        elif loss < 0.02:
            self.lr = 0.2
        elif loss < 0.0002:
            self.lr = 0.1

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

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
        test_image_file  = sys.argv[3]
        test_label_file = 'test_label.csv'
    elif len_arg == 1:
        train_image_file = 'unit_train_image.csv'
        train_label_file = 'unit_train_label.csv'
        test_image_file  = 'test_image.csv'
        test_label_file  = 'test_label.csv'

    tr_image = np.genfromtxt(train_image_file, delimiter=',')
    train_image = tr_image / 255.0
    tr_label = np.genfromtxt(train_label_file, delimiter=',', dtype=int)
    train_label = np.zeros((tr_label.size, 10))
    train_label[np.arange(tr_label.size), tr_label] = 1
    # print(train_label)
    te_image = np.genfromtxt(test_image_file, delimiter=',')
    test_image = te_image / 255.0
    te_label = np.genfromtxt(test_label_file, delimiter=',', dtype=int)
    test_label = np.zeros((te_label.size, 10))
    test_label[np.arange(te_label.size), te_label] = 1
    # print(test_label)

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

    model = MyNN(train_image, np.array(train_label))

    noise = np.random.normal(0, .2, (train_image.shape))
    train_image = train_image + noise
    batch_image = []
    batch_label = []
    num_examples = len(train_image)

    for l in range(int(num_examples/ batch_size)):
        batch_image.append(train_image[l*batch_size:(l+1)*batch_size])
        batch_label.append(train_label[l*batch_size:(l+1)*batch_size])
    if num_examples // batch_size != 0:
        batch_image.append(train_image[l*batch_size:])
        batch_label.append(train_label[l * batch_size:])

    loss_value = []
    training_accuracy = []
    test_accuracy = []
    runs = 0
    for x in range(epochs):
        for b in range(len(batch_image)):
            model.feedforward(batch_image[b])
            loss = model.backprop(batch_image[b], batch_label[b])
            runs += 1
            loss_value.append(loss)
        # train_acc = get_acc(train_image, np.array(train_label))
        # test_acc = get_acc(test_image, np.array(test_label))
        # training_accuracy.append(train_acc)
        # test_accuracy.append(test_acc)
        # print("Training accuracy after ", x, " epochs: ", train_acc)
        # print("Test accuracy after ", x, " epochs: ", test_acc)

    # y = np.arange(epochs * batch_size)

    train_predictions, train_acc = get_acc(train_image, np.array(train_label))
    print("Training accuracy : ", train_acc)

    test_predictions, test_acc = get_acc(test_image, np.array(test_label))
    print("Test accuracy : ", test_acc)

    test_predictions_array = np.array(test_predictions)

    np.savetxt("test_predictions.csv", test_predictions_array, delimiter=",", fmt='%d')

    # filename = "test_predictions.csv"
    # with open(filename, 'w') as csvfile:
    #     # creating a csv writer object
    #     csvwriter = csv.writer(csvfile)
    #     # writing the data rows
    #     csvwriter.writerows(test_predictions)

    end_time = time.time()
    print('Processing time: {}s'.format(round(end_time - start_time, 7)))

    print("Total runs:", runs)
    x = np.arange(runs)
    plt.plot(x, loss_value, label='Loss Value')
    # plt.plot(test_accuracy, y, label='test_acc')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend
    plt.show()