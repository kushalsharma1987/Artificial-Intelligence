import pickle
import gzip
import numpy as np

def load_data():
    with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
        train_image = f.read()
    with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
        train_labels = f.read()
    with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_image = f.read()
    with gzip.open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_label = f.read()
    return train_image, train_labels, test_image, test_label
# def load_data():
#     f = gzip.open('mnist.pkl.gz', 'rb')
#     training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
#     f.close()
#     return (training_data, validation_data, test_data)
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# training_data, validation_data, test_data=load_data()
# training_img=np.int32(training_data[0]*256)
# training_label=np.int32(training_data[1])
# validation_img = np.int32(validation_data[0]*256)
# validation_label=np.int32(validation_data[1])
# test_img=np.int32(test_data[0]*256)
# test_label=np.int32(test_data[1])
#
# tr_img = np.concatenate((training_img,validation_img))
# tr_label = np.concatenate((training_label,validation_label))
#
# tr_label=np.reshape(tr_label,(60000,1))
# test_label=np.reshape(test_label,(10000,1))

train_images, train_labels, test_images, test_labels =load_data()
training_img=np.int32(train_images*256)
training_label=np.int32(train_labels)
test_img=np.int32(test_images*256)
test_label=np.int32(test_labels)

training_label=np.reshape(training_label,(60000,1))
test_label=np.reshape(test_label,(10000,1))

np.savetxt('train_label.csv',training_label,delimiter=',', fmt='%d')
np.savetxt('test_label.csv',test_label,delimiter=',', fmt='%d')
np.savetxt('train_image.csv',training_img,delimiter=',', fmt='%d')
np.savetxt('test_image.csv',test_img,delimiter=',', fmt='%d')

