# # from scipy.io import loadmat
# import numpy as np
# # import scipy.optimize as opt
# import pandas as pd

# # data = loadmat('ex4data1.mat')
# data = pd.read_csv('test_image.csv', delimiter=',')
# X = data['X']
# y = data['y']# visualizing the data
# _, axarr = plt.subplots(10,10,figsize=(10,10))
# for i in range(10):
#     for j in range(10):
#        axarr[i,j].imshow(X[np.random.randint(X.shape[0])].\
# reshape((28,28), order = 'F'))
#        axarr[i,j].axis('off')

import numpy as np
import matplotlib.pyplot as plt# reading the data

my_data = np.genfromtxt('test_image.csv', delimiter=',')

for i in range(1, 5):
    try:
        image = my_data[i]
        image = np.array(image, dtype='int32')
        pixels = image.reshape((28, 28))
        plt.imshow(pixels)
        plt.show()

    except Exception as ex:
        print(ex)