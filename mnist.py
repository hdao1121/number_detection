from six.moves import urllib
from sklearn.datasets import fetch_mldata
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
import os.path
import numpy as np

def load_mnist():
    mnist_path = "./mnist-original.mat"

    if not os.path.isfile(mnist_path):
        try:
            mnist = fetch_mldata('MNIST original')
        except urllib.error.HTTPError as ex:
            # Alternative method to load MNIST, if mldata.org is down
            mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
            response = urllib.request.urlopen(mnist_alternative_url)
            with open(mnist_path, "wb") as f:
                content = response.read()
                f.write(content)
    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    return  mnist
data = load_mnist()
X, y = data["data"], data["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

def show_random_digit(index):
    digit = X[index]
    digit = digit.reshape(28,28)

    plt.imshow(digit, cmap= matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
print("test")
