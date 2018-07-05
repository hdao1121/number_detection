from six.moves import urllib
from sklearn.datasets import fetch_mldata
from scipy.io import loadmat
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
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

# prepare data
X, y = data["data"], data["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

def show_random_digit(index):
    digit = X[index]
    digit = digit.reshape(28,28)

    plt.imshow(digit, cmap= matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

def binary_classification(number):
    y_bin_train = (y_train == number)
    y_bin_test = (y_train != number)

    sgd_clf = SGDClassifier(max_iter=5, tol=None)
    sgd_clf.fit(X_train, y_bin_train)

    # print("confusion matrix: \n",confusion_matrix(y_bin_train, y_train_pred), "\n")
    # print("f1 score: ", f1_score(y_bin_train, y_train_pred), "\n")

    y_scores = cross_val_predict(sgd_clf, X_train, y_bin_train, cv=3, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_bin_train, y_scores)

binary_classification(5)




def multiclass_classification():
    print("hello")