from six.moves import urllib
from sklearn.datasets import fetch_mldata
from scipy.io import loadmat
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
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
    print(precisions[:-1].shape, precisions.shape)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def binary_classification(number, randomState):
    y_bin_train = (y_train == number)
    y_bin_test = (y_train != number)

    # # SGDClassifier
    # sgd_clf = SGDClassifier(max_iter=5, tol=None, random_state=randomState)
    # sgd_clf.fit(X_train, y_bin_train)
    # y_scores_sgd = cross_val_predict(sgd_clf, X_train, y_bin_train, cv=3, method="decision_function")
    #
    # # 90% precision, threshold is determined after looking through the plots
    # precisions, recalls, thresholds = precision_recall_curve(y_bin_train, y_scores_sgd)
    # # plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    # y_bin_train_pred_90 = (y_scores_sgd > 70000)
    # print('Precision score with 70,000 threshold:', precision_score(y_bin_train, y_bin_train_pred_90), "\n")

    # RandomForestClassifer
    forest_clf = RandomForestClassifier(random_state=randomState)
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_bin_train, cv=3, method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_bin_train, y_scores_forest)

    # ROC curve
    fpr_sgd, tpr_sdg, thresholds_sgd = roc_curve(y_bin_train, y_scores_sgd)
    # plot_roc_curve(fpr, tpr, thresholds)
    print("ROC score for SDGClassifer:", roc_auc_score(y_bin_train, y_scores_sgd))



binary_classification(5, 42)




def multiclass_classification():
    print("hello")