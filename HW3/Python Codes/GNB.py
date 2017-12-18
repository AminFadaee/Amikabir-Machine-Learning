import numpy
import pandas
from scipy.stats import multivariate_normal
from math import log
import matplotlib.pyplot as plt


def preprocessings(train, test):
    '''
    Splits the data to features and labels 
    
    Args:
        train: the training data set
        test: the test data set

    Returns:
        training data, test data, their labels 
    '''
    N, D = train.shape
    train_data = train[:, list(range(D - 1))]  # removing labels
    train_labels = train[:, D - 1]

    N, D = test.shape
    test_data = test[:, list(range(D - 1))]  # removing labels
    test_labels = test[:, D - 1]

    return train_data, train_labels, test_data, test_labels


def stats(data, label):
    '''
    Finds mu and covariance matrix and prior probability for each labels (assuming 2 one of which being 1) of data
    Args:
        data: feature matrix
        label: labels corresponding to data

    Returns:
        mu0, cov0,prior0, mu1, cov1,prior1
    '''
    _, D = data.shape
    data0 = data[label != 1]
    data1 = data[label == 1]

    prior0 = len(data0) / len(data)
    prior1 = len(data1) / len(data)

    mu0 = numpy.zeros(D)
    mu1 = numpy.zeros(D)
    cov0 = numpy.zeros((D, D))
    cov1 = numpy.zeros((D, D))

    small_error = 0.00001  # we add this to covariance matrix to avoid non-singular matrix
    for i in range(D):
        mu0[i] = data0[:, i].mean()
        cov0[i, i] = data0[:, i].var() + small_error
        mu1[i] = data1[:, i].mean()
        cov1[i, i] = data1[:, i].var() + small_error

    return mu0, cov0, prior0, mu1, cov1, prior1


def conditional(mu, cov, x):
    '''
    Finds the conditional probability p(x|w) assuming a gaussian underlying distribution
    Args:
        mu: mean of data
        cov: covariance matrix of data
        prior: prior probability of data
        x: instance to find the posterior of 

    Returns:
        unnormalized posterior probability
    '''
    return multivariate_normal(mu, cov).logpdf(x)


def gnb_classify(mu0, cov0, prior0, mu1, cov1, prior1, x, threshold=0):
    '''
    Classifies x into 1 or zero using the stats provided via gaussian naive bayes procedure
    
    Args:
        mu0: mean of class 0
        cov0: covariance matrix of class 0
        mu1: mean of class 1
        cov1: covariance matrix of class 1
        x: instance to classify
        threshold: threshold to classify upon (-1<=threshold<=1)

    Returns:
        0/1
    '''
    c0 = conditional(mu0, cov0, x)
    c1 = conditional(mu1, cov1, x)
    lp0 = log(prior0)
    lp1 = log(prior1)
    normalizing_factor = c0 + lp0 + c1 + lp1
    A = (c0 + lp0) / normalizing_factor
    B = (c1 + lp1) / normalizing_factor
    return (A - B) >= threshold


def false_positive_rate(labels, predictions):
    '''
    Computes the false positive rate of the prediction

    Args:
        labels: the real labels 
        predictions: the predicted labels

    Returns:
        true positive rate
    '''
    labels = numpy.array(labels)
    predictions = numpy.array(predictions)
    N = numpy.sum(labels == -1)
    TN = numpy.sum((labels == -1) * (predictions == -1))
    return 1 - TN / N


def true_positive_rate(labels, predictions):
    '''
    Computes the true positive rate of the prediction

    Args:
        labels: the real labels 
        predictions: the predicted labels

    Returns:
        true positive rate
    '''
    labels = numpy.array(labels)
    predictions = numpy.array(predictions)
    P = numpy.sum(labels == +1)
    TP = numpy.sum((labels == +1) * (predictions == +1))
    return TP / P


# Reading the data sets
train = numpy.array(pandas.read_csv('Train_Data.csv'), dtype='float64')
test = numpy.array(pandas.read_csv('Test_Data.csv'), dtype='float64')

# Preprocessings
train_data, train_labels, test_data, test_labels = preprocessings(train, test)
mu0, cov0, prior0, mu1, cov1, prior1 = stats(train_data, train_labels)

# Training Accuracy
correct = 0
for i, row in enumerate(train_data):
    if gnb_classify(mu0, cov0, prior0, mu1, cov1, prior1, row, threshold=0) == int(train_labels[i] == 1):
        correct += 1
    print(len(train_data) - i)
print('Training Accuracy:', correct / len(train_data))

# Test Accuracy
correct = 0
for i, row in enumerate(test_data):
    if gnb_classify(mu0, cov0, prior0, mu1, cov1, prior1, row, threshold=0) == int(test_labels[i] == 1):
        correct += 1
    print(len(test_data) - i)
print('Test Accuracy:', correct / len(test_data))

# Training Accuracy: 0.804375
# Test Accuracy: 0.7225


# Plotting the ROC
points = numpy.zeros((2, 101))
for k, threshold in enumerate(numpy.linspace(-1, 1, 21)):
    print(threshold)
    prediction = []
    for i, row in enumerate(test_data):
        if gnb_classify(mu0, cov0, prior0, mu1, cov1, prior1, row, threshold=threshold) == 1:
            prediction.append(1)
        else:
            prediction.append(-1)

    points[0][k] = false_positive_rate(test_labels, prediction)
    points[1][k] = true_positive_rate(test_labels, prediction)
    print(points[0][k], points[1][k])

plt.plot(points[0], points[1])
plt.show()
