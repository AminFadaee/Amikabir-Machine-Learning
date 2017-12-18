import numpy
import pandas
from math import e


def predict_probability(H, w):
    '''
    This function implements P(y=1|x,w)
    Args:
        H: The feature matrix
        w: the list of coefficients

    Returns:
        numpy array of sigmoid probabilities
    '''
    scores = numpy.dot(H, w)
    return 1 / (1 + e ** (-scores))


def features_derivatives(H, labels, w, l2_penalty):
    '''
    Computes the derivative of log likelihood
    
    Args:
        H: The feature matrix
        labels: list of labels of H instances
        w: the list of coefficients
        l2_penalty: the parameter 𝜆 for L2 regularization

    Returns:
        numpy array of derivatives of w
    '''
    H = pandas.DataFrame(H)
    y = numpy.array(labels)
    N, D = H.shape
    errors = (y == 1) - predict_probability(H, w)
    derivatives = numpy.zeros(D)
    for j in range(D):
        feature = H[j]
        derivatives[j] = numpy.dot(feature, errors) - 2 * l2_penalty * w[j] * (j != 0)
    return derivatives


def compute_log_likelihood(H, labels, w):
    '''
    Computes the log likelihood based on w
    
    Args:
        H: The feature matrix
        labels: list of labels of H instances
        w: the list of coefficients

    Returns:
        log likelihood
    '''
    y = numpy.array(labels)
    indicator = (y == +1)
    scores = numpy.dot(H, w)
    ll = numpy.sum((indicator - 1) * scores - numpy.log(1. + numpy.exp(-scores)))
    return ll


def logistic_regression(H, labels, w, step_size, max_iter, l2_penalty=0):
    '''
    Implements logistic regression classification
    
    Args:
        H: The feature matrix
        labels: list of labels of H instances
        w: the list of coefficients
        step_size: a parameter controlling the size of the gradient steps
        max_iter: number of iterations to run gradient ascent
        l2_penalty: the parameter 𝜆 for L2 regularization

    Returns:
        w, numpy array of coefficients
    '''
    for iter in range(max_iter):
        print(compute_log_likelihood(H, labels, w))
        w_next = w + step_size * features_derivatives(H, labels, w, l2_penalty)
        w = w_next
    return w


def predict(H, w, threshold):
    '''
    Predicts class (+1,-1) based on threshold and logistic regression probability
    Args:
        H: The feature matrix
        w: the list of coefficients
        threshold: the threshold for classification 

    Returns:
        list of labels
    '''
    N = len(H)
    labels = numpy.ones(N)
    probabilities = predict_probability(H, w)
    for i in range(N):
        if probabilities[i] <= threshold:
            labels[i] = -1
    return labels


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


def prediction_accuracy(labels, predictions):
    '''
    Derives the classification accuracy
    
    Args:
        labels: the real labels 
        predictions: the predicted labels

    Returns:
        accuracy
    '''
    labels = numpy.array(labels)
    predictions = numpy.array(predictions)
    return numpy.mean(labels == predictions)


def scale(H):
    '''
    Scales H based on standard normal distribution: (x-mu)/std
    Args:
        H: The feature matrix

    Returns:
        scaled data
    '''
    H = numpy.array(H)
    N, D = H.shape
    mu = numpy.zeros(D)
    std = numpy.zeros(D)
    for i in range(D):
        mu[i] = H[:, i].mean()
        std[i] = H[:, i].std()
        H[:, i] = (H[:, i] - mu[i]) / std[i]
    return H, mu, std


def parametric_scale(H, mu, std):
    '''
    Scales H based on standard normal distribution with predefined parameters
    Args:
        H: The feature matrix
        mu: list of means
        std: list of standard deviations

    Returns:
        scaled data
    '''
    H = numpy.array(H)
    N, D = H.shape
    for i in range(D):
        H[:, i] = (H[:, i] - mu[i]) / std[i]
    return H


def preprocessings(train, test):
    '''
    Scales and add bias feature for train and test data sets, also sets w and training labels
    Args:
        train: the training data set
        test: the test data set

    Returns:
        preprocessed training data, test data, their labels and w
    '''
    N, D = train.shape
    train_data = train[:, list(range(D - 1))]  # removing labels
    train_data, mu, std = scale(train_data)  # scaling the dataset
    train_data = numpy.concatenate((numpy.ones((N, 1)), train_data), axis=1)  # adding bias feature
    train_labels = train[:, D - 1]

    N, D = test.shape
    test_data = test[:, list(range(D - 1))]  # removing labels
    test_data = parametric_scale(test_data, mu, std)  # scaling the dataset
    test_data = numpy.concatenate((numpy.ones((N, 1)), test_data), axis=1)  # adding bias feature
    test_labels = test[:, D - 1]

    w = [0] * D

    return train_data, train_labels, test_data, test_labels, w