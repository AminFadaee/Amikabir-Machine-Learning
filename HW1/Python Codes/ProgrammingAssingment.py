import numpy
import pandas
from numpy import sqrt
from numpy.linalg import inv as inverse
from sklearn.model_selection import train_test_split
from numpy.matlib import identity


def compute_gradient_RSS_W(X, W, y):
    '''
    This function computes the gradient of RSS(W).
    Args:
        X: |NxD| Matrix of N observations with D features each
        W: |D| Vector of features
        y: |N| vector of true labels
    Returns:
        |D| Vector, the gradient of W
    '''
    if type(X) != numpy.matrix:
        X = numpy.matrix(X)
        if X.shape[0] == 1:  # In case there was one feature
            X = X.T
    if type(W) != numpy.matrix:
        W = numpy.matrix(W).T  # column vector
    elif W.shape[0] == 1:  # creating a column vector
        W = W.T
    if type(y) != numpy.matrix:
        y = numpy.matrix(y).T  # column vector
    elif y.shape[0] == 1:  # creating a column vector
        y = y.T
    N = X.shape[0]
    # delta_W = -2 * X.T * (y - X * W)
    delta_W = -X.T * (y - X * W) / N
    return delta_W


def magnitude(W):
    '''
    Computes the magnitude of the vector W
    Args:
        W: an arbitrary sized vector
    Returns:
        magnitude of W
    '''
    if type(W) != numpy.matrix:
        W = numpy.matrix(W).T
    elif W.shape[0] == 1:  # creating a column vector
        W = W.T
    return sqrt(float(W.T * W))


def add_bias_feature(X):
    '''
    Adds an all 1 feature to serve as the bias feature for W[0]
    Args:
        X: |NxD| Matrix of N observations with D features each
    Returns:
        H: |Nx(D+1)| numpy matrix
    '''
    if not type(X) == numpy.matrix:
        X = numpy.matrix(X)
        if X.shape[0] == 1:  # In case there was one feature
            X = X.T
    return numpy.hstack((numpy.ones((X.shape[0], 1)), X))


def graident_descent(X, W, y, step_size, max_iteration):
    '''
    Conducts the gradient descent and returns the minimum of the function RSS(W)
    Args:
        X: |NxD| Matrix of N observations with D features each
        W: |D| Vector of features
        y: |N| vector of true labels
        step_size: step size of the gradient
        tolerance: error tolerance (-1 if working with maximum iterations)
        max_iteration: number of maximum iterations (-1 if working with tolerance)

    Returns:
        |D| Vector minimizing the RSS
    '''
    if type(W) != numpy.matrix:
        W = numpy.matrix(W).T
    elif W.shape[0] == 1:  # creating a column vector
        W = W.T
    X = add_bias_feature(X)
    delta_W = compute_gradient_RSS_W(X, W, y)
    M = magnitude(delta_W)
    iteration = 1
    while iteration < max_iteration:
        if iteration % 1000 == 0:
            print('Iteration {0} with Magnitude of {1} for gradient(RSS(W))'.format(iteration, M))
        W = W - step_size * delta_W
        delta_W = compute_gradient_RSS_W(X, W, y)
        M = magnitude(delta_W)
        iteration += 1
    return list(map(float, W))


def polynomialize(data, column, order):
    '''
    Creates a new Dataframe containing new columns of column^order,column^(order-1),...,(column)^2
    
    Args:
        data: list/numpy matrix/ numpy array/ pandas Dataframe containing the data
        column: string representing a column in data
        order: order of the polynomialization

    Returns:
        new Dataframe
    '''
    data = pandas.DataFrame(data)
    new_data = data.copy()
    for r in range(2, order + 1):
        new_data['{0}_{1}'.format(column, r)] = new_data.apply(lambda row: float(row[column]) ** r, axis=1)
    return new_data


def closed_form(X, y, r=0):
    '''
    This function compute the regression coefficient using the Closed-form solution.

    Args:
        X: |NxD| Matrix of N observations with D features each
        y: |N| Vector of labels
        r: regularization size
    Returns:
        W: |D| Vector of features
    '''
    if not type(X) == numpy.matrix:
        X = numpy.matrix(X)
        if X.shape[0] == 1:  # In case there was one feature
            X = X.T
    if not type(y) == numpy.matrix:
        y = numpy.matrix(y).T
    elif y.shape[0] == 1:  # creating a column vector
        y = y.T
    X = add_bias_feature(X)
    N, D = X.shape

    I = identity(D)
    I[0, 0] = 0  # We are not going to regularize the intercept

    W = inverse(X.T * X + r * I) * X.T * y
    return list(map(float, W))


def predict(X, W):
    '''
    Predicts for H the labels, based on W 
    Args:
        X: |NxD| Matrix of N observations with D features each
        W: |D| Vector of features
    Returns:
        y: |N| Vector of predicted labels
    '''
    X = add_bias_feature(X)
    if not type(W) == numpy.matrix:
        W = numpy.matrix(W).T
    elif W.shape[0] == 1:  # creating a column vector
        W = W.T
    y = X * W
    return y


def MSE(y, X, W):
    '''
    Computes the Mean Squared Error.
    Args:
        y: |N| vector of true labels
        X: |NxD| Matrix of N observations with D features each
        W: |D| Vector of features
    Returns:
        float, Mean squared error
    '''
    if not type(y) == numpy.matrix:
        y = numpy.matrix(y).T
    elif y.shape[0] == 1:  # creating a column vector
        y = y.T
    N = y.shape[0]
    E = y - predict(X, W)
    return float((E.T * E) / (2 * N))


def find_mean_std(data, columns):
    '''
    Finds the mean(mu) and standard deviation of some columns in data specified by the input
    Args:
        data: list/numpy matrix/ numpy array/ pandas Dataframe containing the data
        columns: list of strings representing desired columns in data

    Returns:
        mu and std, lists of floats
    '''
    data = pandas.DataFrame(data)
    columns = list(columns)
    mu = []
    std = []
    for i in range(len(columns)):
        index = columns[i]
        mu.append(numpy.mean(data[[index]]))
        std.append(numpy.std(data[[index]]))
    return mu, std


def read_split_data(path):
    '''
    Reads in the data from path and split the data into train and test (80-20)
    Args:
        path: string representing path to the data file
    Returns:
        training and test data
    '''
    data = pandas.read_excel(path)
    data.columns = ['x', 'y']
    train, test = train_test_split(data, train_size=0.8, random_state=0)
    return train, test


def scale(data, columns, mu, std):
    '''
    Scales specified columns of data based on standard normal distribution: (x-mu)/std
    Args:
        data: list/numpy matrix/ numpy array/ pandas Dataframe containing the data
        columns: list of strings representing desired columns in data
        mu: list of floats containing mean for each of the specified columns
        std: list of floats containing standard deviation for each of the specified columns

    Returns:
        scaled data
    '''
    data = pandas.DataFrame(data)
    columns = list(columns)
    for i in range(len(columns)):
        index = columns[i]
        data[[index]] = data[[index]].apply(lambda row: (row - mu[i]) / std[i], axis=1)
    return data


def preprocessing(train, test, plotting_data, order):
    '''
    Makes the data ready for linear regression by polynomialization and scaling
    Args:
        train: training data
        test: test data
        plotting_data: data used for plotting the fit function
        order: order of polynomial

    Returns:
        training and test data, mu and standard deviation of each columns, plotting data and name of each column
    '''
    train = pandas.DataFrame(train)
    test = pandas.DataFrame(test)
    plotting_data = pandas.DataFrame(plotting_data)

    column_names = ['x'] + list('x_{0}'.format(i) for i in range(2, order + 1))[:order]
    train = polynomialize(train, 'x', order=order)
    test = polynomialize(test, 'x', order=order)
    mu, std = find_mean_std(train, column_names)
    train = scale(train, column_names, mu, std)
    test = scale(test, column_names, mu, std)
    plotting_data = polynomialize(plotting_data, 'x', order)
    plotting_data = scale(plotting_data, column_names, mu, std)
    return train, test, mu, std, plotting_data, column_names
