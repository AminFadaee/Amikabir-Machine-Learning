import numpy
import pandas
import random


def euclidean_distance(p1, p2):
    '''
    Computes the euclidean distance between p1 and p2

    Args:
        p1: an array like collection representing an n dimensional point
        p2: an array like collection representing an n dimensional point

    Returns:
        float, the euclidean distance
    '''
    p1 = numpy.array(p1)
    p2 = numpy.array(p2)
    return numpy.sqrt(numpy.sum((p1 - p2) ** 2))


def manhattan_distance(p1, p2):
    '''
    Computes the manhattan distance between p1 and p2

    Args:
        p1: an array like collection representing an n dimensional point
        p2: an array like collection representing an n dimensional point

    Returns:
        float, the manhattan distance
    '''
    p1 = numpy.array(p1)
    p2 = numpy.array(p2)
    return numpy.sum(numpy.abs(p1 - p2))


def cosine_distance(p1, p2):
    '''
    Computes the cosine distance between p1 and p2

    Args:
        p1: an array like collection representing an n dimensional point
        p2: an array like collection representing an n dimensional point

    Returns:
        float, the cosine distance
    '''
    p1 = numpy.array(p1)
    p2 = numpy.array(p2)
    return 1 - numpy.dot(p1, p2) / (numpy.linalg.norm(p1) * numpy.linalg.norm(p2))


def minkowski_distance(p1, p2, p):
    '''
    Computes the minkowski distance between p1 and p2 with parameter p

    Args:
        p1: an array like collection representing an n dimensional point
        p2: an array like collection representing an n dimensional point
        p: p parameter in minkowski distance

    Returns:
        float, the minkowski distance
    '''
    p1 = numpy.array(p1)
    p2 = numpy.array(p2)
    return (numpy.sum(numpy.abs(p1 - p2))) ** (1 / p)


def KNN(data, labels, test, k, distance=euclidean_distance):
    '''
    Computes the KNN classification on with the provided distance function
    
    Args:
        data: DataFrame of shape (N,D) of independent variables
        labels: DataSeries of size N of the dependent variable
        test: data for prediction of class
        k: number of nearest neighbors
        distance: a function to be used as distance measure

    Returns:
        predicting label
    '''
    distances = []
    for i in range(data.shape[0]):
        row = data.iloc[i]
        distances.append((distance(row, test), i, labels.iloc[i]))
    N = pandas.Series(map(lambda j: j[2], sorted(distances, key=lambda i: i[0])[:k]))
    prediction = numpy.argmax(N.value_counts())
    return prediction


def cross_validation_best_k(data, labels, options, distance=euclidean_distance):
    '''
    By performing 10-Fold Cross Validation, finds the optimal k for KNN.
    
    Args:
        data: DataFrame of shape (N,D) of independent variables
        labels: Series of size N of the dependent variable
        options: list of options for the value of k
        distance: a function to be used as distance measure

    Returns:
        the optimal k among options
    '''
    N, D = data.shape
    random.seed(0)
    index = list(range(N))
    random.shuffle(index)

    accuracy = []
    for k in options:
        accuracy_for_c = []
        for c in range(10):
            test_indices = index[c * N // 10:(c + 1) * N // 10]
            train_indecies = index[:c * N // 10] + index[(c + 1) * N // 10:]
            test_data = pandas.DataFrame(data, index=test_indices)
            train_data = pandas.DataFrame(data, index=train_indecies)
            test_labels = pandas.Series(labels, index=test_indices)
            train_labels = pandas.Series(labels, index=train_indecies)
            predictions = []
            for i in range(len(test_data)):
                prediction = KNN(train_data, train_labels, test_data.iloc[i], k, distance)
                predictions.append(prediction)
            accuracy_for_c.append(numpy.mean(numpy.array(predictions) == numpy.array(test_labels)))
        accuracy.append(numpy.mean(accuracy_for_c))
    print(accuracy)
    return options[numpy.argmax(accuracy)]


def cross_validation_distances(data, labels, distance_functions, k=5):
    '''
    By performing 10-Fold Cross Validation, finds the test error of KNN for different distance measures.
    
    Args:
        data: DataFrame of shape (N,D) of independent variables
        labels: Series of size N of the dependent variable
        distance_functions: list of distance functions to be used for KNN
        k: number of nearest neighbors

    Returns:
        list of accuracies corresponding to each distance measure.
    '''
    N, D = data.shape
    random.seed(0)
    index = list(range(N))
    random.shuffle(index)

    accuracy = []
    for distance in distance_functions:
        accuracy_for_c = []
        for c in range(10):
            test_indices = index[c * N // 10:(c + 1) * N // 10]
            train_indecies = index[:c * N // 10] + index[(c + 1) * N // 10:]
            test_data = pandas.DataFrame(data, index=test_indices)
            train_data = pandas.DataFrame(data, index=train_indecies)
            test_labels = pandas.Series(labels, index=test_indices)
            train_labels = pandas.Series(labels, index=train_indecies)
            predictions = []
            for i in range(len(test_data)):
                prediction = KNN(train_data, train_labels, test_data.iloc[i], k, distance)
                predictions.append(prediction)
            accuracy_for_c.append(numpy.mean(numpy.array(predictions) == numpy.array(test_labels)))
        accuracy.append(numpy.mean(accuracy_for_c))
    return accuracy


cols = ['area', 'perimeter', 'compactness', 'kernel_length', 'kernel_width', 'asymmetry_coef', 'groove_length', 'type']
data = pandas.read_csv('Data\\Seeds.csv', header=None, names=cols)

k = cross_validation_best_k(data[cols[:-1]], data.type, options=[1, 3, 5, 7, 10])
print(k)

functions = [euclidean_distance, manhattan_distance, lambda p1, p2: minkowski_distance(p1, p2, 4),
             lambda p1, p2: minkowski_distance(p1, p2, 1 / 2), cosine_distance]
print(cross_validation_distances(data[cols[:-1]], data.type, functions, k=5))
