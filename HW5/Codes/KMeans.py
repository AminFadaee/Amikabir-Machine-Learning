import numpy
from random import sample
from math import sqrt


def euclidean_distance(p1, p2):
    '''
    Computes the euclidean distance between p1 and p2

    Args:
        p1: an array like collection representing an n dimensional point
        p2: an array like collection representing an n dimensional point

    Returns:
        float, the euclidean distance
    '''
    return sqrt(sum((p1 - p2) ** 2))


def kmeans(data, k, distance_function):
    N, D = data.shape
    centroids = data[sample(range(N), k)]
    labels = numpy.zeros(N)
    converged = False
    while not converged:
        converged = True
        sums = numpy.zeros((k, D))
        counts = numpy.zeros(k)
        for i in range(N):
            label = numpy.argmin(list(distance_function(centroids[K], data[i]) for K in range(k)))
            if label != labels[i]:
                converged = False
                labels[i] = label
            sums[label] += data[i]
            counts[label] += 1
        centroids = (sums.T / counts).T
    return labels, centroids
