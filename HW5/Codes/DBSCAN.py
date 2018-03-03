import numpy
import pandas
import seaborn
import sklearn
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def distance(p1, p2):
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


def epsilon_neighborhoood(data, epsilon, p):
    '''
    Compute the epsilon neighborhood of point p

    Args:
        data: numpy matrix of |NxD| consisting of N points (D dimension each)
        epsilon: radius
        p: index of a point in data

    Returns:
        N, list of all the points in epsilon neighborhood
    '''
    N = []
    r, c = data.shape
    for i in range(r):
        if i != p and distance(data[i], data[p]) <= epsilon:
            N.append(i)
    return N


def dbscan(data, epsilon, minpoints):
    '''
    Implementation of DBSCAN algorithm

    Args:
        data: numpy matrix of |NxD| consisting of N points (D dimension each)
        epsilon: radius 
        minpoints: threshold of points in neighborhood

    Returns:
        list of labels denoting the cluster of each point
    '''
    print("___Performing DBSCAN")
    C = 0  # cluster number
    rows, cols = data.shape
    labels = numpy.array([-1] * rows)  # initialize every point as not visited

    for p in range(rows):
        if labels[p] == -1:
            N = epsilon_neighborhoood(data, epsilon, p)
            if len(N) < minpoints - 1:  # We should count p as well
                labels[p] = 0  # set to noise
            else:
                C = C + 1
                labels[p] = C
                for q in N:
                    if labels[q] > 0:  # if q belongs to a cluster
                        continue
                    labels[q] = C
                    M = epsilon_neighborhoood(data, epsilon, q)
                    if len(M) >= minpoints:
                        N += M
    return labels
