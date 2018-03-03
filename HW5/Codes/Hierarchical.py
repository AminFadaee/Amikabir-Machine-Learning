import numpy
import pandas
import seaborn
import sklearn
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import heapq
from KMeans import kmeans
from math import sqrt

from scipy.spatial.kdtree import distance_matrix


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


def two_means(data):
    labels, _ = kmeans(data, 2, euclidean_distance)
    while len(numpy.unique(labels)) < 2:
        labels, _ = kmeans(data, 2, euclidean_distance)
    return labels + 1  # forcing to have 1/2 labels


def average_distance(data):
    mu = numpy.mean(data, axis=0)
    return numpy.sum((data - mu) ** 2)


def bottom_up(data, link, distance_function, show_stages=False, index=False):
    # This function is not efficient! AT ALL!!#
    def linkage(distance_Matrix, labels, C):
        # finding best indices
        I, J = numpy.unravel_index(distance_Matrix.argmin(), distance_Matrix.shape)
        # creating the new matrix:
        if link == 'single':
            D = numpy.minimum(distance_Matrix[I], distance_Matrix[J])
        elif link == 'complete':
            D = numpy.maximum(distance_Matrix[I], distance_Matrix[J])
        elif link == 'average':
            m = numpy.sum(labels == (I + 1))
            n = numpy.sum(labels == (J + 1))
            D = numpy.add(distance_Matrix[I] * m, distance_Matrix[J] * n) / (m + n)
        D[I] = float('Inf')  # distance to itself is inf
        distance_Matrix[I] = D
        distance_Matrix[:, I] = D
        new_distances = numpy.zeros((C - 1, C - 1))
        new_distances[:J, :J] = distance_Matrix[:J, :J]
        new_distances[:J, J:] = distance_Matrix[:J, J + 1:]
        new_distances[J:, :J] = distance_Matrix[J + 1:, :J]
        new_distances[J:, J:] = distance_Matrix[J + 1:, J + 1:]
        # updating labels:
        labels[labels == J + 1] = I + 1
        for j in range(J + 2, C + 1):
            labels[labels == j] -= 1
        return new_distances, labels

    N, D = data.shape
    labels = numpy.array(list(range(1, N + 1)))
    C = N
    distances = numpy.zeros((N, N))
    for i in range(N):
        distances[i, i] = float('Inf')
        for j in range(i + 1, N):
            d = distance_function(data[i], data[j])
            distances[i, j] = d
            distances[j, i] = d
    ch_indices = []
    while C > 1:
        print('Progress = {0}%, C = {1}'.format(round(100 * (N - C) / (N - 1), 2), C))
        distances, labels = linkage(distances, labels, C)
        C -= 1

        if index:
            if index == True:
                ch_indices.append((C, CH_index(data, labels, C)))
            elif type(index) == type((0,)) and len(index) == 2:
                if index[0] <= C <= index[1]:
                    ch_indices.append((C, CH_index(data, labels, C)))
        if show_stages:
            if show_stages == True:
                title = '{0}-linkage with {1} Cluster{2}'.format(link.capitalize(), C, '' if C == 1 else 's')
                plot(data, labels, C, title=title, legend=False)
            elif type(show_stages) == type((0,)) and len(show_stages) == 2:
                if show_stages[0] <= C <= show_stages[1]:
                    title = '{0}-linkage with {1} Cluster{2}'.format(link.capitalize(), C, '' if C == 1 else 's')
                    plot(data, labels, C, title=title, legend=False)

    if index:
        return labels, ch_indices
    else:
        return labels


def CH_index(data, labels, C):
    N, D = data.shape
    W, B = 0, 0
    X_bar = numpy.mean(data, axis=0)
    for c in range(1, C + 1):
        Xc = data[labels == c]
        Nc = Xc.shape[0]
        Xc_bar = numpy.mean(Xc, axis=0)
        W += numpy.sum((Xc - Xc_bar) ** 2)
        B += Nc * numpy.sum((Xc_bar - X_bar) ** 2)
    CH = (B / (C - 1)) / (W / (N - C))
    return CH


def top_down(data, clustering_function, measure_function, max_clusters=-1, show_stages=False, index=False):
    N, D = data.shape
    labels = numpy.ones(N)
    cluster_info = [(-measure_function(data), 1)]
    C = 1
    ch_indices = []
    while (C == -1 or C < max_clusters) and len(cluster_info) != 0:
        _, i = heapq.heappop(cluster_info)
        i_index = labels == i
        labels[i_index] = clustering_function(data[i_index])
        C += 1
        one_index = i_index * (labels == 1)
        two_index = i_index * (labels == 2)
        labels[one_index] = i  # reshaping 1
        labels[two_index] = C  # reshaping 2

        if len(data[labels == i]) > 1:
            heapq.heappush(cluster_info, (-measure_function(data[labels == i]), i))
        if len(data[labels == C]) > 1:
            heapq.heappush(cluster_info, (-measure_function(data[labels == C]), C))
        if index:
            if index == True:
                ch_indices.append((C, CH_index(data, labels, C)))
            elif type(index) == type((0,)) and len(index) == 2:
                if index[0] <= C <= index[1]:
                    ch_indices.append((C, CH_index(data, labels, C)))
        if show_stages:
            title = 'Top-Down {0} Cluster{1}'.format(C, '' if C == 1 else 's')
            if show_stages == True:
                plot(data, labels, C, title=title, legend=False)
            elif type(show_stages) == type((0,)) and len(show_stages) == 2:
                if show_stages[0] <= C <= show_stages[1]:
                    plot(data, labels, C, title=title, legend=False)
    if index:
        return labels, ch_indices
    else:
        return labels


def plot(data, labels, C, title, legend=True):
    plt.figure()
    colors = ['#F45B19', '#44A17C', '#A12834', '#ECDF2E', '#3E1443',
              '#251E19', '#CE4134', '#4AAD13', '#43D1AE', '#FFB311']
    for l in range(1, C + 1):
        index = labels == l
        label = 'Cluster {0}'.format(l)
        plt.scatter(data[:, 0][index], data[:, 1][index], color=colors[(l - 1) % 10], label=label)
    if legend:
        plt.legend()

    plt.title(title)
    plt.show()
