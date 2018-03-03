import numpy
import pandas
import seaborn
import sklearn
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from math import sqrt
from random import sample, seed
from KMeans import kmeans, euclidean_distance
from DBSCAN import dbscan

# Plotting Aesthetics
plt.style.use('ggplot')
colormap = LinearSegmentedColormap.from_list('custom', ['#F45B19', '#44A17C', '#A12834', '#ECDF2E', '#3E1443'])


# ===================

def db(data, labels, k, centroids, distance_function):
    D = numpy.zeros(k)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            d = distance_function(centroids[i], centroids[j])
            mui = numpy.mean(
                list(distance_function(centroids[i], data[labels == i][a]) for a in range(data[labels == i].shape[0])))
            muj = numpy.mean(
                list(distance_function(centroids[j], data[labels == j][a]) for a in range(data[labels == j].shape[0])))
            if D[i] < (mui + muj) / d:
                D[i] = (mui + muj) / d
    return numpy.sum(D) / k


file = pandas.read_excel('data1.xlsx', header=None, names=['x', 'y1', 'y2'])
data = numpy.array(list(zip(list(file.x) * 2, list(file.y1) + list(file.y2))))
colors = ['#F45B19', '#44A17C', '#A12834', '#ECDF2E', '#3E1443']
# =================Problem A=================
seed(0)
K = (2, 3, 4, 5)
DB = []
for k in K:
    labels, centroids = kmeans(data, k, euclidean_distance)
    DB.append(db(data, labels, k, centroids, euclidean_distance))
print(list(zip(K, DB)))
k = K[numpy.argmin(DB)]
labels, centroids = kmeans(data, k, euclidean_distance)
for l in range(k):
    index = labels == l
    label = 'Cluster {0}'.format(l+1)
    plt.scatter(data[:, 0][index], data[:, 1][index], color=colors[l], label=label)
plt.legend()
plt.title('KMeans Clustering, K={0}'.format(k))
plt.show()
# =================Problem B=================
labels = dbscan(data, 0.3, 3)
for l in numpy.unique(labels):
    index = labels == l
    label = 'Cluster {0}'.format(l) if l else 'Outlier'
    plt.scatter(data[:, 0][index], data[:, 1][index], color=colors[l], label=label)
plt.legend()
plt.title('DBSCAN Clustering')
plt.show()
