import numpy
import pandas
import seaborn
import sklearn
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from random import randrange, seed

# Plotting Aesthetics
plt.style.use('ggplot')
colormap = LinearSegmentedColormap.from_list('custom', ['#F45B19', '#44A17C', '#A12834', '#ECDF2E', '#3E1443'])
# ===================
seed(0)
X = list([i % 2, i] for i in range(10))
X_normed = normalize(X, axis=0)
model = KMeans(n_clusters=2, random_state=0).fit(X)
model_normed = KMeans(n_clusters=2, random_state=0).fit(X_normed)
plt.figure(figsize=(5,10))
plt.subplot(2, 1, 1)
plt.xlim([-1, 2])
plt.title('Original Data')
for i in range(10):
    plt.scatter(X[i][0], X[i][1], color=['#F45B19', '#44A17C'][model.labels_[i]])
    plt.annotate(i, (X[i][0]+0.05, X[i][1]+0.05))

plt.subplot(2, 1, 2)
plt.title('Normalized Data')
plt.xlim([-1, 2])
for i in range(10):
    plt.scatter(X_normed[i][0], X_normed[i][1], color=['#F45B19', '#44A17C'][model_normed.labels_[i]])
    plt.annotate(i, (X_normed[i][0]+0.05, X_normed[i][1]+0.01))
plt.show()
