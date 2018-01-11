import numpy
import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Plotting Aesthetics
plt.style.use('ggplot')
colormap = LinearSegmentedColormap.from_list('custom', ['#F45B19', '#44A17C', '#A12834', '#ECDF2E', '#3E1443'])
# ===================

data = pandas.read_csv('data1.csv')
X = data[['x1', 'x2']]
Y = data['y']

k1 = lambda x, y: x ** 2 + y ** 2
X1 = X.apply(lambda row: k1(row.x1, row.x2), axis=1)
plt.scatter(numpy.array(X1)[Y == 1], [0] * X1[Y == 1].shape[0], cmap=colormap, label='Class 1', alpha=0.2)
plt.scatter(numpy.array(X1)[Y == -1], [0] * X1[Y == -1].shape[0], cmap=colormap, label='Class -1', alpha=0.2)
plt.yticks([])
plt.title('1D Mapped Data')
plt.xlabel('X₁² + X₂²')
plt.legend()
plt.show()