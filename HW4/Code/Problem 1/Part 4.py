import numpy
import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

# Plotting Aesthetics
plt.style.use('ggplot')
colormap = LinearSegmentedColormap.from_list('custom', ['#F45B19', '#44A17C', '#A12834', '#ECDF2E', '#3E1443'])
# ===================

data = pandas.read_csv('data1.csv')
X = data[['x1', 'x2']]
Y = data['y']

k1 = lambda x, y: x ** 2 + y ** 2
X1 = X.apply(lambda row: k1(row.x1, row.x2), axis=1)

k3 = lambda x, y: (x, y, x ** 2 + y ** 2)
X3 = X
X3['X3'] = X1
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(numpy.array(X3)[Y == 1, 0], numpy.array(X3)[Y == 1, 1], numpy.array(X3)[Y == 1, 2], cmap=colormap,
           label='Class 1')
ax.scatter(numpy.array(X3)[Y == -1, 0], numpy.array(X3)[Y == -1, 1], numpy.array(X3)[Y == -1, 1], cmap=colormap,
           label='Class -1')
ax.set_xlabel('X₁')
ax.set_ylabel('X₂')
ax.set_zlabel('X₁² + X₂²')
plt.title('3D Mapped Data')
plt.legend()
plt.show()