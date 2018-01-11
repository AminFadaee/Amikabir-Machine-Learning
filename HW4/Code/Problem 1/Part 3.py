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

k2 = lambda x, y: (x + y, x ** 2 + y ** 2)
X2 = X.apply(lambda row: numpy.array(k2(row.x1, row.x2)), axis=1)
plt.figure(figsize=(6, 6))
plt.scatter(numpy.array(X2)[Y == 1, 1], numpy.array(X2)[Y == 1, 0], cmap=colormap, label='Class 1')
plt.scatter(numpy.array(X2)[Y == -1, 1], numpy.array(X2)[Y == -1, 0], cmap=colormap, label='Class -1')
plt.xlabel('X₁² + X₂²')
plt.ylabel('X₁ + X₂')
plt.title('2D Mapped Data')
plt.legend()
plt.show()