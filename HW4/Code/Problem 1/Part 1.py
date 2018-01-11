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

plt.figure(figsize=(6, 6))
plt.scatter(numpy.array(X)[Y == 1, 0], numpy.array(X)[Y == 1, 1], cmap=colormap, label='Class 1')
plt.scatter(numpy.array(X)[Y == -1, 0], numpy.array(X)[Y == -1, 1], cmap=colormap, label='Class -1')
plt.xlabel('X₁')
plt.ylabel('X₂')
plt.title('Original Data')
plt.legend()
plt.show()