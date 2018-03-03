import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas
import numpy
from random import seed
from Hierarchical import top_down, two_means, average_distance, bottom_up, euclidean_distance

# Plotting Aesthetics
plt.style.use('ggplot')
colormap = LinearSegmentedColormap.from_list('custom', ['#F45B19', '#44A17C', '#A12834', '#ECDF2E', '#3E1443'])
# ===================
seed(0)
data = numpy.array(pandas.read_csv('data2.csv', header=None))
# data = numpy.array([[-3, 0],
#                     [-4, 0],
#                     [-2, 0],
#                     [-3, 1],
#                     [-3, -1],
#                     [3, 0],
#                     [4, 0],
#                     [2, 0],
#                     [3, 1],
#                     [3, -1]])
stages = (2, 10)
index = (2, 10)
labels1, indecies1 = top_down(data, two_means, average_distance, max_clusters=10, show_stages=stages, index=index)
labels2, indecies2 = bottom_up(data, 'single', euclidean_distance, show_stages=stages, index=index)
labels3, indecies3 = bottom_up(data, 'complete', euclidean_distance, show_stages=stages, index=index)
labels4, indecies4 = bottom_up(data, 'average', euclidean_distance, show_stages=stages, index=index)


for i, ind in enumerate((indecies1, indecies2, indecies3, indecies4)):
    Xind = list(ind[x][0] for x in range(len(ind)))
    Yind = list(ind[y][1] for y in range(len(ind)))
    color = ['#F45B19', '#44A17C', '#A12834', '#3E1443'][i]
    label = ['top down', 'single-linkage', 'complete-linkage', 'average-linkage'][i].title()
    print('Best C for {0} = {1}'.format(label,Xind[numpy.argmax(Yind)]))
    plt.plot(Xind, Yind, label=label, color=color,marker='o')
plt.title('ch-indices'.title())
plt.xlabel('Number of Clusters')
plt.ylabel('CH')
plt.legend()
plt.savefig(r'D:\Tasks\University\Machine Learning\Homeworks\HW5\HTML\Images\\' + 'CH.svg',bbox_inches='tight', format='svg', dpi=1200)
plt.show()
