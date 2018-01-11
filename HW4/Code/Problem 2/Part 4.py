import numpy
import pandas
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import svmutil as svm
import random

# Plotting Aesthetics
plt.style.use('ggplot')
colormap = LinearSegmentedColormap.from_list('custom', ['#F45B19', '#44A17C', '#A12834', '#ECDF2E', '#3E1443'])
# ===================

data = pandas.read_csv('parkinsons.csv')
data_cols = data.columns.values[numpy.multiply(data.columns.values != 'status', data.columns.values != 'name')]
label_col = 'status'
train, rest = train_test_split(data, train_size=0.6, random_state=0)
valid, test = train_test_split(rest, train_size=0.5, random_state=0)
mu = train[data_cols].mean(axis=0)
std = train[data_cols].std(axis=0)
# Scaling Data
train[data_cols] = (train[data_cols] - mu) / std
valid[data_cols] = (valid[data_cols] - mu) / std
test[data_cols] = (test[data_cols] - mu) / std
#

train_data = list(train[data_cols].iloc[i].tolist() for i in range(train.shape[0]))
valid_data = list(valid[data_cols].iloc[i].tolist() for i in range(valid.shape[0]))
test_data = list(test[data_cols].iloc[i].tolist() for i in range(test.shape[0]))
train_labels = train[label_col].tolist()
valid_labels = valid[label_col].tolist()
test_labels = test[label_col].tolist()

param = svm.svm_parameter('-t 1 -q')
param.gamma = 1
param.degree = 4
for C in (10e-20, 10e+20):
    param.C = C
    problem = svm.svm_problem(train_labels, train_data)
    model = svm.svm_train(problem, param)
    print('Test Accuracy:', svm.svm_predict(test_labels, test_data, model)[1][0])
