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

results = []
for d in range(1, 11):
    for g in range(-5, 5):
        param = svm.svm_parameter('-t 1 -q')
        param.gamma = 10 ** g
        param.degree = d
        problem = svm.svm_problem(train_labels, train_data)
        model = svm.svm_train(problem, param)
        p_acc = svm.svm_predict(valid_labels, valid_data, model)[1]
        results.append([param.degree, param.gamma, p_acc[0]])

degree, gamma, acc = max(results, key=lambda i: i[2])
print('Best Validation Result: degree= {0}, gamma={1}, Accuracy={2}'.format(degree, gamma, acc))
param = svm.svm_parameter('-t 1 -q')
param.gamma = gamma
param.degree = d
problem = svm.svm_problem(train_labels, train_data)
model = svm.svm_train(problem, param)
print('Test Accuracy:', svm.svm_predict(test_labels, test_data, model)[1][0])
print('Number of Support Vectors:',len(model.get_sv_indices()))


# Part 2
random.seed(0)
random_parameters = list([random.choice(range(-10, 11)), random.choice(range(-10, 11)), random.choice(range(1, 11)),
                          random.choice(range(-10, 11))] for i in range(20))
results = []
for c, g, d, coef in random_parameters:
    param = svm.svm_parameter('-t 1 -q')
    param.C = 10 ** c
    param.gamma = 10 ** g
    param.degree = d
    param.coef0 = 10 ** coef
    problem = svm.svm_problem(train_labels, train_data)
    model = svm.svm_train(problem, param)
    p_acc = svm.svm_predict(valid_labels, valid_data, model)[1]
    results.append([param.C, param.gamma, param.degree, param.coef0, p_acc[0]])

C, gamma, degree, coef, acc = max(results, key=lambda i: i[-1])
print(
    'Best Validation Result: C={0}, gamma={1}, degree={2}, coef0={3}, Accuracy={4}'.format(C, gamma, degree, coef, acc))
param = svm.svm_parameter('-t 1 -q')
param.C = C
param.gamma = gamma
param.degree = degree
param.coef0 = coef
problem = svm.svm_problem(train_labels, train_data)
model = svm.svm_train(problem, param)
print('Test Accuracy:', svm.svm_predict(test_labels, test_data, model)[1][0])
print('Number of Support Vectors:',len(model.get_sv_indices()))
