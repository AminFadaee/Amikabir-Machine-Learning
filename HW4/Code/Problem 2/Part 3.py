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
for coef in range(-5, 5):
    for g in range(-5, 5):
        for c in range(-5,5):
            param = svm.svm_parameter('-t 3 -q')
            param.gamma = 10 ** g
            param.coef0 = 10 ** coef
            param.C = 10 ** c
            problem = svm.svm_problem(train_labels, train_data)
            model = svm.svm_train(problem, param)
            p_acc = svm.svm_predict(valid_labels, valid_data, model)[1]
            results.append([param.C, param.gamma,param.coef0, p_acc[0]])

C, gamma, coef0 ,acc= max(results, key=lambda i: i[-1])
print('Best Validation Result: C={0}, gamma={1}, coef0={2}, Accuracy={3}'.format(C, gamma, coef0, acc))
param = svm.svm_parameter('-t 3 -q')
param.gamma = gamma
param.C = C
param.coef0 = coef0
problem = svm.svm_problem(train_labels, train_data)
model = svm.svm_train(problem, param)
print('Test Accuracy:', svm.svm_predict(test_labels, test_data, model)[1][0])
print('Number of Support Vectors:',len(model.get_sv_indices()))


# Part 2
random.seed(0)
random_parameters = list([random.choice(range(-10, 11)), random.choice(range(1, 11)),
                          random.choice(range(-10, 11))] for i in range(20))
results = []
for c, g, coef in random_parameters:
    param = svm.svm_parameter('-t 3 -q')
    param.C = 10 ** c
    param.gamma = 10 ** g
    param.coef0 = 10 ** coef
    problem = svm.svm_problem(train_labels, train_data)
    model = svm.svm_train(problem, param)
    p_acc = svm.svm_predict(valid_labels, valid_data, model)[1]
    results.append([param.C, param.gamma, param.coef0, p_acc[0]])

C, gamma, coef, acc = max(results, key=lambda i: i[-1])
print(
    'Best Validation Result: C={0}, gamma={1}, coef0={2}, Accuracy={3}'.format(C, gamma, coef, acc))
param = svm.svm_parameter('-t 3 -q')
param.C = C
param.gamma = gamma
param.coef0 = coef
problem = svm.svm_problem(train_labels, train_data)
model = svm.svm_train(problem, param)
print('Test Accuracy:', svm.svm_predict(test_labels, test_data, model)[1][0])
print('Number of Support Vectors:',len(model.get_sv_indices()))
