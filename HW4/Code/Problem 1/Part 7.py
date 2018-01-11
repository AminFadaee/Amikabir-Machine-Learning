import numpy
import pandas
import svmutil as svm


data = pandas.read_csv('data1.csv')
X = data[['x1', 'x2']]
Y = data['y']

k1 = lambda x, y: x ** 2 + y ** 2
X1 = X.apply(lambda row: k1(row.x1, row.x2), axis=1)

k2 = lambda x, y: (x + y, x ** 2 + y ** 2)
X2 = X.apply(lambda row: numpy.array(k2(row.x1, row.x2)), axis=1)

k3 = lambda x, y: (x, y, x ** 2 + y ** 2)
X3 = X
X3['X3'] = X1

train1 = list([X1.iloc[i]] for i in range(X.shape[0]))
train2 = list(list(X2.iloc[i]) for i in range(X.shape[0]))
train3 = list(list(X3.iloc[i]) for i in range(X.shape[0]))
labels = list(Y)

options = '-t 0 '  # linear model
options += '-v 5'  # 5-Fold CV
for train in (train1, train2, train3):
    print('=' * 150)
    # Training Model
    model = svm.svm_train(labels, train, options)
