import pandas
import svmutil as svm

data = pandas.read_csv('data1.csv')
X = data[['x1', 'x2']]
Y = data['y']

train = list(list(X.iloc[i]) for i in range(X.shape[0]))
labels = list(Y)
options = '-t 0 '  # linear model
options += '-v 5'  # 5-Fold CV

# Training Model
model = svm.svm_train(labels, train, options)
