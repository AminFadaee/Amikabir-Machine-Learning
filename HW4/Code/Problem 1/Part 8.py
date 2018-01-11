import pandas
import svmutil as svm


data = pandas.read_csv('data1.csv')
X = data[['x1', 'x2']]
Y = data['y']

labels = list(Y)

prek1 = pandas.read_csv('prek1.csv', header=None)
prek2 = pandas.read_csv('prek2.csv', header=None)
prek3 = pandas.read_csv('prek3.csv', header=None)
for prek in (prek1, prek2, prek3):
    print('=' * 150)
    train = list([i + 1] + list(prek.iloc[i]) for i in range(prek.shape[0]))
    prob = svm.svm_problem(labels, train, isKernel=True)
    param = svm.svm_parameter('-t 4 -v 5')
    model = svm.svm_train(prob, param)
