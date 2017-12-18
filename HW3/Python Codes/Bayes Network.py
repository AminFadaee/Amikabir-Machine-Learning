import pandas
from random import shuffle
from random import seed
from pgmpy.models import BayesianModel

seed(0)


# columns = ['buying', 'maint', 'doors', 'persons', 'safety', 'lug_boot', 'label']
# states = {'buying': ['vhigh', 'high', 'med', 'low'],
#           'maint': ['vhigh', 'high', 'med', 'low'],
#           'doors': ['2', '3', '4', '5more'],
#           'persons': ['2', '4', 'more'],
#           'safety': ['low', 'med', 'high'],
#           'lug_boot': ['small', 'med', 'big'],
#           'label': ['unacc', 'acc', 'good', 'vgood']}


def bayes_network(train, test, columns, label, edges):
    model = BayesianModel(edges)
    model.fit(train)
    prediction = model.predict(test[columns])
    accuracy = (prediction == test[[label]]).sum() / len(test)
    return accuracy.label


def ten_fold(data, columns, label, edges):
    data = data[columns + [label]]
    N = len(data)
    indices = list(range(N))
    shuffle(indices)
    accuracies = []
    for k in range(10):
        train_indices = indices[:int(k * N / 10)] + indices[int((k + 1) * N / 10):]
        test_indices = indices[int(k * N / 10):int((k + 1) * N / 10)]
        accuracies.append(bayes_network(data.iloc[train_indices], data.iloc[test_indices], columns, label, edges))
        print(accuracies[-1])
    return sum(accuracies) / 10


data = pandas.read_csv('Cars.csv')
# ================First Model=================
columns = ['buying', 'maint', 'doors', 'persons', 'safety', 'lug_boot']
label = 'label'
edges = [['buying', 'label'],['maint', 'label'], ['doors', 'label'], ['persons', 'label'], ['safety', 'label'], ['lug_boot', 'label']]
print('Accuracy:', ten_fold(data, columns, label, edges))
# ================Second Model================
# columns = ['buying', 'persons', 'safety']
# label = 'label'
# edges = [['buying', 'label'], ['persons', 'label'], ['safety', 'label']]
# print('Accuracy:', ten_fold(data, columns, label, edges))
# Accuracy: 0.812518483667
# ================Third Model=================
# columns = ['maint', 'doors', 'safety', 'lug_boot']
# label = 'label'
# edges = [['maint', 'label'], ['doors', 'label'], ['safety', 'label'], ['lug_boot', 'label']]
# print('Accuracy:', ten_fold(data, columns, label, edges))
# Accuracy: 0.630205672805
# ================Fourth Model================
# columns = ['buying', 'maint', 'doors', 'persons']
# label = 'label'
# edges = [['buying', 'maint'], ['maint', 'label'], ['buying', 'label'], ['doors', 'persons'], ['persons', 'label'],
#          ['doors', 'label']]
# print('Accuracy:', ten_fold(data, columns, label, edges))
# Accuracy: 0.632554778868
# ================Fifth Model=================
# columns = ['buying', 'maint', 'doors', 'persons', 'safety']
# label = 'label'
# edges = [['buying', 'maint'], ['maint', 'label'], ['buying', 'label'], ['doors', 'persons'], ['persons', 'label'],
#          ['doors', 'label'], ['safety', 'label']]
# print('Accuracy:', ten_fold(data, columns, label, edges))
# Accuracy: 0.828703454765

# columns = ['buying', 'maint', 'doors', 'persons', 'safety', 'lug_boot']
# label = 'label'
# edges = [['buying', 'maint'], ['maint', 'label'], ['buying', 'label'], ['doors', 'persons'], ['persons', 'label'],
#          ['doors', 'label'], ['safety', 'label'], ['lug_boot', 'label']]
# print('Accuracy:', ten_fold(data, columns, label, edges))
# Accuracy: 0.222217367926
