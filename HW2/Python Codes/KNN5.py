import numpy
import pandas
import random


def distance(p1, p2):
    '''
    Computes the euclidean distance between p1 and p2

    Args:
        p1: an array like collection representing an n dimensional point
        p2: an array like collection representing an n dimensional point

    Returns:
        float, the euclidean distance
    '''
    p1 = numpy.array(p1)
    p2 = numpy.array(p2)
    return numpy.sqrt(numpy.sum((p1 - p2) ** 2))


def find_KNN(data, labels, test, k, test_in_data=False):
    '''
    Computes the k nearest neighbor of test.

    Args:
        data: DataFrame of shape (N,D) of independent variables
        labels: DataSeries of size N of the dependent variable
        test: data for prediction of class
        k: number of nearest neighbors
        test_in_data: True if 'test' is an instance in 'data'

    Returns:
        pandas DataFrame containing k rows of nearest neighbors with fields 'distance','index','label'
    '''
    distances = []
    for i in range(data.shape[0]):
        row = data.iloc[i]
        distances.append((distance(row, test), i, labels.iloc[i]))
    N = pandas.DataFrame(sorted(distances, key=lambda i: i[0])[test_in_data:k + test_in_data],
                         columns=['distance', 'index', 'label'])
    return N


def WDKNN(data, labels, test, k, test_in_data=False):
    '''
    Computes the WDKNN classification on test with the provided distance function

    Args:
        data: DataFrame of shape (N,D) of independent variables
        labels: DataSeries of size N of the dependent variable
        test: data for prediction of class
        k: number of nearest neighbors
        test_in_data: True if 'test' is an instance in 'data'

    Returns:
        predicted label of test
    '''
    N = find_KNN(data, labels, test, k, test_in_data)
    coeffs = 1 / N.distance
    strength = dict()
    for i in range(k):
        if N.label[i] not in strength:
            strength[N.label[i]] = coeffs[i]
        else:
            strength[N.label[i]] += coeffs[i]
    return max(strength, key=lambda i: strength[i])


def MCKNN(data, labels, test, k, minority_label, test_in_data=False):
    '''
    Computes the Minority Class KNN classification on test with the provided distance function

    Args:
        data: DataFrame of shape (N,D) of independent variables
        labels: DataSeries of size N of the dependent variable
        test: data for prediction of class
        k: number of nearest neighbors
        minority_label: the label of the minority class
        test_in_data: True if 'test' is an instance in 'data'

    Returns:
        predicted label of test
    '''
    N = find_KNN(data, labels, test, k, test_in_data)

    coeffs = []
    for i in range(k):
        dist, index, label = N.iloc[i]
        if label != minority_label:
            coeffs.append(1 / dist)
        else:
            M = find_KNN(data, labels, data.iloc[int(index)], k, True)
            majority_number = 0
            for i in range(k):
                if M.iloc[i].label != minority_label:
                    majority_number += 1
            Lm = majority_number / k
            W = Lm + 1
            coeffs.append(W / dist)

    strength = dict()
    for i in range(k):
        if N.label[i] not in strength:
            strength[N.label[i]] = coeffs[i]
        else:
            strength[N.label[i]] += coeffs[i]
    return max(strength, key=lambda i: strength[i])


def precision(TP, FP):
    return TP / (TP + FP)


def recall(TP, FN):
    return TP / (TP + FN)


def g_mean(TP, TN, FP, FN):
    return numpy.sqrt(recall(TP, FN) * TN / (TN + FP))


def f_measure(TP, TN, FP, FN):
    return 2 * recall(TP, FN) * precision(TP, FP) / (recall(TP, FN) + precision(TP, FP))


def five_fold(data, labels, k, function):
    N, D = data.shape
    random.seed(0)
    index = list(range(N))
    random.shuffle(index)

    TP = [0, 0, 0, 0, 0]
    TN = [0, 0, 0, 0, 0]
    FP = [0, 0, 0, 0, 0]
    FN = [0, 0, 0, 0, 0]
    for c in range(5):
        test_indices = index[c * N // 5:(c + 1) * N // 5]
        train_indices = index[:c * N // 5] + index[(c + 1) * N // 5:]
        test_data = pandas.DataFrame(data, index=test_indices)
        train_data = pandas.DataFrame(data, index=train_indices)
        test_labels = pandas.Series(labels, index=test_indices)
        train_labels = pandas.Series(labels, index=train_indices)
        for i in range(len(test_data)):
            if function == 'MCKNN':
                prediction = MCKNN(train_data, train_labels, test_data.iloc[i], k, minority_label=1)
            elif function == 'WDKNN':
                prediction = WDKNN(train_data, train_labels, test_data.iloc[i], k)
            else:
                return
            true_label = test_labels.iloc[i]
            TN[c] += (prediction == true_label == 0)
            TP[c] += (prediction == true_label == 1)
            FP[c] += (prediction == 1 and true_label == 0)
            FN[c] += (prediction == 0 and true_label == 1)
    TP = numpy.mean(TP)
    TN = numpy.mean(TN)
    FP = numpy.mean(FP)
    FN = numpy.mean(FN)
    return {'G-Mean': g_mean(TP, TN, FP, FN), 'F-Measure': f_measure(TP, TN, FP, FN)}


data_names = ['yeast3',
              'ecoli3',
              'yeast-2_vs_4',
              'yeast-0-3-5-9_vs_7-8',
              'yeast-0-2-5-6_vs_3-7-8-9',
              'yeast-0-2-5-7-9_vs_3-6-8',
              'ecoli-0-2-6-7_vs_3-5']

for name in data_names:
    data = pandas.read_csv('Data\\{0}.csv'.format(name))
    print('*' * 50)
    print('Results for {0} data'.format(name))
    print('WDKNN:', five_fold(data[data.columns[:-1]], data.Class, k=5, function='WDKNN'))
    print('MCKNN:', five_fold(data[data.columns[:-1]], data.Class, k=5, function='MCKNN'))