import matplotlib.pyplot as plt
from random import shuffle
from Logistic_Regression import *


def k_fold_cross_validation_indices(k, N):
    '''
    Creates the indices use for k fold cross validation
    Args:
        k: number of folds
        N: size of data set

    Returns:
        indices
    '''
    index = list(range(N))
    shuffle(index)
    train_indices = []
    test_indices = []
    for i in range(k):
        test = index[int(i * N / k):int((i + 1) * N / k)]
        train = index[:int(i * N / k)] + index[int((i + 1) * N / k):]
        train_indices.append(train)
        test_indices.append(test)
    return train_indices, test_indices


# Reading the data sets
train = numpy.array(pandas.read_csv('Train_Data.csv'), dtype='float64')
test = numpy.array(pandas.read_csv('Test_Data.csv'), dtype='float64')

k = 10
train_indices, test_indices = k_fold_cross_validation_indices(k, N=len(train))
penalties = [10**i for i in range(-5,3)]
accuracies = []
for penalty in penalties:
    print('*'*50)
    print('Penalty:',penalty)
    folds_accuracies = []
    for i in range(k):
        valid_train = train[train_indices[i]]
        valid_test = train[test_indices[i]]

        # Preprocessings
        train_data, train_labels, test_data, test_labels, w = preprocessings(valid_train, valid_test)

        # Building the Model
        w = logistic_regression(train_data, train_labels, w, step_size=0.01, max_iter=100, l2_penalty=penalty)

        # This Fold's Test Accuracy
        test_prediction = predict(test_data, w, 0.5)
        accuracy = prediction_accuracy(test_labels, test_prediction)
        folds_accuracies.append(accuracy)

    accuracies.append(numpy.mean(folds_accuracies))

    print('Accuracy:',accuracies[-1])
    print('*'*50)

best_penalty = penalties[numpy.argmax(accuracies)]

# Now that we have the best penalty, we train the system using that penalty.

# Preprocessings
train_data, train_labels, test_data, test_labels, w = preprocessings(train, test)

# Building the Model
w = logistic_regression(train_data, train_labels, w, 0.01, 100, best_penalty)

# Training and Test Accuracy
train_prediction = predict(train_data, w, 0.5)
test_prediction = predict(test_data, w, 0.5)

# Plotting the ROC
points = numpy.zeros((2, 101))
for i, threshold in enumerate(numpy.linspace(0, 1, 101)):
    test_prediction = predict(test_data, w, threshold=threshold)
    points[0][i] = false_positive_rate(test_labels, test_prediction)
    points[1][i] = true_positive_rate(test_labels, test_prediction)

plt.plot(points[0], points[1])
print("Best 𝜆:", best_penalty)
print("Training Accuracy for:".format(), prediction_accuracy(train_labels, train_prediction))
print("Test Accuracy:", prediction_accuracy(test_labels, test_prediction))
plt.show()


# **************************************************
# Penalty: 1e-05
# Accuracy: 0.83875
# **************************************************
# **************************************************
# Penalty: 0.0001
# Accuracy: 0.83875
# **************************************************
# **************************************************
# Penalty: 0.001
# Accuracy: 0.83875
# **************************************************
# **************************************************
# Penalty: 0.01
# Accuracy: 0.839375
# **************************************************
# **************************************************
# Penalty: 0.1
# Accuracy: 0.838125
# **************************************************
# **************************************************
# Penalty: 1
# Accuracy: 0.8225
# **************************************************
# **************************************************
# Penalty: 10
# Accuracy: 0.69125
# **************************************************
# **************************************************
# Penalty: 100
# Accuracy: 0.439375
# **************************************************
# Best 𝜆: 0.01
# Training Accuracy for: 1.0
# Test Accuracy: 0.51

