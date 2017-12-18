import matplotlib.pyplot as plt
from Logistic_Regression import *





# Reading the data sets
train = numpy.array(pandas.read_csv('Train_Data.csv'), dtype='float64')
test = numpy.array(pandas.read_csv('Test_Data.csv'), dtype='float64')

# Preprocessings
train_data, train_labels, test_data, test_labels, w = preprocessings(train, test)

# Building the Model
w = logistic_regression(train_data, train_labels, w, 0.01, 500)

# Training and Test Accuracy
train_prediction = predict(train_data, w, 0.5)
print("Training Accuracy for:".format(), prediction_accuracy(train_labels, train_prediction))
test_prediction = predict(test_data, w, 0.5)
print("Test Accuracy:", prediction_accuracy(test_labels, test_prediction))

# Plotting the ROC
points = numpy.zeros((2, 101))
for i, threshold in enumerate(numpy.linspace(0, 1, 101)):
    test_prediction = predict(test_data, w, threshold=threshold)
    points[0][i] = false_positive_rate(test_labels, test_prediction)
    points[1][i] = true_positive_rate(test_labels, test_prediction)

plt.plot(points[0], points[1])
plt.show()


# Training Accuracy for: 1.0
# Test Accuracy: 0.84
