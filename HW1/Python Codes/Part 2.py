import numpy
import pandas
import matplotlib.pyplot as plt
from ProgrammingAssingment import read_split_data,closed_form,preprocessing,MSE,predict

train_data, test_data = read_split_data('data.xlsx')
plot_data = pandas.DataFrame(numpy.linspace(0, 16, 500), columns=['x'])

training_errors = []
testing_errors = []
for order in [3, 5, 7]:
    train, test, mu, std, scaled_plot_data, columns = preprocessing(train_data, test_data, plot_data, order=order)
    W = closed_form(train[columns], train.y, r=0)
    train_error = MSE(train.y, train[columns], W)
    test_error = MSE(test.y, test[columns], W)

    training_errors.append(train_error)
    testing_errors.append(test_error)
    plot_data_labels = predict(scaled_plot_data, W)
    print('=========================')
    print('Order:          {0}'.format(order))
    print('Training Error: {0}'.format(round(train_error, 4)))
    print('Test Error:     {0}'.format(round(test_error, 4)))
    print('=========================')

    plt.subplot(1, 2, 1)
    plt.plot(plot_data, plot_data_labels, label='Order:{0}'.format(order))

plt.subplot(1, 2, 1)
plt.title('Fits')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(train_data.x, train_data.y, label='Training Data', color='#268CA5')
plt.scatter(test_data.x, test_data.y, label='Test Data', color='#F45E49')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('Errors')
plt.xlabel('Order')
plt.ylabel('MSE')
plt.plot([3, 5, 7], training_errors, 'o-', label='Training Error', color='#268CA5')
plt.plot([3, 5, 7], testing_errors, 'o-', label='Test Error', color='#F45E49')
plt.legend()
plt.show()