import pandas
import numpy
import matplotlib.pyplot as plt
from numpy import log10
from ProgrammingAssingment import read_split_data,preprocessing,graident_descent,MSE,predict

train_data, test_data = read_split_data('data.xlsx')
plot_data = pandas.DataFrame(numpy.linspace(0, 16, 500), columns=['x'])

for order in [7]:
    train, test, mu, std, scaled_plot_data, columns = preprocessing(train_data, test_data, plot_data, order=order)
    iterations = list(10 ** i for i in range(2, 6))
    stepsizes = {3: 0.5, 5: 0.4, 7: 1.998}
    training_errors = []
    testing_errors = []
    for iteration in iterations:
        W = [0] * (order + 1)

        W= graident_descent(train[columns],W, train.y,stepsizes[order],iteration )
        train_error = MSE(train.y, train[columns], W)
        test_error = MSE(test.y, test[columns], W)

        training_errors.append(train_error)
        testing_errors.append(test_error)
        plot_data_labels = predict(scaled_plot_data, W)
        print('=========================')
        print('Order:          {0}'.format(order))
        print('Alpha:          {0}'.format(stepsizes[order]))
        print('Iterations:     {0}'.format(iteration))
        print('Training Error: {0}'.format(round(train_error, 4)))
        print('Test Error:     {0}'.format(round(test_error, 4)))
        print('=========================')

        plt.subplot(1, 2, 1)
        plt.plot(plot_data, plot_data_labels, label='Order:{0},Iters:{1}'.format(order, iteration))

    plt.subplot(1, 2, 1)
    plt.title('Order {0} Polynomial, Step Size: {1}'.format(order, stepsizes[order]))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(train_data.x, train_data.y, label='Training Data', color='#268CA5')
    plt.scatter(test_data.x, test_data.y, label='Test Data', color='#F45E49')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Errors')
    plt.xlabel('Iterations (Log 10)')
    plt.ylabel('MSE')
    plt.plot(list(map(log10, iterations)), training_errors, 'o-', label='Training Error', color='#268CA5')
    plt.plot(list(map(log10, iterations)), testing_errors, 'o-', label='Test Error', color='#F45E49')
    plt.legend()
    plt.show()