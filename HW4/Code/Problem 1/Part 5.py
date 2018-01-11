import numpy
import pandas


data = pandas.read_csv('data1.csv')
X = data[['x1', 'x2']]
Y = data['y']

K_1 = lambda Xi, Xj: (Xi[0] ** 2 + Xi[1] ** 2) * (Xj[0] ** 2 + Xj[1] ** 2)
K_2 = lambda Xi, Xj: (Xi[0] + Xi[1]) * (Xj[0] + Xj[1]) + (Xi[0] ** 2 + Xi[1] ** 2) * (Xj[0] ** 2 + Xj[1] ** 2)
K_3 = lambda Xi, Xj: Xi[0] * Xj[0] + Xi[1] * Xj[1] + (Xi[0] ** 2 + Xi[1] ** 2) * (Xj[0] ** 2 + Xj[1] ** 2)

kernel_matrix1 = numpy.zeros((X.shape[0], X.shape[0]))
kernel_matrix2 = numpy.zeros((X.shape[0], X.shape[0]))
kernel_matrix3 = numpy.zeros((X.shape[0], X.shape[0]))


for i in range(X.shape[0]):
    print(400 - i)
    for j in range(X.shape[0]):
        Xi = X.iloc[i]
        Xj = X.iloc[j]
        kernel_matrix1[i, j] = K_1(Xi, Xj)
        kernel_matrix2[i, j] = K_2(Xi, Xj)
        kernel_matrix3[i, j] = K_3(Xi, Xj)
numpy.savetxt("prek1.csv", kernel_matrix1, delimiter=",")
numpy.savetxt("prek2.csv", kernel_matrix2, delimiter=",")
numpy.savetxt("prek3.csv", kernel_matrix3, delimiter=",")
