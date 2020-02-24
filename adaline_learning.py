import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

class Adaline:
    def __init__(self, rate=0.01, n_iter=10):
        self.rate = rate
        self.n_iter = n_iter

    def fit(self, X, y):  # X(matrix) is the training sample, y(vector) is the correct label
        self.w = np.zeros(X.shape[1] + 1)  # X.shape returns the (No.rows, No.columns)
        self.costs = []
        for _ in range(self.n_iter):
            output = self.net_input(X).T  # array-like, shape = [n_samples]
            errors = (y.T-output)[0]    # array-like, shape = [n_samples]
            X_t = X.T
            product = self.rate*(X_t @ errors)
            self.w[1:] += self.rate*product # X needs to be transposed first to do dot multiplication
            self.w[0] += self.rate*(errors.sum())
            cost = (errors**2).sum()*0.5        # numpy allows every element to be squared simultaneously
            self.costs.append(cost)
        return self

    def net_input(self,X):
        return np.dot(X,self.w[1:])+self.w[0]

    def predict(self, sample):
        net_input = np.dot(sample, self.w[1:]) + self.w[0]
        return np.where(net_input >= 0.5, 1, 0)

def standardization(X):
    x_std = np.copy(X)
    x_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:,0].std()
    x_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:,1].std()
    return x_std

def read_matrix(filename):
    """
    Parse data from the file with the given filename into a matrix.

    input:
        - filename: a string representing the name of the file

    returns: a matrix containing the elements in the given file
    """
    file_data_frame = pd.read_csv(filename)
    matrix = file_data_frame.values
    return matrix
train_stat = read_matrix(r"E:\Rice\datathon\divorce\processed data\training data\divorce_training_csv.csv")
train_divorces = read_matrix(r"E:\Rice\datathon\divorce\processed data\training data\divorce_training_out.csv")
test_stat = read_matrix(r"E:\Rice\datathon\divorce\processed data\test data\test_data_csv.csv")
test_divorces = read_matrix(r"E:\Rice\datathon\divorce\processed data\test data\Test_data_out.csv")
df = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data', header=None)

X_std = standardization(train_stat)   # standardize data
ada_std = Adaline(n_iter=15)
ada_std.fit(X_std,train_divorces)
prediction = ada_std.predict(test_stat)
def error_percent(predictions, actual_result):
    """
    I remember docstring!
    """
    error = 0
    for i in zip(predictions,actual_result):
        if i[0] != i[1]:
            error +=1
    return error/len(predictions)
print(error_percent(prediction,test_divorces))

"""
plt.figure(1)
plt.plot(range(1,len(ada_std.costs)+1),ada_std.costs)
plt.xlabel('epoch')
plt.ylabel('Sum-Square_Error')
"""
"""
def decision(X,y,classifier, resolution= 0.02):
    markers= ('o','x')
    colors= ('red','blue')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z= classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

plt.figure(2)

decision(X_std,y,classifier=ada_std)
plt.show()
"""

