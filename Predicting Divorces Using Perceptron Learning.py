import numpy as np
import pandas as pd
import pylab as pyl

class perceptron():
    def __init__(self, X_training, rate=0.1, n_iter=10):
        """
        :param X_training: numpy matrix, the training data that is used to fit the model
        :param rate: float, the learning rate
        :param n_iter: int, the number of iterations for learning
        """

        self.eta= rate
        self.n_iter= n_iter
        self.w = np.zeros(1 + X_training.shape[1]) #allow for the threshold
        self.errors = []

    def fit(self,X,y):
        '''
        :param X: numpy matrix, the training data that is used to fit the model
        :param y: 1D numpy matrix, the actual class of the training set
        :return: self with updated weight, w
        '''

        for _ in range(self.n_iter):
            error= 0
            for xi,label in zip(X,y):
                update = self.eta *(label- self.predict(xi))
                error += int(update!=0.0)
                self.w[1:] += update*xi
                self.w[0] += update
            self.errors.append(error)
        return self

    def predict(self,X):
        """

        :param X: numpy matrix, input data
        :return: 1D numpy matrix, the prediction based on input data X
        """
        dot_product = np.dot(X, self.w[1:]) + self.w[0]
        return np.where(dot_product>=0.5, 1, 0)

    def errors_graph(self):
        pyl.plot(range(1, len(self.errors) + 1), self.errors)
        pyl.xlabel('Number of Iterations')
        pyl.ylabel('Errors')
        pyl.title('Number of Errors Over Each Iteration')
        pyl.show()

def read_matrix(filename):
    """
    Parse data from the file with the given filename into a matrix.

    input:
        - filename: a string representing the name and the address of the file

    returns: a numpy matrix containing the elements in the given file
    """
    file_data_frame = pd.read_csv(filename)
    matrix = file_data_frame.values
    return matrix

def accuracy(predictions, actual_result):
    """

    :param predictions: 1D array
    :param actual_result: 1D array
    :return: the accuracy of the result
    """
    error = 0
    for i in zip(predictions,actual_result):
        if i[0] != i[1]:
            error +=1
    return 1- (error/len(predictions))

def run(train_stat,train_label, test_stat, test_label):
    ppn = perceptron(train_stat)
    ppn.fit(train_stat, train_label)
    result = ppn.predict(test_stat)
    accuracy_percent = accuracy(result, test_label)
    print("The accuracy using perceptron learning to predict divorces is ", accuracy_percent)
    ppn.errors_graph()

train_stat = read_matrix(r"E:\Rice\datathon\divorce\processed data\training data\divorce_training_csv.csv")
train_labels = read_matrix(r"E:\Rice\datathon\divorce\processed data\training data\divorce_training_out.csv")
test_stat = read_matrix(r"E:\Rice\datathon\divorce\processed data\test data\test_data_csv.csv")
test_labels = read_matrix(r"E:\Rice\datathon\divorce\processed data\test data\Test_data_out.csv")

run(train_stat,train_labels,test_stat,test_labels)













