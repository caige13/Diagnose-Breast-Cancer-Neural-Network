#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import enum
import Graphing
import os


class Err(str, enum.Enum):
    mse = "mean_squared_error"
    bin = "binary_crossentropy"


class History:
    def __init__(self, test_size: float, learning_rate: float, epoch: int,
                 hidden_layer_count: int, unit_count: [int], activation: [str]):
        self.epoch = epoch
        self.activation = activation
        self.learning_rate = learning_rate
        self.hidden_layer_count = hidden_layer_count
        self.unit_count = unit_count
        self.test_size = test_size
        self.best_test_accuracy = -10
        self.best_train_accuracy = -10
        self.best_test_err = ""
        self.best_train_err = ""
        self.train_r2 = {
            Err.mse.value: 0,
            Err.bin.value: 0
        }
        self.test_r2 = {
            Err.mse.value: 0,
            Err.bin.value: 0
        }
        self.train_err = {
            Err.mse.value: 0,
            Err.bin.value: 0
        }
        self.test_err = {
            Err.mse.value: 0,
            Err.bin.value: 0
        }
        self.test_accuracy = {
            Err.mse.value: 0,
            Err.bin.value: 0
        }
        self.train_accuracy = {
            Err.mse.value: 0,
            Err.bin.value: 0
        }

    def set_r2(self, key, train, test):
        self.train_r2[key] = train
        self.test_r2[key] = test

    def set_err(self, key, train, test):
        self.train_err[key] = train
        self.test_err[key] = test

    def set_accuracy(self, key, train, test):
        self.test_accuracy[key] = test
        self.train_accuracy[key] = train

    def eval_accuracy(self):
        for err in Err:
            if self.test_accuracy[err.value] > self.best_test_accuracy:
                self.best_test_accuracy = self.test_accuracy[err.value]
                self.best_test_err = err.value
            if self.train_accuracy[err.value] > self.best_train_accuracy:
                self.best_train_accuracy = self.train_accuracy[err.value]
                self.best_train_err = err.value

class NeuralNet:
    def __init__(self, dataFile, seperator, header=True):
        self.raw_input = pd.read_csv(dataFile, seperator)
        self.dataframe_columns = list(self.raw_input.columns)
        self.processed_data = None

    def correlation_matrix(self, path):
        if self.processed_data.empty:
            self.raw_input.corr('pearson').to_excel(path)
        else:
            self.processed_data.corr('pearson').to_excel(path)

    def preprocess(self, method='normalization', convertCategoryToNum: [str] = [],
                   dropColumns: [str] = [], delEntryContent: [str] = ["?"], delBasedOnColumn: [str] = []):

        if len(dropColumns) > 0:
            self.processed_data = self.raw_input.drop(dropColumns, axis=1)
            for i in range(0, len(dropColumns)):
                for z in range(0, len(self.dataframe_columns)):
                    if dropColumns[i] == self.dataframe_columns[z]:
                        self.dataframe_columns[z] = "delete_me"
            for i in range(0, len(self.dataframe_columns)):
                if i < len(self.dataframe_columns) and self.dataframe_columns[i] == "delete_me":
                    del self.dataframe_columns[i]
        else:
            self.processed_data = self.raw_input

        if len(convertCategoryToNum) > 0:
            for factorize in convertCategoryToNum:
                self.processed_data[factorize] = pd.factorize(self.processed_data[factorize])[0]

        self.processed_data.dropna()
        self.processed_data.drop_duplicates()

        if len(delBasedOnColumn) > 0:
            for col in delBasedOnColumn:
                for content in delEntryContent:
                    self.processed_data = self.processed_data[self.processed_data[col] != content]
                self.processed_data = self.processed_data.reset_index(drop=True)

        if method == 'normalization':
            normalized_data = preprocessing.normalize(self.processed_data.iloc[:, 1:])
            normalized_data = np.insert(arr=normalized_data, obj=0, values=self.processed_data['class'].to_numpy(),
                                        axis=1)
            self.processed_data = pd.DataFrame(normalized_data, columns=self.dataframe_columns)
        elif method == 'standardization':
            standardized_data = preprocessing.scale(self.processed_data.iloc[:, 1:])
            standardized_data = np.insert(arr=standardized_data, obj=0, values=self.processed_data['class'].to_numpy(),
                                          axis=1)
            self.processed_data = pd.DataFrame(standardized_data, columns=self.dataframe_columns)

    def __train_helper(self, history: History):
        if len(history.unit_count) == history.hidden_layer_count and \
                len(history.activation) == 2:
            if history.activation[1] == "sigmoid":
                threshold = .5
            else:
                threshold = 0
            ncols = len(self.processed_data.columns)
            X = self.processed_data.iloc[:, 1:(ncols)]
            y = self.processed_data.iloc[:, 0]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=history.test_size, random_state=5)

            classifier = Sequential()
            # Input Layer, but keras treats this as a hidden layer its just the first hidden layer.
            classifier.add(Dense(units=history.unit_count[0], activation=history.activation[0], input_dim=ncols - 1))
            # Add all the Hidden layers
            for layer in range(1, history.hidden_layer_count):
                classifier.add(Dense(units=history.unit_count[layer], activation=history.activation[0]))
            # Output Layer
            classifier.add(Dense(units=1, activation=history.activation[1]))

            # Does all of the error functions in the enum Err and store into history for the hyper parameter config.
            for err in Err:
                opt = tf.keras.optimizers.Adam(learning_rate=history.learning_rate)
                classifier.compile(optimizer=opt, loss=err.value)
                
                # allows me to re use the same neural network set up with initial weights for diff loss function.
                classifier.save_weights("my_model")
                classifier.fit(X_train, y_train, batch_size=1, epochs=history.epoch, verbose=0)
                Y_test_real_pred = classifier.predict(X_test)
                Y_train_real_pred = classifier.predict(X_train)
                Y_train_pred = []
                Y_test_pred = []
                for y in Y_test_real_pred:
                    if y >= threshold:
                        Y_test_pred.append(1)
                    elif y < threshold:
                        Y_test_pred.append(0)
                for y in Y_train_real_pred:
                    if y >= threshold:
                        Y_train_pred.append(1)
                    elif y < threshold:
                        Y_train_pred.append(0)

                total = 0
                correct = 0
                wrong = 0
                Y_test = y_test.to_numpy()
                for i in range(0, len(Y_test_pred)):
                    total = total + 1
                    if Y_test[i] == float(Y_test_pred[i]):
                        correct = correct + 1
                    else:
                        wrong = wrong + 1
                total_train = 0
                correct_train = 0
                wrong_train = 0
                Y_train = y_train.to_numpy()
                for i in range(0, len(Y_train_pred)):
                    total_train = total + 1
                    if Y_train[i] == float(Y_train_pred[i]):
                        correct_train = correct + 1
                    else:
                        wrong_train = wrong + 1
                print("Test Size: ", history.test_size, " learning Rate: ", history.learning_rate, " epoch: ",
                      history.epoch, " Error Func: ", err.value, "\nHidden Layer Count: ",
                      history.hidden_layer_count, " Unit Count: ", history.unit_count, " activations: ", history.activation)
                print("Test Total ", total, "Test Correct ", correct, "Test Wrong ", wrong)
                print("Train Total ", total_train, " Train Correct ", correct_train, " Train Wrong ", wrong_train)

                history.set_err(err.value, classifier.evaluate(X_train, y_train), classifier.evaluate(X_test, y_test))
                history.set_accuracy(err.value, float(correct_train)/float(total_train), float(correct)/float(total))
                history.set_r2(err.value, r2_score(y_pred=Y_train_pred, y_true=y_train),
                               r2_score(y_pred=Y_test_pred, y_true=y_test))
                classifier.load_weights("my_model")

            return history

    # Does every combination of the hyper parameters and used helper function for training.
    def train_evaluate(self):
        history = []
        # If you want to see this with more parameters you can comment out and uncomment as you want
        # But it starts taking a long time with too many because it does every possible combination.
        test_sizes = [.2]
        learning_rate = [0.01, 0.1]
        epochs = [10, 50, 100]
        num_hidden_layers = [3, 4]
        unit_count = [[16, 8, 6], [20, 12, 8, 4]]
        activations = [["tanh", "sigmoid"], ["sigmoid", "sigmoid"], ["relu", "sigmoid"]]
        # test_sizes = [.2]
        # learning_rate = [.001, .01, .1, 1]
        # epochs = [50, 100]
        # # 3 max
        # num_hidden_layers = [3]
        # unit_count = [[16, 8, 6]]
        # activations = [["tanh", "sigmoid"]]

        for epoch in epochs:
            for rate in learning_rate:
                for size in test_sizes:
                    for activation in activations:
                        for i in range(0, 1):
                            history.append(self.__train_helper(
                                History(size, rate, epoch, num_hidden_layers[0], unit_count[i], activation)))
                        if len(num_hidden_layers) > 1:
                            for i in range(1, 2):
                                history.append(self.__train_helper(
                                    History(size, rate, epoch, num_hidden_layers[1], unit_count[i], activation)))
                        if len(num_hidden_layers) > 2:
                            for i in range(2,3):
                                history.append(self.__train_helper(
                                    History(size, rate, epoch, num_hidden_layers[2], unit_count[i], activation)))

        df = pd.DataFrame()
        df['best_test_accuracy'] = []
        df['best_train_accuracy'] = []
        df['epoch'] = []
        df['learning_rate'] = []
        df['hidden_layer_count'] = []
        df['unit_count'] = []
        df['test_size'] = []
        df['best_test_error_func'] = []
        df['best_train_error_func'] = []
        df['hidden_layer_activation'] = []
        df['train_r2_MSE'] = []
        df['train_MSE'] = []
        df['test_r2_MSE'] = []
        df['test_MSE'] = []
        df['train_r2_cross_entropy'] = []
        df['train_cross_entropy'] = []
        df['test_r2_cross_entropy'] = []
        df['test_cross_entropy'] = []
        for hist in history:
            hist.eval_accuracy()
            new_row = {
                'best_test_accuracy': hist.best_test_accuracy,
                'best_train_accuracy': hist.best_train_accuracy,
                'epoch': hist.epoch,
                'learning_rate': hist.learning_rate,
                'hidden_layer_count': hist.hidden_layer_count,
                'unit_count': hist.unit_count,
                'test_size': hist.test_size,
                'best_test_error_func': hist.best_test_err,
                'best_train_error_func': hist.best_train_err,
                'hidden_layer_activation': hist.activation[0],
                'train_r2_MSE': hist.train_r2[Err.mse.value],
                'train_MSE': hist.train_err[Err.mse.value],
                'test_r2_MSE': hist.test_r2[Err.mse.value],
                'test_MSE': hist.test_err[Err.mse.value],
                'train_r2_cross_entropy': hist.train_r2[Err.bin.value],
                'train_cross_entropy': hist.train_err[Err.bin.value],
                'test_r2_cross_entropy': hist.test_r2[Err.bin.value],
                'test_cross_entropy': hist.test_err[Err.bin.value]
            }
            df = df.append(new_row, ignore_index=True)


        Graphing.accuracy_vs_epoch([df['best_test_accuracy']],df['epoch'], "Graph/accuracy_vs_epoch/test_accuracy.png",
                                           "test")
        Graphing.accuracy_vs_epoch([df['best_train_accuracy']], df['epoch'], "Graph/accuracy_vs_epoch/train_accuracy.png",
                                           "train")
        Graphing.accuracy_vs_epoch([df['best_train_accuracy'], df['best_test_accuracy']], df['epoch'],
                                   "Graph/accuracy_vs_epoch/all_accuracy.png", "All", ["Train", "Test"])
        Graphing.accuracy_vs_epoch([df['best_train_accuracy'], df['best_test_accuracy']], df['epoch'],
                                   "Graph/accuracy_vs_epoch/all_accuracy.png", "All", ["Train", "Test"])
        learn_rate = []
        train_acc = []
        test_acc = []
        for i in range(0, len(epochs)):
            temp_df = df[df['epoch'] == epochs[i]]
            learn_rate.append(temp_df['learning_rate'])
            train_acc.append(temp_df['best_train_accuracy'])
            test_acc.append(temp_df['best_test_accuracy'])
        Graphing.accuracy_vs_learning_rate([df['best_test_accuracy']], df['learning_rate'], "Graph/accuracy_vs_learning_rate/test_accuracy.png",
                                           "Test", [])
        Graphing.accuracy_vs_learning_rate([df['best_train_accuracy']], df['learning_rate'], "Graph/accuracy_vs_learning_rate/train_accuracy.png",
                                           "Train", [])
        Graphing.accuracy_vs_learning_rate([df['best_train_accuracy'], df['best_test_accuracy']], df['learning_rate'],
                                           "Graph/accuracy_vs_learning_rate/all_accuracy", "All", ["Train", "Test"])
        Graphing.accuracy_vs_learning_rate_epoch_all([train_acc, test_acc], learn_rate,
                                           "Graph/accuracy_vs_learning_rate/all_accuracy_with_epoch", "All", epochs)
        Graphing.layer_count_vs_err(df['hidden_layer_count'], df['train_MSE'],
                                    "Graph/layer_count_vs_err/train_MSE", "Trained MSE")
        Graphing.layer_count_vs_err(df['hidden_layer_count'], df['test_MSE'],
                                    "Graph/layer_count_vs_err/test_MSE", "Test MSE")
        Graphing.layer_count_vs_err(df['hidden_layer_count'], df['train_cross_entropy'],
                                    "Graph/layer_count_vs_err/train_crossentropy", "Train Binary Crossentropy")
        Graphing.layer_count_vs_err(df['hidden_layer_count'], df['test_cross_entropy'],
                                    "Graph/layer_count_vs_err/test_crossentropy", "Test Binary Crossentropy")
        Graphing.layer_count_vs_err(df['hidden_layer_count'], df['test_cross_entropy'],
                                    "Graph/layer_count_vs_err/test_crossentropy", "Test Binary Crossentropy")
        Graphing.layer_count_vs_err(df['hidden_layer_count'], df['train_r2_MSE'],
                                    "Graph/layer_count_vs_err/train_r2_mse", "Train R^2 for MSE Portion")
        Graphing.layer_count_vs_err(df['hidden_layer_count'], df['test_r2_MSE'],
                                    "Graph/layer_count_vs_err/test_r2_mse", "test R^2 for MSE Portion")
        Graphing.layer_count_vs_err(df['hidden_layer_count'], df['test_r2_cross_entropy'],
                                    "Graph/layer_count_vs_err/test_r2_crossentropy", "test R^2 for Binary Crossentropy Portion")
        Graphing.layer_count_vs_err(df['hidden_layer_count'], df['train_r2_cross_entropy'],
                                    "Graph/layer_count_vs_err/train_r2_crossentropy", "train R^2 for Binary Crossentropy Portion")
        Graphing.layer_count_vs_all_same_err(df['hidden_layer_count'], df['train_MSE'],
                                        df['train_cross_entropy'], "Graph/layer_count_vs_err/all_train_error", "Train")
        Graphing.layer_count_vs_all_same_err(df['hidden_layer_count'], df['test_MSE'], df['test_cross_entropy'],
                                        "Graph/layer_count_vs_err/all_test_error", "Test")
        Graphing.layer_count_vs_all_err(df['hidden_layer_count'], df['test_MSE'], df['test_cross_entropy'],
                                        df['train_MSE'], df['train_cross_entropy'], "Graph/layer_count_vs_err/all_error")
        df.to_excel("Output_Table.xlsx")


if __name__ == "__main__":
    neural_network = NeuralNet("https://raw.githubusercontent.com/caige13/Diagnose-Breast-Cancer-Neural-Network/main/wdbc.data", ",")
    neural_network.preprocess(method="standardization", dropColumns=["id"], convertCategoryToNum=['class'])
    # neural_network.correlation_matrix('Data Info/CorrelationMatrix.xlsx')
    # neural_network.processed_data.to_csv("C:/Dev/Neural network/Data Info/data.data")
    if(os.path.exists("output_table.xlsx")):
        print("Please delete or rename the \"output_table.xlsx\" file.")
        exit(0)
    print("In an attempt to save your time: Reminder to close all graphs that are opened for this project."
          "Other wise it will error out after training.")
    input("Enter anything: ")
    neural_network.train_evaluate()
