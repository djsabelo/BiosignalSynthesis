import numpy as np
from abc import ABCMeta, abstractmethod
from utils.functions.signal2model import *
import time
import sys
import math
import os
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from utils.functions.common import *

GRU_DATA_DIRECTORY = "../data/trained_models/"

class LibphysGRU:
    def __init__(self, signal2model, model_type, parameters):

            # Assign instance variables
            self.model_type = model_type
            if signal2model is not None:
                self.signal_type = signal2model.signal_type
                self.signal_dim = signal2model.signal_dim
                self.batch_size = 0
                if self.signal_dim is None:
                    self.signal_dim = "RG"

                self.hidden_dim = signal2model.hidden_dim
                self.bptt_truncate = signal2model.bptt_truncate
                self.current_learning_rate = signal2model.learning_rate_val
                self.model_name = signal2model.model_name
                self.start_time = 0
                self.signal_dim = signal2model.signal_dim
                self.mini_batch_size = signal2model.mini_batch_size

                # Theano: Created GRU variables
                E, U, W, V, b, c = parameters

                # SGD / rmsprop: Initialize parameters

                # Theano: Created shared variables
                self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
                self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
                self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
                self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
                self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
                self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))

                # SGD / rmsprop: Initialize parameters
                self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
                self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
                self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
                self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
                self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
                self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
            else:
                self.signal_type = 0
                self.signal_dim = 0
                self.hidden_dim = 0
                self.bptt_truncate = -1
                self.current_learning_rate = 0
                self.model_name = ""
                self.start_time = 0
                self.signal_dim = 0
                self.mini_batch_size = 0

                self.E, self.U, self.W, self.V, self.b, self.c = [], [], [], [], [], []

    def calculate_gradients(self, cost, parameters):
        return [T.grad(cost, parameter) for parameter in parameters]

        # threshold = 1.1
        # gradients = []
        # for parameter in parameters:
        #     gradient = T.grad(cost, parameter)
        #     indexes = T.nonzero(T.gt(T.abs_(gradient * cost), threshold), True)
        #     T.set_subtensor(gradient[indexes], \
        #                     threshold * gradient[indexes] \
        #                     * cost / T.abs_(gradient[indexes] * cost))
        #     gradients.append(gradient)
        #return gradients

    def get_m(self, decay, m, d):
        return decay * m + (1 - decay) * d ** 2

    def update_RMSPROP(self, cost, parameters, derivatives, x, y):
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        [E, V, U, W, b, c] = parameters
        [dE, dV, dU, dW, db, dc] = derivatives

        mE = self.get_m(decay, self.mE, dE)
        mU = self.get_m(decay, self.mU, dU)
        mW = self.get_m(decay, self.mW, dW)
        mV = self.get_m(decay, self.mV, dV)
        mb = self.get_m(decay, self.mb, db)
        mc = self.get_m(decay, self.mc, dc)

        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.In(decay, value=0.95)],
            [],
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                     ], allow_input_downcast=True)

    def train_block(self, signals, signal2model, signal_indexes=None, n_for_each=12, overlap=0.33, random_training=True,
                    start_index=0, track_loss=False, loss_interval=1, train_ratio=0.33, validation=False):
        """
        This method embraces several datasets (or one) according to a number of records for each

        :param signals: - list - a list containing two int vectors:
                                    signal - input vector X, used for the input;

        :param signal2model: - Signal2Model object - object containing the information about the model, for more info
                                check Biosignals.utils.functions.signal2model

        :param signal_indexes: - list - a list containing the indexes of the "signals" variable to be trained.
                                        If None is given, all signals will be used.

        :param n_for_each: - int - number of windows from each signal to be inserted in the model training

        :param overlap: - float - value in the interval [0,1] that corresponds to the overlapping ratio of windows

        :param random_training: - boolean - value that if True random windows will be inserted in the training

        :param start_index: - int - value from which the windows will be selected

        :param track_loss: - boolean - value to plot loss as the model is trained

        :return: trained model
        """

        if signal_indexes is None:
            signal_indexes = range(len(signals))

        self.save(signal2model.signal_directory, self.get_file_tag(-1, -1))

        x_train = []
        y_train = []
        for i in signal_indexes:

            # Creation of the Time Windows from the dataset
            if n_for_each == 1:
                if len(x_train) == 0:
                    x_train = signals[i][:signal2model.window_size]
                    y_train = signals[i][1:signal2model.window_size + 1]
                    if validation:
                        x_validation = signals[i][:signal2model.window_size]
                        y_validation = signals[i][1:signal2model.window_size + 1]

                else:
                    x_train = np.vstack((x_train, signals[i][:signal2model.window_size]))
                    y_train = np.vstack((y_train, signals[i][1:signal2model.window_size+1]))
                    if validation:
                        x_validation = np.vstack((x_validation, signals[i][:signal2model.window_size]))
                        y_validation = np.vstack((y_validation, signals[i][1:signal2model.window_size+1]))

            else:
                X_windows, y_end_values, n_windows, last_index = segment_signal(signals[i][:-1], signal2model.window_size,
                                                                                overlap=overlap, start_index=start_index)
                Y_windows, y_end_values, n_windows, last_index = segment_signal(signals[i][1:], signal2model.window_size,
                                                                                overlap=overlap, start_index=start_index)

                # n_for_each = n_for_each if n_for_each < np.shape(X_windows)[0] else np.shape(X_windows)[0]
                # n_for_each = n_for_each if n_for_each % self.mini_batch_size == 0 \
                #     else self.mini_batch_size * int(n_for_each/self.mini_batch_size)

                last_training_index = int(n_windows * train_ratio)
                # List of the windows to be inserted in the dataset
                if random_training:
                    window_indexes = np.random.permutation(last_training_index)  # randomly select windows
                else:
                    window_indexes = list(range((n_windows))) # first windows are selected


                # Insertion of the windows of this signal in the general dataset
                if len(x_train) == 0:
                    # First is for train data
                    x_train = X_windows[window_indexes[0:n_for_each], :]
                    y_train = Y_windows[window_indexes[0:n_for_each], :]

                    if validation:
                        x_validation = X_windows[window_indexes[n_for_each:n_for_each*2], :]
                        y_validation = Y_windows[window_indexes[n_for_each:n_for_each*2], :]


                    # # The rest is for test data
                    # x_test = X_windows[last_training_index:, :]
                    # y_test = Y_windows[last_training_index:, :]
                else:
                    x_train = np.append(x_train, X_windows[window_indexes[0:n_for_each], :], axis=0)
                    y_train = np.append(x_train, Y_windows[window_indexes[0:n_for_each], :], axis=0)
                    # x_test = np.append(x_train, X_windows[window_indexes[n_for_each:], :], axis=0)
                    # y_test = np.append(x_train, Y_windows[window_indexes[n_for_each:], :], axis=0)

                    if validation:
                        x_validation = np.append(x_validation, X_windows[window_indexes[n_for_each:n_for_each*2], :], axis=0)
                        y_validation = np.append(y_validation, Y_windows[window_indexes[n_for_each:n_for_each*2], :], axis=0)

                # Save test data
                # self.save_test_data(signal2model.signal_directory, [x_test, y_test])


        # Start time recording
        self.start_time = time.time()
        t1 = time.time()
        # for x in x_train:
        #     plt.plot(x)
        #     plt.show()
        # Start training model
        if validation:
            returned = self.train_model(x_train, y_train, signal2model, track_loss, loss_interval,
                                        x_validation=x_validation, y_validation=y_validation)
        else:
            returned = self.train_model(x_train, y_train, signal2model, track_loss, loss_interval)
        print("Dataset trained in: ~%d seconds" % int(time.time() - t1))

        # Model last training is then saved
        if returned:
            self.save(signal2model.signal_directory, self.get_file_tag(-5, -5))
            return True
        else:
            return False

    def train(self, X, signal2model, overlap=0.33, random_training=True, start_index=0, loss_interval=1, train_ratio=0.33,
                    validation=False):
        return self.train_block([X],
                                signal2model, [0],
                                signal2model.batch_size,
                                overlap,
                                random_training,
                                start_index,
                                loss_interval,
                                train_ratio=train_ratio,
                                validation=validation)

    def train_model(self, x_train, y_train, signal2model, track_loss=False, loss_interval=1,
                    x_validation=None, y_validation=None, dataset=0):
        # print(x_train)
        # print(y_train)
        self.start_time = time.time()
        if x_validation is None:
            x_validation = x_train
            y_validation = y_train
        self.batch_size = np.shape(x_train)[0]
        loss = [self.calculate_total_loss(x_validation, y_validation)]
        lower_error_threshold, higher_error_threshold = [10**(-5), 1]
        lower_error = signal2model.lower_error
        lower_learning_rate = signal2model.lower_learning_rate
        count_to_break_max = signal2model.count_to_break_max
        count_to_break = 0
        count_up_slope = 0
        test_gradient = False
        is_nan = False
        last_parameters = self.get_parameters()
        # if self.current_learning_rate <= lower_learning_rate:
        #     self.current_learning_rate = 0.00001

        for epoch in range(signal2model.number_of_epochs):
            t_epoch_1 = time.time()

            if lower_learning_rate > self.current_learning_rate:
                break

            if epoch % loss_interval == 0:
                loss.append(self.calculate_total_loss(x_validation, y_validation))
                if epoch == 0:
                    print("Time to calculate loss: {0} ".format(time.time() - t_epoch_1))

                if epoch > 2:
                    self.train_time = int((time.time() - self.start_time) * 1000)
                    print("Loss x100: {0}; Time: {1} min".format((loss[-1] * 100), int(self.train_time/60000)) + str())

                    relative_loss_gradient = (loss[-2] - loss[-1]) / (loss[-2] + loss[-1])
                    if math.isnan(loss[-1]):
                        if is_nan:
                            print("ERROR NaN!!! RECOMPILE THIS ONE!!")
                            return False
                            # self.restart_parameters()
                            # print("Restarting parameters due to NaN loss")
                        else:
                            loss.pop()
                            self.set_parameters(last_parameters)
                            relative_loss_gradient = (loss[-2] - loss[-1]) / (loss[-2] + loss[-1])
                            is_nan = True
                    else:
                        last_parameters = self.get_parameters()
                        is_nan = False

                    if relative_loss_gradient < 0:
                        count_up_slope += 1
                        if count_up_slope >= 5:
                            count_up_slope = 0
                            if np.size(loss) > 10:
                                print("Mean Loss in the last {0} epochs: {1:.3f} < {2:.3f} ?".format
                                      (10, np.mean(loss[-5:]) * 1000, np.mean(loss[-10:-5]) * 1000))
                                if np.mean(loss[-5:]) > np.mean(loss[-10:-5]):
                                    self.current_learning_rate = self.current_learning_rate * 3 / 4
                                    print("Adjusting learning rate: " + str(self.current_learning_rate))
                                #     print("Min Loss in the last {0} epochs: {1:.3f} < {2:.3f} ?".format
                                #       (10, np.min(loss[-5:]) * 1000, np.min(loss[-10:-5]) * 1000))
                                # if np.min(loss[-5:]) > np.min(loss[-10:-5]):
                                #     self.current_learning_rate = self.current_learning_rate * 3 / 4
                                #     print("Adjusting learning rate: " + str(self.current_learning_rate))

                        count_to_break = 0
                    elif relative_loss_gradient > higher_error_threshold:
                        self.current_learning_rate = self.current_learning_rate * 5 / 4
                        count_to_break = 0
                    elif relative_loss_gradient < lower_error_threshold and epoch > 10:
                        self.current_learning_rate = self.current_learning_rate * 3 / 4
                        test_gradient = True
                        count_to_break += 1
                        print("Adjusting learning rate to lower value: " + str(self.current_learning_rate))

                    if count_to_break > count_to_break_max or loss[-1] < lower_error:
                        break

                    elif test_gradient:
                        test_gradient = False

                # if epoch % 10 == 0 and track_loss:
                #     plt.clf()
                #     plt.plot(loss[1:])
                #     plt.ylim([np.min(loss[-20:]), np.max(loss[-100:])])
                #     if epoch % 100 == 0:
                #         plt.ylim([0, np.max(loss)])
                #     plt.pause(0.05)

                t1 = time.time()
            if epoch % 10 == 0 or epoch == 0:
                print("In epoch %d of %d" % (epoch, signal2model.number_of_epochs))
                # For each training example...
            indexes = np.random.permutation(np.shape(y_train)[0])
            for i in range(0, len(indexes), self.mini_batch_size):
                # One SGD step
                ind = indexes[i:i + self.mini_batch_size]
                if self.model_type == ModelType.SGD:
                    N = np.shape(x_train)[-1]
                    self.sgd_step(np.reshape(x_train[i, :], (N)), np.reshape(y_train[i, :],(N)),
                                  self.current_learning_rate, signal2model.decay)
                elif self.model_type == ModelType.CROSS_SGD:
                    N = np.shape(x_train)[-2]
                    M = np.shape(x_train)[-1]
                    self.sgd_step(np.reshape(x_train[i], (N, M)), np.reshape(y_train[i], (N, M)),
                                  self.current_learning_rate, signal2model.decay)
                elif self.model_type == ModelType.CROSS_MBSGD:
                    N = np.shape(x_train)[-2]
                    M = np.shape(x_train)[-1]
                    self.sgd_step(np.reshape(x_train[ind], (self.mini_batch_size, N, M)),
                                  np.reshape(y_train[ind], (self.mini_batch_size, N, M)),
                                  self.current_learning_rate, signal2model.decay)
                else:
                    self.sgd_step(x_train[ind, :], y_train[ind, :], self.current_learning_rate, signal2model.decay)

            if epoch % 10 == 0 or epoch == 0:
                t2 = time.time()
                print("SGD Step time: ~%d seconds" % int(t2 - t1))
                sys.stdout.flush()

            if epoch > 1 and epoch % signal2model.save_interval == 0:
                self.save(dir_name=signal2model.signal_directory, file_tag=self.get_file_tag(dataset, epoch))

            if epoch % 10 == 0 or epoch == 0:
                t2 = time.time()
                print("Epoch time: ~%d seconds" % int(t2 - t_epoch_1))
                sys.stdout.flush()

        return True

    def save(self, dir_name=None, file_tag=None):
        """
        Saves the model according to the file_tag
        :param dir_name: -string - directory name where the corresponding to the model for saving is
                            -> may use model.get_directory_tag(directory_name, batch_size, window_size)
                            -> if given None it will have the value model.get_directory_tag(model_name, 0, 0)

        :param file_tag: - string - file_tag corresponding to the model for loading
                            -> use model.get_file_tag(dataset, epoch)
                            -> if given None it will assume that is the last version of the model get_file_tag(-5,-5)
        :return: None
        """

        if file_tag is None:
            file_tag = self.get_file_tag(-5, -5)

        self.train_time = int((time.time() - self.start_time))

        if dir_name is None:
            dir_name = self.get_directory_tag(self.model_name.upper(), 0, 0)

        dir_name = GRU_DATA_DIRECTORY +dir_name + '/'

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        filename = dir_name + file_tag + '.npz'
        print("Saving model to file: " + filename)
        np.savez(filename,
                 E=self.E.get_value(),
                 U=self.U.get_value(),
                 W=self.W.get_value(),
                 V=self.V.get_value(),
                 b=self.b.get_value(),
                 c=self.c.get_value(),
                 train_time=self.train_time,
                 start_time=self.start_time
                 )

    def load(self, file_tag=None, dir_name=None):
        """
        Loads the model

        :param dir_name: -string - directory name where the corresponding to the model for loading is
                            -> may use model.get_directory_tag(dataset, epoch)

        :param file_tag: - string - file_tag corresponding to the model for loading
                            -> use model.get_file_tag(dataset, epoch)
                            if given None it will assume that is the last version of the model get_file_tag(-5,-5)
        :return: None
        """

        print("Starting sinal loading...")
        if file_tag is None:
            file_tag = self.get_file_tag(-5, -5)

        if dir_name is None:
            dir_name = self.get_directory_tag(self.model_name.upper(), 0, 0)

        dir_name = GRU_DATA_DIRECTORY + dir_name + '/'

        npzfile = np.load(dir_name + file_tag + ".npz")
        E, U, W, V, b, c = [], [], [], [], [], []
        print("Building model from {0} with hidden_dim={1}, signal_dim={2}, epoch={3}".format(
            self.model_name, self.hidden_dim, self.signal_dim, file_tag.split(".")[-1][:-1]))
        try:
            E = npzfile["E"]
            self.E.set_value(E)
        except:
            print("Error loading variable {0}".format("E"))
        try:
            U = npzfile["U"]
            self.U.set_value(U)
        except:
            print("Error loading variable {0}".format("U"))
        try:
            W = npzfile["W"]
            self.W.set_value(W)
        except:
            print("Error loading variable {0}".format("W"))
        try:
            V = npzfile["V"]
            self.V.set_value(V)
        except:
            print("Error loading variable {0}".format("V"))
        try:
            b = npzfile["b"]
            self.b.set_value(b)
        except:
            print("Error loading variable {0}".format("b"))
        try:
            c = npzfile["c"]
            self.c.set_value(c)
        except:
            print("Error loading variable {0}".format("c"))
        try:
            train_time = npzfile["train_time"]
            self.train_time = train_time
        except:
            print("Error loading variable {0}".format("train_time"))
        try:
            start_time=npzfile["start_time"]
            self.start_time = start_time
        except:
            print("Error loading variable {0}".format("start_time"))
        try:
            batch_size = npzfile["start_time"]
            self.batch_size = batch_size
        except:
            print("Error loading variable {0}".format("start_time"))

        sys.stdout.flush()

    def save_test_data(self, dir_name, test_data):
        """
        Saves test data used for the training of this model.
        :param dir_name: - string - directory name where the testing file is
        :param test_data: - list - list[0] - vector with the corresponding X_test (input) time windows
                                   list[1] - vector with the corresponding Y_test (labels) time windows
        :return: None
        """

        print("Saving test data...")
        filename = GRU_DATA_DIRECTORY + dir_name + '/' + self.model_name + '_test_data.npz'
        np.savez(filename, test_data=test_data)

    @staticmethod
    def load_test_data(model_name, dir_name):
        """
        Loads test data used for the training of this model.
        :param model_name: - string - model_name (may access by model.model_name)
        :param dir_name: - string - directory name where the testing file is
        :return: - list - list[0] - vector with the corresponding X_test (input) time windows
                          list[1] - vector with the corresponding Y_test (labels) time windows
        """
        print("Loading test data...")
        filename = GRU_DATA_DIRECTORY + dir_name + '/' + model_name + '_test_data.npz'
        npzfile = np.load(filename)
        return npzfile["test_data"]

    def get_file_tag(self, dataset=-5, epoch=-5):
        """
        Gives a standard name for the file, depending on the #dataset and #epoch
        :param dataset: - int - dataset number
                        (-1 if havent start training, -5 when the last batch training condition was met)
        :param epoch: - int - the last epoch number the dataset was trained
                        (-1 if havent start training, -5 when the training condition was met)
        :return: file_tag composed as GRU_SIGNALNAME[SD.HD.BTTT.DATASET.EPOCH] -> example GRU_ecg[64.16.0.-5]
        """

        return 'GRU_{0}[{1}.{2}.{3}.{4}.{5}]'.\
                    format(self.model_name, self.signal_dim, self.hidden_dim, self.bptt_truncate, dataset, epoch)


    def get_directory_tag(self, dir_name=None, B=128, W=256):
        """
        Gives a standard name to the directoy.

        :param dir_name: - string - TAG for the directory name - discribing the dataset for training
        :param B: - int - Batch size
        :param W: - int - Window size

        :return: Standard directory name composed as TAG[B.W] -> example ECG[256.128]
        """
        if dir_name is None:
            dir_name = self.model_name.upper()

        return dir_name+'[{0}.{1}]'.format(B, W)

    def get_parameters(self):
        return [self.E.get_value(), self.V.get_value(), self.U.get_value(), self.W.get_value(), self.b.get_value(),
                self.c.get_value()]

    def set_parameters(self, parameters):
        self.E, self.V, self.U, self.W, self.b, self.c = parameters

    def restart_parameters(self):
        E, V, U, W, b, c = self._get_new_parameters()

        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))

        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

    @abstractmethod
    def _get_new_parameters(self):
        pass