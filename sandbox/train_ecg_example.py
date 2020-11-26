import time

import numpy as np

from utils.functions.signal2model import Signal2Model
from utils.functions.common import *
import models.LibphysMBGRU as GRU




def train_fantasia(hidden_dim, mini_batch_size, batch_size, window_size, model_directory, indexes, signals,save_interval,signal_dim, number_of_epochs=10000):
    for i, signal in enumerate(signals[indexes]):
        name = 'ecg_' + str(i)

        signal2model = Signal2Model(name, model_directory, signal_dim=signal_dim, hidden_dim=hidden_dim, batch_size=batch_size,
                                    mini_batch_size=mini_batch_size, window_size=window_size,
                                    save_interval=save_interval, lower_error=3e-5, lower_learning_rate=1e-4,
                                    number_of_epochs=number_of_epochs, count_to_break_max=30)
        print("Compiling Model {0}".format(name))

        last_index = int(len(signal)*0.33)
        x_train, y_train = prepare_test_data([signal[:last_index]], signal2model, mean_tol=0.9, std_tol=0.5)
        # It removes the first part that is often corrupted, segments, and extracts the training part
        model = GRU.LibphysMBGRU(signal2model)
        # Creates the model with the parameters

        print("Initiating training... ")
        model.model_name = 'ecg_' + str(i+1)

        model.start_time = time.time()
        # if this model exists you can load the previous version. If the gradient explodes it will stop and returned
        # becomes "False"
        # model.load(model.get_file_tag(), model_directory)
        returned = model.train_model(x_train, y_train, signal2model)
        if returned:
            model.save("", model.get_file_tag(-5, -5))


if __name__ == '__main__':
    signal_dim = 512      # this parameter represents the steps of the quantization process
    hidden_dim = 256      # size of the network in terms of hidden units in the GRU cells
    mini_batch_size = 16  # size of the minibatch used for training
    batch_size = 256      # size of the batch which is trained at a time
    window_size = 512     # the window size for the segmentation
    save_interval = 100   # how much epochs before saving the model
    model_directory = ''
    # Directory in which the model is going to be saved

    indexes = np.arange(2)

    print("Loading signals...")
    filename = "../data/processed/processed_ecg_signals.npz"
    signals = np.load(filename)["processed_signals"]
    train_fantasia(hidden_dim, mini_batch_size, batch_size, window_size, model_directory, indexes, signals, save_interval,
                   signal_dim, number_of_epochs=1)
