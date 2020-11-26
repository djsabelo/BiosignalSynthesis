import matplotlib.pyplot as plt
from utils.functions.common import process_dnn_signal
import numpy as np
import scipy.io as sio
from novainstrumentation import smooth
from utils.functions.common import remove_moving_std, remove_moving_avg, process_dnn_signal, quantitize_signal

def process_and_save_ecgs(signals_array, plot=False, signal_dim=512):
    """
    This function is aimed to process the ecg before entering the Deep Learning model.
    In some it removes the wandering mean and std variation, quantizes, normalize, and finally saves it

    If the plot is set to "True", be aware that the cycle will stop in everytime it plots.
    :param file_paths: this takes in an array of signals and processes them one by one.
    :param plot: This is set to "False", if you want to plot a part of the signal, to check how the signal is being transformed set it to "True"
    :param signal_dim: How many "steps" do you want in the quantization processing
    :return:
    """

    processed_signals = []
    for ecg_signal in signals_array:

        ecg_signal = ecg_signal - np.mean(ecg_signal) # This removes the mean of all signal
        signal_without_avg = remove_moving_avg(ecg_signal) # This removes the wandering average of all signal
        signal_without_std = remove_moving_std(signal_without_avg) # This removes the wandering standard deviation of
                                                                   # all signal
        smoothed_signal = smooth(signal_without_std)# Smooths by applying a convolution with a hanning window
        processed_signal = quantitize_signal(# Normalization and quantization
            smoothed_signal,
            signal_dim, # where signal_dim is the number of # "steps" of the quantization process
            0.001) # confidence is how much of the signal is to be croped



        # Or just use this function that does the last steps:
        # processed = process_dnn_signal(signal, signal_dim)

        processed_signals.append(processed_signal) # Adds to the processed array

        if plot:
            plt.plot((ecg_signal[1000:5000]-np.min(ecg_signal))*signal_dim/np.max(ecg_signal-np.min(ecg_signal)),
                     label="Original Signal")
            plt.plot(processed_signal[1000:5000], label="Quantized Signal")
            plt.legend()
            plt.show()

    print("Saving signals...")
    np.savez("../data/processed/processed_ecg_signals.npz", processed_signals=processed_signals)

if __name__ == '__main__':
    # Preprocessing example of 2 datasets
    # 2 examples of Fantasia dataset, feel free to try others! Remember this dataset has a sampling rate of 250Hz!
    fantasia_file_paths = ["../data/raw/f1y10m.mat", "../data/raw/f1o10m.mat"]
    signals_array = []
    for file_path in fantasia_file_paths:
        try:
            print("Pre-processing signal - " + file_path)
            # This is a subset of data to be less computationally demanding
            signal = sio.loadmat(file_path)['val'][1][:500000]
            signals_array += [signal]
        except:
            print("Pre-processing signal - " + file_path +
                  " failed. Check if file is in the file_path and if the file is .mat")

    process_and_save_ecgs(signals_array, plot=True, signal_dim=512)




