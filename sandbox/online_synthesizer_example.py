from utils.functions.common import *
import models.LibphysSGDGRU as GRU
from utils.functions.signal2model import Signal2Model


# Please do the processing in the following order -> process_ecg_example -> train_ecg_example before this one

name = "ecg_1"
model_dir = ""
signal_dim = 512  # this parameter represents the steps of the quantization process
hidden_dim = 256  # size of the network in terms of hidden units in the GRU cells
mini_batch_size = 16  # size of the minibatch used for traininge
window_size = 512  # the window size for the segmentation
filename = "synthesized_0.npz"

signal2model = Signal2Model(name, model_dir, signal_dim=signal_dim, hidden_dim=hidden_dim,
                            mini_batch_size=mini_batch_size, window_size=window_size)
model = GRU.LibphysSGDGRU(signal2model)

model.load(dir_name=model_dir)

model.online_sinthesizer(20000, [np.random.randint(signal_dim)], window_seen_by_GRU_size=window_size, uncertaintly=0.1)