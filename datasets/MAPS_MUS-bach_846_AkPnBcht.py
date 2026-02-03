import numpy as np
import soundfile as sf
from scipy import signal

from benchopt import BaseDataset
from benchopt import config


class Dataset(BaseDataset):

    name = "MAPS_MUS-bach_846_AkPnBcht"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'target_duration': [30.0],
        'max_freq': [5300.0],
        'target_sr': [11025],
        'n_fft': [2048],
        'hop_length': [256],
        'estimated_rank' : [2, 11, 23, 45] #
    }

    def get_data(self):
        path = config.get_data_path("MAPS_MUS-bach_846_AkPnBcht.wav")
        # path = config.get_data_path(key="MAPS_MUS-bach_846_AkPnBcht_path")
        
        the_signal, sampling_rate_local = sf.read(path)
        # Using the settings of the Attack-Decay transcription paper
        the_signal = the_signal[:,0] # left channel only
        frequencies, time_atoms, Y = signal.stft(the_signal, fs=sampling_rate_local, nperseg=4096, nfft=8192, noverlap=4096 - 882)
        time_step = time_atoms[1] #20 ms
        freq_step = frequencies[1] #5.3 hz
        # Taking the amplitude spectrogram
        Y = np.abs(Y)
        # Cutting silence, end song and high frequencies (>5300 Hz)
        cutf = 1000 
        cutt_in = int(1/time_step) # song beginning after 1 second
        cutt_out = int(30/time_step)# 30seconds with 20ms steps #time_atoms.shape[0]
        Y = Y[:cutf, cutt_in:cutt_out]
        # normalization
        Y = Y/np.max(Y)  # does not change much
        return dict(X=Y, rank=self.estimated_rank, true_factors=None)