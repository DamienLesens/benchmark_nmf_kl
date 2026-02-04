import numpy as np
import soundfile as sf
from scipy import signal

from benchopt import BaseDataset
from benchopt import config


class Dataset(BaseDataset):

    # https://www.irit.fr/~Cedric.Fevotte/extras/icassp11/

    name = "MyHeart"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'estimated_rank' : [10] #
    }

    def get_data(self):
        path = config.get_data_path("MyHeart.wav")
        
        the_signal, sampling_rate_local = sf.read(path)

        #Following the set up from Févotte
        l_win = 256
        overlap = l_win/2
        frequencies, time_atoms, Y = signal.stft(the_signal, fs=sampling_rate_local, nperseg=l_win, nfft=l_win, noverlap=overlap)
        
        # Taking the amplitude spectrogram
        Y = np.abs(Y)**2

        return dict(X=Y, rank=self.estimated_rank, true_factors=None)