import numpy as np
from scipy.io import loadmat

from benchopt import BaseDataset
from benchopt import config


class Dataset(BaseDataset):

    name = "urban"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'estimated_rank' : [6]
    }

    def get_data(self):
        path = config.get_data_path("Urban.mat")
        
        # Loading the data
        dico = loadmat(path)

        # dict is a python dictionnary. It contains the matrix we want to NMF
        M = np.transpose(dico['A']) # permutation because we like spectra in W
        m, n = M.shape

        # It can be nice to normalize the data
        M = M/np.max(M)
        return dict(X=M, rank=self.estimated_rank, true_factors=None)