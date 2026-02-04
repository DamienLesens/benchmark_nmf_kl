import numpy as np
from scipy.io import loadmat

from benchopt import BaseDataset
from benchopt import config


class Dataset(BaseDataset):

    # https://cs.nyu.edu/home/people/in_memoriam/roweis/data.html

    name = "Frey"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'estimated_rank' : [25]
    }

    def get_data(self):
        path = config.get_data_path("frey_rawface.mat")
        
        # Loading the data
        dico = loadmat(path)

        # dict is a python dictionnary. It contains the matrix we want to NMF
        M = dico['ff']

        return dict(X=M, rank=self.estimated_rank, true_factors=None)