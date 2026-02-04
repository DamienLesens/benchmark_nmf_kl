import numpy as np
from scipy.io import loadmat
from pypnm import pnmlpnm

from benchopt import BaseDataset
from benchopt import config


class Dataset(BaseDataset):

    # https://github.com/galeone/mlcnn/tree/master/mitcbcl/train/face

    name = "MIT-CBCL-face"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'estimated_rank' : [49]
    }

    def get_data(self):
        path_base = str(config.get_data_path("MIT-CBCL-face"))
        
        V = np.zeros((361,2429))

        for i in range(1,2430):
            filename = "/face"+"0"*(5-len(str(i)))+str(i)+".pgm"
            # print(filename)
            X, Y, Z, maxcolors, image3D = pnmlpnm.pnm2list(path_base+filename)

            V[:,i-1]=np.array(image3D).flatten()
        
        return dict(X=V, rank=self.estimated_rank, true_factors=None)