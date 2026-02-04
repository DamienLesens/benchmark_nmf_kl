import numpy as np
from scipy.io import loadmat
from pypnm import pnmlpnm

from benchopt import BaseDataset
from benchopt import config


class Dataset(BaseDataset):

    # https://www.kaggle.com/datasets/kasikrit/att-database-of-faces

    name = "ORLfaces"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'estimated_rank' : [25]
    }

    def get_data(self):
        path_base = str(config.get_data_path("ORL6"))
        
        V = np.zeros((112*92,400))
        for i in range(1,41):
            for j in range(1,11):
                filename = "/s"+str(i)+"/"+str(j)+".pgm"
                # print(filename)
                X, Y, Z, maxcolors, image3D = pnmlpnm.pnm2list(path_base+filename)
                # print(image3D)
                V[:,(i-1)*10+j-1]=np.array(image3D).flatten()
        
        return dict(X=V, rank=self.estimated_rank, true_factors=None)