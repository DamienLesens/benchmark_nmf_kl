import numpy as np

from benchopt import BaseDataset
from benchopt import config

class Dataset(BaseDataset):

    # https://github.com/roychowdhuryresearch/eNMF/blob/main/Dataset/verb/right_matrix.npy
    # NMF article: "An Exterior Method for Nonnegative Matrix Factorization" Qiujing Lu * 1 Tonmoy Monsoor * 1 Ehsan Ebrahimzadeh 2 Kartik Sharma 1 Vwani Roychowdhury 
    # data article: "“Mommy Blogs” and the Vaccination Exemption Narrative: Results From A Machine-Learning Approach for Story Aggregation on Parenting Social Media Sites"

    name = "Verb"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'estimated_rank' : [10,20,40,80,100]
    }

    def get_data(self):
        path = config.get_data_path("right_matrix.npy")
        
        # Loading the data
        M = np.load(path)

        return dict(X=M, rank=self.estimated_rank, true_factors=None)