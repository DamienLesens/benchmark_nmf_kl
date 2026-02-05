import numpy as np
from scipy.io import loadmat
from scipy.sparse import coo_array

from benchopt import BaseDataset
from benchopt import config


class Dataset(BaseDataset):

    # https://github.com/cohenjer/MM-nmf/tree/main/data_and_scripts

    name = "classic"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'estimated_rank' : [4]
    }

    def get_data(self):

        path = str(config.get_data_path("CLUTO/classic.mat"))

        with open(path) as f:
            n_docs, n_terms, n_nnz = map(int, f.readline().split())
            
            data = np.zeros(n_nnz)
            i = np.zeros(n_nnz)
            j = np.zeros(n_nnz)

            k=0

            for doc_id, line in enumerate(f):
                entries = list(map(int, line.split()))
                for p in range(0, len(entries), 2):
                    term_id = entries[p] - 1      # 1-based → 0-based
                    value = entries[p + 1]
                    data[k]=value
                    i[k]=doc_id
                    j[k]=term_id

                    k+=1
        
        V = coo_array((data, (i, j)), shape=(n_docs, n_terms))

        return dict(X=V, rank=self.estimated_rank, true_factors=None)