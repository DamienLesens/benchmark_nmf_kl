import numpy as np
from scipy.sparse import coo_array

from benchopt import BaseDataset
from benchopt import config
from benchmark_utils.sparse_op import remove_zero_columns_coo

class Dataset(BaseDataset):

    # https://karypis.github.io/glaros/software/cluto/overview.html
    # https://karypis.github.io/glaros/files/sw/cluto/datasets.tar.gz

    name = "CLUTO"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'estimated_rank' : [None],
        'collection' : ["cacmcisi","classic","cranmed","fbis","hitech",
                      "k1a","k1b","la1","la2","la12","mm","new3","ohscal",
                      "re0","re1","reviews","sports","tr11","tr12","tr23",
                      "tr31","tr41","tr45","wap"],
        'sparse' : [True]
    }

    collection_to_rank = {
        'cacmcisi': 2,
        'classic': 4,
        'cranmed': 2,
        'fbis': 17,
        'hitech': 6,
        'k1a': 20,
        'k1b': 6,
        'la1': 6,
        'la2': 6,
        'la12': 6,
        'mm': 2,
        'new3': 44,
        'ohscal': 10,
        're0': 13,
        're1': 25,
        'reviews': 5,
        'sports': 7,
        'tr11': 9,
        'tr12': 8,
        'tr23': 6,
        'tr31': 7,
        'tr41': 10,
        'tr45': 10,
        'wap': 20
    }

    def get_data(self):

        path = str(config.get_data_path("CLUTO/"+self.collection+".mat"))

        with open(path) as f:
            n_docs, n_terms, n_nnz = map(int, f.readline().split())
            
            data = np.zeros(n_nnz)
            i = np.zeros(n_nnz)
            j = np.zeros(n_nnz)

            k=0

            for doc_id, line in enumerate(f):
                entries = list(map(float, line.split()))
                for p in range(0, len(entries), 2):
                    term_id = int(entries[p]) - 1 # 1-based to 0-based
                    value = entries[p + 1]
                    data[k]=value
                    i[k]=doc_id
                    j[k]=term_id

                    k+=1
        
        V = coo_array((data, (i, j)), shape=(n_docs, n_terms))
        V = remove_zero_columns_coo(V)
        
        if not(self.sparse):
            V = V.toarray()
        
        return dict(X=V, rank=self.collection_to_rank[self.collection], true_factors=None)