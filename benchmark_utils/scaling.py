import scipy.sparse as sp
# from scipy.sparse import coo_array,csr_array,csc_array
import numpy as np

def sinkhorn(V,W,H,it=10):
    _,R = W.shape
    N,M = V.shape

    u = np.ones(N)
    v = np.ones(M)

    a = V@v
    b = V.T@u

    if sp.issparse(V):

        for i in range(it):
            Hv = H@v
            u = a/(W@Hv)
            WTu = W.T@u
            v = b/(H.T@WTu)

    else:

        K = W @ H

        for i in range(it):
            u = a/(K@v)
            v = b/(K.T@u)

    W = W*u[:,np.newaxis]
    H = H*v

    return W,H