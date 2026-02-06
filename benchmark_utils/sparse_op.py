import scipy.sparse as sp
from scipy.sparse import coo_array,csr_array,csc_array
import numpy as np

def VoverWH(V,W,H,type="coo"):
    i,j = V.nonzero()
    WHdata = np.einsum('ik,ik->i', W[i], H.T[j])
    eps=np.finfo(float).eps
    Qdata = V.data/(WHdata+eps)
    if type=="coo":
        Q = coo_array((Qdata,(i,j)),shape=V.shape)
    elif type=="csr":
        Q = csr_array((Qdata,(i,j)),shape=V.shape)
    else:
        Q = csc_array((Qdata,(i,j)),shape=V.shape)
    return Q

def VoverWH2(V,W,H,type="coo"):
    i,j = V.nonzero()
    WHdata = np.einsum('ik,ik->i', W[i], H.T[j])
    eps=np.finfo(float).eps
    Qdata = V.data/(WHdata**2+eps)
    if type=="coo":
        Q = coo_array((Qdata,(i,j)),shape=V.shape)
    elif type=="csr":
        Q = csr_array((Qdata,(i,j)),shape=V.shape)
    else:
        Q = csc_array((Qdata,(i,j)),shape=V.shape)
    return Q

def remove_zero_columns_coo(X: sp.coo_array) -> sp.coo_array:
    if not isinstance(X, sp.coo_array):
        raise TypeError("Input must be a scipy.sparse.coo_array")

    N, M = X.shape
    if X.nnz == 0:
        # all columns are zero: return empty with 0 columns
        return sp.coo_array((N, 0))

    # find columns that actually appear in the data
    used_cols = np.unique(X.col)

    # map old column indices to new compact indices
    col_map = np.zeros(M, dtype=int) - 1
    col_map[used_cols] = np.arange(len(used_cols))

    new_col = col_map[X.col]

    return sp.coo_array(
        (X.data, (X.row, new_col)),
        shape=(N, len(used_cols))
    )