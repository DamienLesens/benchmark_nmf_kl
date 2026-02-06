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