import scipy.sparse as sp
from scipy.sparse import coo_array
import numpy as np

def VoverWH(V,W,H):
    i,j = V.nonzero()
    WHdata = np.einsum('ik,ik->i', W[i], H.T[j])
    eps=np.finfo(float).eps
    Qdata = V.data/(WHdata+eps)
    Q = coo_array((Qdata,(i,j)),shape=V.shape)
    return Q

def VoverWH2(V,W,H):
    i,j = V.nonzero()
    WHdata = np.einsum('ik,ik->i', W[i], H.T[j])
    eps=np.finfo(float).eps
    Qdata = V.data/(WHdata**2+eps)
    Q = coo_array((Qdata,(i,j)),shape=V.shape)
    return Q