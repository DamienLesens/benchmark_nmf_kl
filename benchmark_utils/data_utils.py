import numpy as np

def sparsify(M, s=0.5, epsilon=1e-8):
    """Adds zeroes in matrix M in order to have a ratio s of nnzeroes/nnentries.

    Parameters
    ----------
    M : 2darray
        The input numpy array
    s : float, optional
        the sparsity ratio (0 for fully sparse, 1 for density of the original array), by default 0.5
    """    
    vecM = M.flatten()
    # use quantiles
    val = np.quantile(vecM, 1-s)
    # put zeros in M
    M[M<val]=epsilon
    return M

def generate_mask(n,m,density):
    """
    generates a mask, making sure that there is no 0 line or column
    
    :param n: Description
    :param m: Description
    :param density: Description
    """
    mask = np.random.rand(n, m) > density
    for i in range(n):
        if np.all(mask[i,:]):
            j = np.random.randint(0,m-1)
            mask[i,j]=False
    for j in range(m):
        if np.all(mask[:,j]):
            i = np.random.randint(0,n-1)
            mask[i,j]=False
    return mask