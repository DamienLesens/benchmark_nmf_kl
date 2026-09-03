from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion
import scipy.sparse as sp
from benchmark_utils.sparse_op import VoverWH,VoverWH2
from benchmark_utils.scaling import sinkhorn

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Scalar Newton algorithms in Python from
    - "Algorithms for Nonnegative Matrix Factorization with the Kullback-Leibler Divergence"
    by Hien, L. T. K. and Gillis, N.
    - "Fast coordinate descent methods with variable selection for non-negative matrix factorization"
    by Hsieh, Cho-Jui and Dhillon, Inderjit S
    """
    name = "scalar_newton"

    parameters = {
        'n_inner_iter': [5],
        'method': ['SN','CCD'],
        'sinkhorn_init': [True],
        'sinkhorn_freq': [None]
    }

    sampling_strategy = "callback"

    stopping_criterion = NoCriterion()

    def set_objective(self, X, rank, factors_init):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.X = X
        self.rank = rank
        self.factors_init = factors_init  # None if not initialized beforehand

    def run(self, callback):
        N, M = self.X.shape
        R = self.rank
        D = self.n_inner_iter

        eps=np.finfo(float).eps

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(N, R), np.random.rand(R, M)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]
        
        if self.sinkhorn_init:
            self.W, self.H = sinkhorn(self.X,self.W,self.H)
        
        if sp.issparse(self.X):
            i,j = self.X.nonzero()
            WHdata = np.maximum(np.einsum('ik,ik->i', self.W[i], self.H.T[j]),eps)
        else:
            WH = np.maximum(self.W @ self.H,eps)
        
        #computing self-concordance constants
        if self.method == "SN":
            if sp.issparse(self.X):
                vals = 1.0 / np.sqrt(self.X.data)
                chj = np.zeros(self.X.shape[1])
                np.maximum.at(chj, self.X.col, vals)
                cwi = np.zeros(self.X.shape[0])
                np.maximum.at(cwi, self.X.row, vals)
            else:
                mask = self.X > 0
                inv_sqrt = np.zeros_like(self.X)
                inv_sqrt[mask] = 1.0 / np.sqrt(self.X[mask])
                cwi = np.max(inv_sqrt, axis=1)
                chj = np.max(inv_sqrt, axis=0)

        it=0

        while callback():

            #update W
            sum_H = np.sum(self.H, axis=1)

            for innerit in range(D):

                for q in range(R):

                    if sp.issparse(self.X):
                        XoWHdata = self.X.data / WHdata
                        tmp = XoWHdata * self.H[q, j]
                        grad = -np.bincount(i, weights=tmp, minlength=N) + sum_H[q]
                        tmp2 = (XoWHdata / WHdata) * (self.H[q, j]**2)
                        hess = np.bincount(i, weights=tmp2, minlength=N)
                    else:
                        XoWH = self.X / WH
                        grad = - XoWH.dot(self.H[q, :]) + sum_H[q]
                        hess = (XoWH/WH).dot(self.H[q, :]**2)
                        hess = np.maximum(hess, eps)
                    s = np.maximum(self.W[:, q] - grad/hess, eps)
                    if self.method == "SN":
                        # safe update
                        d = s - self.W[:, q]
                        lamb = cwi*np.sqrt(hess)*np.abs(d)
                        newcolW = np.maximum(np.where((grad <= 0) + (lamb <= 0.683802), s, self.W[:, q] + (1/(1+lamb)) * d),eps)
                    else:
                        newcolW = s

                    if sp.issparse(self.X):
                        WHdata += (newcolW-self.W[:,q])[i]*(self.H[q,:])[j]
                    else:
                        WH += np.outer(newcolW - self.W[:, q], self.H[q, :])
                    
                    self.W[:, q] = newcolW
                
            
            #update H
            sum_W = np.sum(self.W, axis=0)

            for innerit in range(D):
                
                for q in range(R):

                    if sp.issparse(self.X):
                        XoWHdata = self.X.data / WHdata
                        tmp = XoWHdata * self.W[i, q]
                        grad = -np.bincount(j, weights=tmp, minlength=M) + sum_W[q]
                        tmp2 = (XoWHdata  / WHdata) * (self.W[i, q]**2)
                        hess = np.bincount(j, weights=tmp2, minlength=M)
                    else:
                        XoWH = self.X / WH
                        grad = - (self.W[:, q]).dot(XoWH) + sum_W[q]
                        hess = ((self.W[:, q]**2)).dot(XoWH/WH)
                        hess = np.maximum(hess, eps)
                    s = np.maximum(self.H[q, :] - grad/hess, eps)
                    if self.method == "SN":
                        # safe update
                        d = s - self.H[q, :]
                        lamb = chj*np.sqrt(hess)*np.abs(d)
                        newlineH = np.maximum(np.where((grad <= 0) + (lamb <= 0.683802), s, self.H[q, :] + (1/(1+lamb)) * d),eps)
                    else:
                        newlineH = s

                    if sp.issparse(self.X):
                        WHdata += (self.W[:, q])[i]*(newlineH - self.H[q, :])[j]
                    else:
                        WH += np.outer(self.W[:, q], newlineH - self.H[q, :])
                    
                    self.H[q, :] = newlineH
                
            
            it+=1
            if self.sinkhorn_freq is not None and it%self.sinkhorn_freq==0:
                self.W, self.H = sinkhorn(self.X,self.W,self.H)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objectiself.Xe.
        # They are customizable.
        return dict(W=self.W, H=self.H)