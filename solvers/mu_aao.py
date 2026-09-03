from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion
import scipy.sparse as sp
from benchmark_utils.sparse_op import VoverWH
from benchmark_utils.scaling import sinkhorn

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Multiplicative Updates all at Once from "Kullback-Leibler Principal Component for  Tensors is not NP-hard" by Kejun Huang and Nicholas D. Sidiropoulos
    """
    name = "mu_aao"

    parameters = {
        'sinkhorn_init': [True],
        'sinkhorn_freq': [None],
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
        m, n = self.X.shape
        rank = self.rank

        eps=np.finfo(float).eps

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(m, rank), np.random.rand(rank, n)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]
        
        if self.sinkhorn_init:
            self.W, self.H = sinkhorn(self.X,self.W,self.H)

        # normalizing columns of W
        sumW = np.sum(self.W,axis=0)
        self.W = self.W/sumW
        self.H = self.H*sumW[:,None]

        it=0

        while callback():

            if sp.issparse(self.X):
                i = self.X.row
                j = self.X.col
                WHdata = np.einsum('ik,ik->i', self.W[i], self.H.T[j])
                VoverWH = sp.csr_array((self.X.data/WHdata,(i,j)),shape=self.X.shape)
            else:
                VoverWH = (self.X/(self.W@self.H+eps))


            newH = self.H*(self.W.T @ VoverWH)
            newW = self.W*(VoverWH @ self.H.T)
            newW = newW/np.sum(newW,axis=0)

            self.H = newH
            self.W = newW 

            
            it+=1
            if self.sinkhorn_freq is not None and it%self.sinkhorn_freq==0:
                self.W, self.H = sinkhorn(self.X,self.W,self.H)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)