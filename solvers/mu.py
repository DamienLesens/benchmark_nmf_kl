from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion
import scipy.sparse as sp
from benchmark_utils.sparse_op import VoverWH
from benchmark_utils.scaling import sinkhorn

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Multiplicative Updates
    """
    name = "mu"

    parameters = {
        'n_inner_iter': [1],
        'sinkhorn_init': [True],
        'sinkhorn_freq': [None]
    }

    sampling_strategy = "callback"

    stopping_criterion = NoCriterion()#SufficientProgressCriterion(strategy="callback", key_to_monitor="objective_kullback-leibler")

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
        n_inner_iter = self.n_inner_iter

        eps=np.finfo(float).eps

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(m, rank), np.random.rand(rank, n)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]
        
        if self.sinkhorn_init:
            self.W, self.H = sinkhorn(self.X,self.W,self.H)

        it=0

        while callback():
            # W update
            oneHT = np.tile(np.sum(self.H,axis=1),(m,1))
            for _ in range(n_inner_iter):
                
                if sp.issparse(self.X):
                    Q = VoverWH(self.X,self.W,self.H,'csr')
                    self.W = self.W * (Q @ self.H.T)/(oneHT+eps)
                else:
                    self.W = self.W * ((self.X/(self.W@self.H+eps))@self.H.T)/(oneHT+eps)

            # H update
            WT1 = np.tile(np.sum(self.W,axis=0),(n,1)).T
            for _ in range(n_inner_iter):
                
                if sp.issparse(self.X):
                    Q = VoverWH(self.X,self.W,self.H,'csc')
                    self.H = self.H * (self.W.T @ Q)/(WT1+eps)
                else:
                    self.H = self.H * (self.W.T @ (self.X/(self.W@self.H+eps)))/(WT1+eps)
            
            it+=1
            if self.sinkhorn_freq is not None and it%self.sinkhorn_freq==0:
                self.W, self.H = sinkhorn(self.X,self.W,self.H)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)