from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion
import scipy.sparse as sp
from benchmark_utils.sparse_op import VoverWH
from benchmark_utils.scaling import sinkhorn

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Multiplicative Updates with Burg entropy
    """
    name = "mu_berg"

    parameters = {
        'n_inner_iter': [1],
        'sinkhorn_init': [True]
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
        
        linegammaH = 1/(self.X.sum(axis=0)*2) #size n
        colgammaW = 1/(self.X.sum(axis=1)*2) #size m
        gammaH =  np.tile(linegammaH,(rank,1))#1/(np.ones((rank,m))@self.X*2)
        gammaW = np.tile(colgammaW,(rank,1)).T#1/(np.ones((rank,n))@self.X.T*2)

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(m, rank), np.random.rand(rank, n)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]
        
        if self.sinkhorn_init:
            self.W, self.H = sinkhorn(self.X,self.W,self.H)

        while callback():
            # W update
            oneHT = np.tile(np.sum(self.H,axis=1),(m,1))
            for _ in range(n_inner_iter):
                if sp.issparse(self.X):
                    Q = VoverWH(self.X,self.W,self.H,'csr')
                    self.W = self.W / (np.ones((m,rank)) + gammaW * self.W * (oneHT - Q @ self.H.T))
                else:
                    self.W = self.W / (np.ones((m,rank)) + gammaW * self.W * (oneHT - (self.X/(self.W@self.H+eps))@self.H.T))

            # H update
            WT1 = np.tile(np.sum(self.W,axis=0),(n,1)).T
            for _ in range(n_inner_iter):
                if sp.issparse(self.X):
                    Q = VoverWH(self.X,self.W,self.H,'csc')
                    self.H = self.H / (np.ones((rank,n)) + gammaH * self.H * (WT1 - self.W.T @ Q))
                else:
                    self.H = self.H / (np.ones((rank,n)) + gammaH * self.H * (WT1 - self.W.T @(self.X/(self.W@self.H+eps))))

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)