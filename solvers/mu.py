from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion
import scipy.sparse as sp
from benchmark_utils.sparse_op import VoverWH
from benchmark_utils.scaling import sinkhorn

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Multiplicative Updates from "Algorithms for non-negative matrix factorization" by Lee, Daniel and Seung, H Sebastian
    Extrapolated version follows "Block Majorization Minimization with Extrapolation and Application to β-NMF" by Le Thi Khanh Hien, Valentin Leplat, and Nicolas Gillis
    """
    name = "MU"

    parameters = {
        'n_inner_iter': [1],
        'sinkhorn_init': [True],
        'sinkhorn_freq': [None],
        'extrapolated': [True,False] 
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

        t1=1
        H_prev = self.H
        W_prev = self.W

        while callback():

            if self.extrapolated:
                t2=1/2*(1+np.sqrt(1+4*t1*t1))
                ex_coef=(t1-1)/t2
                t1=t2

                W_bar=np.maximum(self.W-W_prev,0)  
                W_prev = self.W                                              
                self.W = self.W + ex_coef*W_bar
                
                
            
            # W update
            oneHT = np.tile(np.sum(self.H,axis=1),(m,1))
            for _ in range(n_inner_iter):
                
                if sp.issparse(self.X):
                    Q = VoverWH(self.X,self.W,self.H,'csr')
                    VoverWHHT = Q @ self.H.T
                else:
                    VoverWHHT = (self.X/(self.W@self.H+eps))@self.H.T
                
                self.W = self.W * VoverWHHT /(oneHT+eps)

            if self.extrapolated:
                H_bar=np.maximum(self.H-H_prev,0)   
                H_prev = self.H   
                self.H = self.H + ex_coef*H_bar

            # H update
            WT1 = np.tile(np.sum(self.W,axis=0),(n,1)).T
            for _ in range(n_inner_iter):
                
                if sp.issparse(self.X):
                    Q = VoverWH(self.X,self.W,self.H,'csc')
                    WTVoverWH = self.W.T @ Q
                    
                else:
                    WTVoverWH = self.W.T @ (self.X/(self.W@self.H+eps))

                self.H = self.H * WTVoverWH/ (WT1 + eps)
            
            it+=1
            if self.sinkhorn_freq is not None and it%self.sinkhorn_freq==0:
                self.W, self.H = sinkhorn(self.X,self.W,self.H)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)