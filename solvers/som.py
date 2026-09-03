from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion
from benchmark_utils.sparse_op import VoverWH,VoverWH2
from benchmark_utils.scaling import sinkhorn
import scipy.sparse as sp

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    AmSOM and AMUSOM from "A Second Order Majorant Algorithm for Nonnegative Matrix Factorization"
    by Pham, Mai-Quyen and Cohen, Jérémy E and Chonavel, Thierry
    """
    name = "SOM"

    parameters = {
        'n_inner_iter': [1],
        'gamma': [1.9],
        'method': ["AMUSOM","AmSOM"],
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
        gamma = self.gamma

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(N,R), np.random.rand(R, M)]
        else:
            self.W, self.H = [np.copy(self.factors_init[0]), np.copy(self.factors_init[1])]
        
        if self.sinkhorn_init:
            self.W, self.H = sinkhorn(self.X,self.W,self.H)

        eps = np.finfo(float).eps

        if not sp.issparse(self.X):
            WH = self.W.dot(self.H)

        it=0

        while callback():

            sum_H = np.sum(self.H, axis = 1)[None,:] 
            sum_H2 = np.sum(self.H, axis = 0)[None,:]
            HH2 = (self.H*sum_H2).T

            for iw in range(D): 
                
                if sp.issparse(self.X):
                    i = self.X.row
                    j = self.X.col
                    WHdata = np.einsum('ik,ik->i', self.W[i], self.H.T[j])
                    VoverWH = sp.csr_array((self.X.data/WHdata,(i,j)),shape=self.X.shape)
                    if self.method == "AMUSOM":
                        temp_grad = VoverWH@(self.H.T)
                        aux_W = gamma*self.W/sum_H
                        deltaW = np.maximum(aux_W*(temp_grad - sum_H), eps-self.W)
                    elif self.method == "AmSOM":
                        VoverWH2 = sp.csr_array((self.X.data/(WHdata**2),(i,j)),shape=self.X.shape)
                        aux_W = gamma*1/(VoverWH2@HH2)
                        deltaW = np.maximum(aux_W*(VoverWH@(self.H.T) - sum_H), eps-self.W)
                    self.W = self.W + deltaW
                else:
                    if self.method == "AMUSOM":
                        temp_grad = (self.X/WH).dot(self.H.T)
                        aux_W = gamma*self.W/sum_H
                        deltaW = np.maximum(aux_W*(temp_grad - sum_H), eps-self.W)
                    elif self.method == "AmSOM":
                        aux_W = gamma*1/((self.X/WH**2).dot(HH2))
                        deltaW = np.maximum(aux_W*((self.X/WH).dot(self.H.T) - sum_H), eps-self.W)
                    self.W = self.W + deltaW
                    WH = self.W.dot(self.H)
                

            sum_W = np.sum(self.W, axis = 0)[:, None]
            sum_W2= np.sum(self.W, axis = 1)[:, None]
            WW2 = (self.W*sum_W2).T
            
            for ih in range(D):
                if sp.issparse(self.X):
                    i = self.X.row
                    j = self.X.col
                    WHdata = np.einsum('ik,ik->i', self.W[i], self.H.T[j])
                    VoverWH = sp.csr_array((self.X.data/WHdata,(i,j)),shape=self.X.shape)
                    if self.method == "AMUSOM":
                        temp_grad = (self.W.T) @ VoverWH
                        aux_H = gamma*self.H/sum_W
                        deltaH = np.maximum(aux_H*(temp_grad - sum_W), eps-self.H)
                    elif self.method == "AmSOM":
                        VoverWH2 = sp.csr_array((self.X.data/(WHdata**2),(i,j)),shape=self.X.shape)
                        den = WW2 @ VoverWH2
                        aux_H = gamma / den
                        deltaH = np.maximum(aux_H*((self.W.T)@(VoverWH) - sum_W), eps-self.H)
                    self.H = self.H + deltaH
                else:
                    if self.method == "AMUSOM":
                        temp_grad = (self.W.T).dot(self.X/WH)
                        aux_H = gamma*self.H/sum_W
                        deltaH = np.maximum(aux_H*(temp_grad - sum_W), eps-self.H)
                    elif self.method == "AmSOM":
                        aux_H = gamma*1/(WW2.dot(self.X/(WH**2)))
                        deltaH = np.maximum(aux_H*((self.W.T).dot(self.X/WH) - sum_W), eps-self.H)
                    self.H = self.H + deltaH
                    WH = self.W.dot(self.H)
            
            it+=1
            if self.sinkhorn_freq is not None and it%self.sinkhorn_freq==0:
                self.W, self.H = sinkhorn(self.X,self.W,self.H)
                    

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)