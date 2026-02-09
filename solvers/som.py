from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion
from benchmark_utils.sparse_op import VoverWH,VoverWH2
from benchmark_utils.scaling import sinkhorn
import scipy.sparse as sp

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Multiplicative Updates with Burg entropy
    """
    name = "som"

    parameters = {
        'n_inner_iter': [10],
        'gamma': [1.9],
        'method': ["AMUSOM","AmSOM"],
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

            # Uses the true Hessian but only at first iteration of inner loop (should change but too costly)
            sum_H = np.sum(self.H, axis = 1)[None,:] 
            sum_H2 = np.sum(self.H, axis = 0)[None,:]
            HH2 = (self.H*sum_H2).T
            # inner_change_0 = 1
            # inner_change_l = np.inf
            for iw in range(D): 
                #if k==0:
                    # Lee Seung first iter
                #    deltaW =  np.maximum(W *(((V/WH).dot(H.T))/sum_H-1), epsilon-W)
                #else:
                if sp.issparse(self.X):
                    if self.method == "AMUSOM":
                        temp_grad = VoverWH(self.X,self.W,self.H,"csr",eps=0)@(self.H.T)
                        aux_W = gamma*self.W/temp_grad
                        deltaW = np.maximum(aux_W*(temp_grad - sum_H), eps-self.W)
                    elif self.method == "AmSOM":
                        aux_W = gamma*1/(VoverWH2(self.X,self.W,self.H,"csr",eps=0)@HH2)
                        deltaW = np.maximum(aux_W*(VoverWH(self.X,self.W,self.H,"csr",eps=0)@(self.H.T) - sum_H), eps-self.W)
                    self.W = self.W + deltaW
                else:
                    if self.method == "AMUSOM":
                        temp_grad = (self.X/WH).dot(self.H.T)
                        aux_W = gamma*self.W/temp_grad
                        deltaW = np.maximum(aux_W*(temp_grad - sum_H), eps-self.W)
                    elif self.method == "AmSOM":
                        aux_W = gamma*1/((self.X/WH**2).dot(HH2))
                        deltaW = np.maximum(aux_W*((self.X/WH).dot(self.H.T) - sum_H), eps-self.W)
                    self.W = self.W + deltaW
                    WH = self.W.dot(self.H)
                # if k>0: # no early stopping the first iteration, default is no dynamic stopping
                #     if iw==0:
                #         inner_change_0 = np.linalg.norm(deltaW)**2
                #     else:
                #         inner_change_l = np.linalg.norm(deltaW)**2
                #     if inner_change_l < self.delta*inner_change_0:
                #         break
                
            # FIXED W ESTIMATE H  
            sum_W = np.sum(self.W, axis = 0)[:, None]
            sum_W2= np.sum(self.W, axis = 1)[:, None]
            WW2 = (self.W*sum_W2).T
            # inner_change_0 = 1
            # inner_change_l = np.inf
            for ih in range(D):
                if sp.issparse(self.X):
                    if self.method == "AMUSOM":
                        temp_grad = (self.W.T) @ VoverWH(self.X, self.W, self.H, "csr",eps=0)
                        aux_H = gamma*self.H/temp_grad
                        deltaH = np.maximum(aux_H*(temp_grad - sum_W), eps-self.H)
                    elif self.method == "AmSOM":
                        den = WW2 @ VoverWH2(self.X, self.W, self.H, "csr",eps=0)
                        aux_H = gamma / den
                        # aux_H = gamma*1/(WW2.dot(VoverWH2(self.X,self.W,self.H,"csr")))
                        deltaH = np.maximum(aux_H*((self.W.T)@(VoverWH(self.X,self.W,self.H,"csr",eps=0)) - sum_W), eps-self.H)
                    self.H = self.H + deltaH
                else:
                    if self.method == "AMUSOM":
                        temp_grad = (self.W.T).dot(self.X/WH)
                        aux_H = gamma*self.H/temp_grad
                        deltaH = np.maximum(aux_H*(temp_grad - sum_W), eps-self.H)
                    elif self.method == "AmSOM":
                        aux_H = gamma*1/(WW2.dot(self.X/(WH**2)))
                        deltaH = np.maximum(aux_H*((self.W.T).dot(self.X/WH) - sum_W), eps-self.H)
                    self.H = self.H + deltaH
                    WH = self.W.dot(self.H)
                # if k>0: # no early stopping the first iteration
                #     if ih==0:
                #         inner_change_0 = np.linalg.norm(deltaH)**2
                #     else:
                #         inner_change_l = np.linalg.norm(deltaH)**2
                #     if inner_change_l < delta*inner_change_0:
                #         break
            
            it+=1
            if self.sinkhorn_freq is not None and it%self.sinkhorn_freq==0:
                self.W, self.H = sinkhorn(self.X,self.W,self.H)
                    

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)