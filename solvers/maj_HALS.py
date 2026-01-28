from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Second order partial majorisation minimized with HALS
    """
    name = "maj_HALS"

    parameters = {
        'n_inner_iter': [5],
        'iter_HALS': [10,20,50]
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

    @staticmethod
    def update_maj_HALS_H(V,W,H0,I=50):
        H = H0.copy()
        R,_ = H.shape
        eps=np.finfo(float).eps

        WtW = W.T@W
        
        WH = W@H
        B2 = V/(WH**2+eps)
        beta = np.max(B2,axis=0)
        Y = WH + (V/(WH+eps)-np.ones(V.shape))/beta
        WtY = W.T@Y
        for it in range(I):
            for k in range(R): 
                num = WtY[k, :] - np.dot(WtW[k, :], H) + WtW[k, k] * H[k, :] 
                den = WtW[k, k] 
                H[k,:] = np.maximum(num / (den+eps),0)

        return H

    def run(self, callback):
        N, M = self.X.shape
        R = self.rank
        D = self.n_inner_iter

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(N, R), np.random.rand(R, M)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]

        eps = np.finfo(float).eps

        while callback():

            #update H
            for _ in range(D):
                self.H = self.update_maj_HALS_H(self.X,self.W,self.H,I=self.iter_HALS)

            #update W
            for _ in range(D):
                self.W = self.update_maj_HALS_H(self.X.T,self.H.T,self.W.T,I=self.iter_HALS).T

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)