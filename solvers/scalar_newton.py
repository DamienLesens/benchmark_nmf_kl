from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    SN
    """
    name = "scalar_newton"

    parameters = {
        'n_inner_iter': [5],
        'method': ['SN','CCD']
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

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(N, R), np.random.rand(R, M)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]
        
        WH = self.W @ self.H

        eps=np.finfo(float).eps
        
        #computing self-concordance constants
        if self.method == "SN":
            chj = np.max((self.X > 0) / np.sqrt(self.X), axis=0)
            cwi = np.max((self.X > 0) / np.sqrt(self.X), axis=1)

        while callback():

            #update W
            sum_H = np.sum(self.H, axis=1)

            for innerit in range(D):

                Wnew = np.copy(self.W)

                for q in range(R):
                    grad = - (self.X/WH).dot(self.H[q, :]) + sum_H[q]
                    hess = (self.X/WH**2).dot(self.H[q, :]**2)  # .T)
                    s = np.maximum(self.W[:, q] - grad/hess, eps)
                    if self.method == "SN":
                        # safe update
                        d = s - self.W[:, q]
                        lamb = cwi*np.sqrt(hess)*np.abs(d)  # broadcasting check
                        Wnew[:, q] = np.where((grad <= 0) + (lamb <= 0.683802), s, self.W[:, q] + (1/(1+lamb)) * d)
                    else:
                        Wnew[:, q] = s

                    WH += np.outer(Wnew[:, q] - self.W[:, q], self.H[q, :])  # updated
                    WH = np.maximum(WH, eps)
                
                self.W = Wnew
            
            #update H
            sum_W = np.sum(self.W, axis=0)

            for innerit in range(D):

                Hnew = np.copy(self.H)
                
                for q in range(R):
                    grad = - (self.W[:, q]).dot(self.X/WH) + sum_W[q]
                    hess = ((self.W[:, q]**2)).dot(self.X/(WH**2))  # elementwise 2d order derivative
                    s = np.maximum(self.H[q, :] - grad/hess, eps)
                    if self.method == "SN":
                        # safe update
                        d = s - self.H[q, :]
                        lamb = chj*np.sqrt(hess)*np.abs(d)  # broadcasting check
                        Hnew[q, :] = np.where((grad <= 0) + (lamb <= 0.683802), s, self.H[q, :] + (1/(1+lamb)) * d)
                    else:
                        Hnew[q, :] = s

                    WH += np.outer(self.W[:, q], Hnew[q, :] - self.H[q, :])  # updated
                    WH = np.maximum(WH, eps)
                
                self.H = Hnew

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objectiself.Xe.
        # They are customizable.
        return dict(W=self.W, H=self.H)