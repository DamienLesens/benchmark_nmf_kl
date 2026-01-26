from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Multiplicative Updates with Burg entropy
    """
    name = "som"

    parameters = {
        'n_inner_iter': [10],
        'gamma': [1.9]
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
            self.W, self.H = [np.random.rand(R, N), np.random.rand(R, M)]
        else:
            self.W, self.H = [np.copy(self.factors_init[0]).T, np.copy(self.factors_init[1])]

        eps = np.finfo(float).eps

        while callback():

            #precomputing
            margWN = self.W @ np.ones(N) 
            margWR = np.ones(R) @ self.W
            productWR = self.W*(np.tile(margWR,(R,1)))
            productWNM = np.tile(margWN[:,None],(1,M))
            #update H
            for _ in range(D):
                WTH = self.W.T@self.H
                self.H = np.maximum(self.H+gamma*(self.W@(self.X/(WTH+eps))-productWNM)/(productWR@(self.X/(WTH**2+eps))),eps)

            #precomputing
            margHM = self.H @ np.ones(M)
            margHR = np.ones(R) @ self.H
            productHR = self.H*(np.tile(margHR,(R,1)))
            productHMN = np.tile(margHM[:,None],(1,N))
            #update W
            for _ in range(D):
                HTW = self.H.T@self.W
                self.W = np.maximum(self.W+gamma*(self.H@(self.X.T/(HTW+eps))-productHMN)/(productHR@(self.X.T/(HTW**2+eps))),eps)
                

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W.T, H=self.H)