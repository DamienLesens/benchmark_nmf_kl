from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Multiplicative Updates with Burg entropy
    """
    name = "fpa"

    parameters = {
        'n_inner_iter': [5],
        'loss': ['divergence']
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
    def updateH_MU_burg(V,W,H,gamma):
    #gamma will be matrix, to have different steps on different columns
        eps=np.finfo(float).eps
        return H / (np.ones(H.shape) + gamma * H * (W.T @(np.ones(V.shape)- V/(W@H+np.full(V.shape,eps)))) + np.full(H.shape,eps))

    def run(self, callback):
        N, M = self.X.shape
        R = self.rank
        n_inner_iter = self.n_inner_iter

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(N, R), np.random.rand(R, M)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]
        
        WH = self.W @ self.H
        chi = WH#-V / WH

        # translating the mathlab formula: chi = chi * (1 ./ max( -W' * chi .* (1./sum(W,1)')))
        denom = -self.W.T @ chi                 # shape (r, m)
        denom = denom * (1.0 / np.sum(self.W, axis=0))[:, None]
        chi = chi * (1.0 / np.max(denom))

        Wbar = self.W.copy()
        Wold = self.W.copy()
        Hbar = self.H.copy()
        Hold = self.H.copy()
        

        # eps = np.finfo(float).eps

        while callback():
            
            # ======= Update H =======
            # this choice of parameters is motivated in the article
            sigma = np.sqrt(N / R) * np.sum(self.W) / np.sum(self.X, axis=0) / np.linalg.norm(self.W,ord=2)
            tau   = np.sqrt(R / N) * np.sum(self.X, axis=0) / np.sum(self.W) / np.linalg.norm(self.W,ord=2)

            for _ in range(n_inner_iter):
                WHbar = self.W @ Hbar                    # shape (n×m)

                chi = chi + WHbar * sigma
                chi = (chi - np.sqrt(chi**2 + self.X * (4 * sigma))) / 2

                self.H = np.maximum(self.H - (self.W.T @ (chi + 1)) * tau, 0)
                Hbar = 2 * self.H - Hold
                Hold = self.H.copy()

            # ======= Update W =======
            sigma = np.sqrt(M / R) * np.sum(self.H) / np.sum(self.X, axis=1)[:, None] / np.linalg.norm(self.H,ord=2)
            tau   = np.sqrt(R / M) * np.sum(self.X, axis=1)[:, None] / np.sum(self.H) / np.linalg.norm(self.H,ord=2)

            for _ in range(n_inner_iter):
                WbarH = Wbar @ self.H

                chi = chi + WbarH * sigma
                chi = (chi - np.sqrt(chi**2 + self.X * (4 * sigma))) / 2

                self.W = np.maximum(self.W - (chi + 1) @ self.H.T * tau, 0)
                Wbar = 2 * self.W - Wold
                Wold = self.W.copy()

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)