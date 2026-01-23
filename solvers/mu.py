from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Multiplicative Updates
    """
    name = "mu"

    parameters = {
        'n_inner_iter': [1],
        'loss': ['divergence']
    }

    stopping_criterion = SufficientProgressCriterion(
        strategy="callback", key_to_monitor="objective_kullback-leibler"
    )

    def set_objective(self, X, rank, factors_init):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.X = X
        self.rank = rank
        self.factors_init = factors_init  # None if not initialized beforehand
    
    def updateH_MU(V,W,H):
        eps=np.finfo(float).eps
        return H * (W.T @ (V/(W@H+np.full(V.shape,eps))))/(W.T @ np.ones(V.shape)+np.full(W.T.shape,eps))

    def run(self, callback):
        m, n = self.X.shape
        rank = self.rank
        n_inner_iter = self.n_inner_iter

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(m, rank), np.random.rand(rank, n)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]

        while callback():
            # W update
            for _ in range(n_inner_iter):
                self.W = self.updateH_MU(self.X.T,self.W,self.H).T

            # H update
            for _ in range(n_inner_iter):
                self.H = self.updateH_MU(self.X,self.W,self.H)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)