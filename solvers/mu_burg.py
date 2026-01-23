from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Multiplicative Updates with Burg entropy
    """
    name = "mu_berg"

    parameters = {
        'n_inner_iter': [1],
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
        m, n = self.X.shape
        rank = self.rank
        n_inner_iter = self.n_inner_iter

        gammaH = 1/(np.ones((rank,m))@self.X*2)
        gammaW = 1/(np.ones((rank,n))@self.X.T*2)

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(m, rank), np.random.rand(rank, n)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]

        while callback():
            # W update
            for _ in range(n_inner_iter):
                self.W = self.updateH_MU_burg(self.X.T,self.H.T,self.W.T,gammaW).T

            # H update
            for _ in range(n_inner_iter):
                self.H = self.updateH_MU_burg(self.X,self.W,self.H,gammaH)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)