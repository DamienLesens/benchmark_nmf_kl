from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion
from benchmark_utils.scaling import sinkhorn

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    ADMM, from "Alternating direction method of multipliers for non-negative matrix factorization with the beta-divergence" 
    by Sun, Dennis L. and Fevotte, Cedric
    """
    name = "ADMM"

    parameters = {
        'rho': [1,10,1000,10000],
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

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(N, R), np.random.rand(R, M)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]

        if self.sinkhorn_init:
            self.W, self.H = sinkhorn(self.X,self.W,self.H)

        # Y is W, self.W is W+
        # Z is H, self.H is H+

        Y = self.W.copy()
        Z = self.H.copy()

        X = self.W@self.H
        aX = np.zeros((N,M))
        aY = np.zeros((N,R))
        aZ = np.zeros((R,M))

        it=0

        while callback():

            lhs = Z@Z.T+np.eye(R)
            rhs = Z@X.T+self.W.T+(Z@aX.T-aY.T)/self.rho
            Y = np.linalg.solve(lhs,rhs).T

            lhs = Y.T@Y+np.eye(R)
            rhs = Y.T@X+self.H+(Y.T@aX-aZ)/self.rho
            Z = np.linalg.solve(lhs,rhs)

            YZ = Y@Z

            b = self.rho*YZ-aX-np.ones((N,M))
            X = (b+np.sqrt((b)**2+4*self.rho*self.X))/(2*self.rho)

            self.W = np.maximum(Y+aY/self.rho,0)
            self.H = np.maximum(Z+aZ/self.rho,0)

            aX = aX+self.rho*(X-YZ)
            aY = aY+self.rho*(Y-self.W)
            aZ = aZ+self.rho*(Z-self.H)

            it+=1
            if self.sinkhorn_freq is not None and it%self.sinkhorn_freq==0:
                self.W, self.H = sinkhorn(self.X,self.W,self.H)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)