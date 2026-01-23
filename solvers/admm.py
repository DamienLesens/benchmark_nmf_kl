from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    ADMM
    """
    name = "admm"

    parameters = {
        'rho': [1,10,1000,10000]
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

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(N, R), np.random.rand(R, M)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]

        #not sure about this init
        Y = self.W.copy()
        Z = self.H.copy()

        #idk how to init these
        X = np.zeros((N,M)) 
        aX = np.zeros((N,M))
        aY = np.zeros((N,R))
        aZ = np.zeros((R,M))

        while callback():

            Y = (np.linalg.inv(Z@Z.T+np.eye(R))@(Z@X.T+self.W.T+(Z@aX.T-aY.T)/self.rho)).T
            Z = np.linalg.inv(Y.T@Y+np.eye(R))@(Y.T@X+self.H+(Y.T@aX-aZ)/self.rho)

            YZ = Y@Z

            b = self.rho*YZ-aX-np.ones((N,M))
            X = (b+np.sqrt((b)**2+4*self.rho*self.X))/(2*self.rho)

            W = np.maximum(Y+aY/self.rho,0)
            H = np.maximum(Z+aZ/self.rho,0)

            aX = aX+self.rho*(X-YZ)
            aY = aY+self.rho*(Y-W)
            aZ = aZ+self.rho*(Z-H)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)