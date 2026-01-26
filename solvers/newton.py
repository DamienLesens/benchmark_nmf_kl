from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Multiplicative Updates with Burg entropy
    """
    name = "newton"

    parameters = {
        'n_inner_iter': [5],
        'solving': ['hals'],
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
    def update_HALS_H(V,W,H,I=50):#works but the dependance in R^2 is felt
        R,_ = H.shape
        eps=np.finfo(float).eps

        Hwork = H.copy()

        WH = W@H

        B2 = V/(WH**2+eps)
        G = W.T@(2*V/(WH+eps)-np.ones(V.shape))
        D = (W**2).T @B2
        T = np.einsum("ia,ij,ik->ajk",W,B2,W)

        for it in range(I):
            for a in range(R):
                num = G[a,:]- (T[a,:,:] * Hwork.T).sum(axis=1)+T[a,:,a]*Hwork[a,:]
                den = D[a,:]
                Hwork[a,:] = np.maximum(num/(den+eps),eps)
            

        return Hwork

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
                self.H = self.update_HALS_H(self.X,self.W,self.H,I=self.iter_HALS)

            #update W
            for _ in range(D):
                self.W = self.update_HALS_H(self.X.T,self.H.T,self.W.T,I=self.iter_HALS).T

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)