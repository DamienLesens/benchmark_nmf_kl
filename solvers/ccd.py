from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion
from benchmark_utils.ccd import CCDcpp

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Cyclic Coordinate Descent
    """
    name = "ccd"

    parameters = {
        'n_inner_iter': [5]
    }

    sampling_strategy = "iteration"

    stopping_criterion = NoCriterion()#SufficientProgressCriterion(strategy="callback", key_to_monitor="objective_kullback-leibler")

    def set_objective(self, X, rank, factors_init):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.X = X
        self.rank = rank
        self.factors_init = factors_init  # None if not initialized beforehand
    
    def run(self, stop_val):
        n,m = self.X.shape
        k = self.rank

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(n, k), np.random.rand(k, m)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]

        maxiter = stop_val
        maxtime = 1000
        V = np.asarray(V, dtype=np.float64, order="F")
        W = np.asarray(self.W, dtype=np.float64, order="C", copy=True)
        H = np.asarray(self.H, dtype=np.float64, order="F", copy=True)
        objlist = np.zeros(maxiter)
        timelist = np.zeros(maxiter)

        #D is set to 2 by default inside the cpp function

        reallength = CCDcpp.run(
            n, m, k, maxiter, maxtime,
            V, W, H,
            0,
            objlist, timelist
        )

        self.W = W
        self.H = H
        

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)