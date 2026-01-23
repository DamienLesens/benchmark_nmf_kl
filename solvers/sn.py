from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion
from benchmark_utils.sn import SNcpp

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Scalar Newton
    """
    name = "sn"

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
        maxtime = 1000#just something big
        V = np.asarray(self.X, dtype=np.float64, order="F")
        Wt = np.array(self.W.T, dtype=np.float64, order="F", copy=True)
        H  = np.array(self.H, dtype=np.float64, order="F", copy=True)
        objlist = np.zeros(maxiter)
        timelist = np.zeros(maxiter)
        inneriter=self.n_inner_iter
        delta=0.5#idk
        obj_compute=0 #indicates that the objective must be evaluated at each iteration


        reallength = SNcpp.run(
            m, n, k, maxiter, maxtime,
            V, Wt, H,
            objlist, timelist,
            inneriter,delta,obj_compute
        )

        #reallenght is equal to the actual number of iteration perform until maxtime is reached

        self.W = Wt.T
        self.H = H
        

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)