from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    CCD
    """
    name = "ccd_python"

    parameters = {
        'n_inner_iter': [2]
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
        

        while callback():

            #update W
            for i in range(N):
                for k in range(R):

                    for innerit in range(D):
                        tmp = self.X[i,:]/(WH[i,:]+eps)
                        g = np.dot(self.H[k,:],1-tmp)
                        h = np.dot(self.H[k,:]**2,tmp/(WH[i,:]+eps))
                        s = -g/h
                        oldW = self.W[i,k]
                        newW = max(oldW+s,eps)
                        diff = newW-oldW
                        self.W[i,k]=newW
                        WH[i,:] = np.maximum(WH[i,:] + diff*self.H[k,:],eps)
                        
                        if abs(diff)<abs(oldW)*0.5:
                            break
            
            #update H
            for j in range(M):
                for k in range(R):

                    for innerit in range(D):
                        tmp = self.X[:,j]/(WH[:,j]+eps)
                        g = np.dot(self.W[:,k],1-tmp)
                        h = np.dot(self.W[:,k]**2,tmp/(WH[:,j]+eps))
                        s = -g/h
                        oldH = self.H[k,j]
                        newH = max(oldH+s,eps)
                        diff = newH-oldH
                        self.H[k,j]=newH
                        WH[:,j] = np.maximum(WH[:,j] + diff*self.W[:,k],eps)
                        
                        if abs(diff)<abs(oldH)*0.5:
                            break         
        

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objectiself.Xe.
        # They are customizable.
        return dict(W=self.W, H=self.H)