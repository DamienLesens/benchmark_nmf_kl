from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    SN
    """
    name = "sn_python"

    parameters = {
        'n_inner_iter': [5],
        'delta': [0.2]
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
        delta = self.delta

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(N, R), np.random.rand(R, M)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]
        
        WH = self.W @ self.H

        eps=np.finfo(float).eps
        
        #computing self-concordance constants
        row = np.zeros(N)
        for i in range(N):
            for j in range(M):
                if self.X[i,j]>0:
                    row[i]=max(row[i],1/np.sqrt(self.X[i,j]))
        col = np.zeros(M)
        for j in range(M):
            for i in range(N):
                if self.X[i,j]>0:
                    col[j]=max(col[j],1/np.sqrt(self.X[i,j]))

        while callback():

            #update W
            for i in range(N):

                for innerit in range(D):
                    diff_tot = 0
                    for k in range(R):
                        tmp = self.X[i,:]/(WH[i,:]+eps)
                        g = np.dot(self.H[k,:],1-tmp)
                        h = np.dot(self.H[k,:]**2,tmp/(WH[i,:]+eps))
                        oldW = self.W[i,k]
                        newW = max(oldW-g/(h+eps),eps)
                        if g>0:
                            diff = newW-oldW
                            lambda_ = row[i]*np.sqrt(h)*np.abs(diff)
                            if (lambda_ > 0.683802): #damped Newton step
                                newW = max(oldW+1/(1+lambda_)*diff,eps)
                        
                        diff = newW-oldW
                        diff_tot += diff*diff
                        WH[i,:] = np.maximum(WH[i,:] + diff*self.H[k,:],eps)
                        
                        self.W[i,k]=newW

                    if innerit==0:
                        eps0=np.sqrt(diff_tot)
                        eps1=eps0
                    else:
                        eps1 = np.sqrt(diff_tot)
                    
                    if (eps1<delta*eps0):
                        break
            
            #update H
            for j in range(M):

                for innerit in range(D):
                    diff_tot = 0
                    for k in range(R):
                        tmp = self.X[:,j]/(WH[:,j]+eps)
                        g = np.dot(self.W[:,k],1-tmp)
                        h = np.dot(self.W[:,k]**2,tmp/(WH[:,j]+eps))
                        oldH = self.H[k,j]
                        newH = max(oldH-g/(h+eps),eps)
                        if g>0:
                            diff = newH-oldH
                            lambda_ = col[j]*np.sqrt(h)*np.abs(diff)
                            if (lambda_ > 0.683802): #damped Newton step
                                newH = max(oldH+1/(1+lambda_)*diff,eps)
                        
                        diff = newH-oldH
                        diff_tot += diff*diff
                        WH[:,j] = np.maximum(WH[:,j] + diff*self.W[:,k],eps)
                        
                        self.H[k,j]=newH

                    if innerit==0:
                        eps0=np.sqrt(diff_tot)
                        eps1=eps0
                    else:
                        eps1 = np.sqrt(diff_tot)
                    
                    if (eps1<delta*eps0):
                        break

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objectiself.Xe.
        # They are customizable.
        return dict(W=self.W, H=self.H)