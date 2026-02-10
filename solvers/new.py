from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion
import scipy.sparse as sp
from benchmark_utils.sparse_op import VoverWH
from benchmark_utils.scaling import sinkhorn

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.special import kl_div


class Solver(BaseSolver):
    """
    Multiplicative Updates
    """
    name = "new"

    parameters = {
        'n_inner_iter': [1],
        'sinkhorn_init': [True],
        'sinkhorn_freq': [None],
        'beta': [0.5]
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
        m, n = self.X.shape
        rank = self.rank
        n_inner_iter = self.n_inner_iter

        eps=np.finfo(float).eps

        if not self.factors_init:
            # Random init if init is not provided
            self.W, self.H = [np.random.rand(m, rank), np.random.rand(rank, n)]
        else:
            self.W, self.H = [np.copy(self.factors_init[i]) for i in range(2)]
        
        if self.sinkhorn_init:
            self.W, self.H = sinkhorn(self.X,self.W,self.H)

        it=0

        while callback():
            # W update

            oneHT = np.sum(self.H,axis=1)
            oneoneHT = np.tile(oneHT,(m,1))
            oneHT2 = np.sum(oneHT*oneHT)
            for _ in range(n_inner_iter):
                
                if sp.issparse(self.X):
                    Q = VoverWH(self.X,self.W,self.H,'csr')
                    self.W = self.W * (Q @ self.H.T)/(oneHT+eps)
                else:
                    # self.W = self.W * ((self.X/(self.W@self.H+eps))@self.H.T)/(oneHT+eps)
                    WH = self.W@self.H
                    B = (self.X/(WH+eps))@self.H.T
                    num = B@oneHT
                    Delta = B - num[:,None]*oneoneHT/oneHT2
                    A = (self.W - eps) / (-Delta)
                    A_masked = np.where(Delta <= 0, A, np.inf)
                    gamma = np.min(A_masked, axis=1)
                    # gamma = np.min(((self.W-eps)/(-Delta))[Delta<=0],axis=1)
                    #backtracking
                    active = np.ones(m,dtype=bool) #keeping track of active columns
                    rhs = np.sum(kl_div(self.X,WH),axis=1)
                    count=0
                    while np.any(active) and count < 10: #or do a fix nb of iter and set to 0 if too much iter
                        W_trial = self.W[active,:]+ gamma[active,None]*Delta[active,:]
                        f_trial = np.sum(kl_div(self.X[active,:],W_trial@self.H),axis=1)
                        rhs_trial = rhs[active]
                        ok = f_trial <= rhs_trial
                        active_indices = np.where(active)[0]
                        active[active_indices[ok]]=False

                        gamma[active] *= self.beta
                        count+=1

                    gamma[active]=0

                    # gamma/=2

                    Wmu = self.W * ((self.X/(WH+eps))@self.H.T)/(oneoneHT+eps)
                    Wnew = self.W + gamma[:,None]*Delta

                    lossmu = np.sum(kl_div(self.X,Wmu@self.H),axis=1)
                    lossnew = np.sum(kl_div(self.X,Wnew@self.H),axis=1)

                    mubetter = lossmu<=lossnew

                    # print(np.sum(mubetter)/m)

                    self.W = Wnew
                    self.W[mubetter,:] = Wmu[mubetter,:]

            # H update
            WT1 = np.sum(self.W,axis=0)
            WT11 = np.tile(WT1,(n,1)).T
            WT12 = np.sum(WT1*WT1)
            for _ in range(n_inner_iter):
                
                if sp.issparse(self.X):
                    Q = VoverWH(self.X,self.W,self.H,'csc')
                    self.H = self.H * (self.W.T @ Q)/(WT1+eps)
                else:
                    WH = self.W@self.H
                    B = self.W.T @ (self.X/(WH+eps))
                    num = WT1.dot(B)
                    Delta = B - WT11*num/WT12
                    A = (self.H - eps) / (-Delta)
                    A_masked = np.where(Delta <= 0, A, np.inf)
                    gamma = np.min(A_masked, axis=0)
                    # gamma = np.min(((self.H-eps)/(-Delta))[Delta<=0],axis=0)
                    #backtracking
                    active = np.ones(n,dtype=bool) #keeping track of active columns
                    rhs = np.sum(kl_div(self.X,WH),axis=0)
                    count=0
                    while np.any(active) and count<10: #or do a fix nb of iter and set to 0 if too much iter
                        H_trial = self.H[:,active]+ Delta[:,active]*gamma[active]
                        f_trial = np.sum(kl_div(self.X[:,active],self.W@H_trial),axis=0)
                        rhs_trial = rhs[active]
                        ok = f_trial <= rhs_trial
                        active_indices = np.where(active)[0]
                        active[active_indices[ok]]=False

                        gamma[active] *= self.beta
                        count+=1
                    
                    gamma[active]=0

                    # gamma/=2

                    Hmu = self.H * (self.W.T @ (self.X/(WH+eps)))/(WT11+eps)
                    Hnew = self.H + Delta*gamma

                    lossmu = np.sum(kl_div(self.X,self.W@Hmu),axis=0)
                    lossnew = np.sum(kl_div(self.X,self.W@Hnew),axis=0)

                    mubetter = lossmu<=lossnew

                    # print(np.sum(mubetter)/n)

                    self.H = Hnew
                    self.H[:,mubetter] = Hmu[:,mubetter]


            
            it+=1
            if self.sinkhorn_freq is not None and it%self.sinkhorn_freq==0:
                self.W, self.H = sinkhorn(self.X,self.W,self.H)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)