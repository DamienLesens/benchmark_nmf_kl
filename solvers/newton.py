from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion
import scipy.sparse as sp
# from benchmark_utils.sparse_op import VoverWH,VoverWH2
from benchmark_utils.scaling import sinkhorn
import torch

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.special import kl_div


class Solver(BaseSolver):
    """
    "An Efficient Newton Algorithm for Nonnegative Matrix Factorization with the Kullback-Leibler Divergence"
      by Damien Lesens, Jérémy E. Cohen and Bora Uçar
"
    """
    name = "newton"

    parameters = {
        'n_inner_iter': [1],
        'iter_HALS': [5],
        'sinkhorn_init': [True],
        'sinkhorn_freq': [None],
        'descent': [True, False],
        'sigma': [0.2]
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

    def compute_hessians(self,W,VoverWHs):
        """
        Computes the tensor of all hessians
        """
        N,M = VoverWHs.shape
        
        T = self.WWT @ VoverWHs #shape (R*R,M)
        T = T.reshape(self.rank,self.rank,M)
        return T

    def update_HALS_H(self,V,W,H,MfH=None): 
        R,_ = H.shape
        eps=np.finfo(float).eps

        N,M = V.shape
        Hwork = H.copy()

        if sp.issparse(V):
            i = V.row
            j = V.col
            WHdata = np.einsum('ik,ik->i', W[i], H.T[j])
            VoverWH2 = sp.csr_array((V.data/(WHdata**2),(i,j)),shape=V.shape)
            T = self.WWT @ VoverWH2
            T = T.reshape(self.rank,self.rank,M)
            VoverWH = sp.csr_array((V.data/WHdata,(i,j)),shape=V.shape)
            G = (2*(W.T@VoverWH)-self.sum_W)
        else:
            WH = W@H
            B2 = V/(WH**2)
            G = (W.T@(2*V/(WH))-self.sum_W)
            T = self.compute_hessians(W,B2)

        for it in range(self.iter_HALS):
             
            for a in range(R):

                deltaH = np.maximum((G[a,:]- (T[a,:,:] * Hwork).sum(axis=0))/T[a,a,:], eps-Hwork[a,:]) 
                #deltaH can be used to decide early exit, or backtracking

                Hwork[a,:] += deltaH
                
        deltaHtot = Hwork-H

        if self.descent:
            lambdak = MfH*np.sqrt(np.einsum('aj,abj,bj->j',deltaHtot,T,deltaHtot))
            alphak = 1/(1+lambdak)
            dampedstep = lambdak>self.sigma
            fullstep = np.logical_and(np.logical_not(dampedstep),lambdak>eps)
            nostep = lambdak<=1e-3
            alphak[fullstep]=1
            alphak[nostep]=0
            Hwork = H + alphak*deltaHtot

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

        if self.sinkhorn_init:
            self.W, self.H = sinkhorn(self.X,self.W,self.H)

        if self.descent:
            if sp.issparse(self.X):
                vals = 1.0 / np.sqrt(self.X.data)
                MfH = np.zeros(self.X.shape[1])
                np.maximum.at(MfH, self.X.col, vals)
                MfW = np.zeros(self.X.shape[0])
                np.maximum.at(MfW, self.X.row, vals)
            else:
                mask = self.X > 0
                inv_sqrt = np.zeros_like(self.X)
                inv_sqrt[mask] = 1.0 / np.sqrt(self.X[mask])
                MfW = np.max(inv_sqrt, axis=1)
                MfH = np.max(inv_sqrt, axis=0)
        else:
            MfH = None
            MfW = None

        eps = np.finfo(float).eps

        it=0

        while callback():
            
            #update H
            #precomputing
            self.sum_W = np.sum(self.W, axis = 0)[:, None]
            self.WWT = ((self.W[:,:,None]*self.W[:,None,:]).reshape(N,-1)).T#shape (R*R,N)
            
            for _ in range(D):
                self.H = self.update_HALS_H(self.X,self.W,self.H,MfH)

            #update W
            #precomputing
            self.sum_W = np.sum(self.H.T, axis = 0)[:, None]
            self.WWT = (((self.H.T[:,:,None])*(self.H.T[:,None,:])).reshape(M,-1)).T
            
            for _ in range(D):
                self.W = self.update_HALS_H(self.X.T,self.H.T,self.W.T,MfW).T

            it+=1
            if self.sinkhorn_freq is not None and it%self.sinkhorn_freq==0:
                self.W, self.H = sinkhorn(self.X,self.W,self.H)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)
