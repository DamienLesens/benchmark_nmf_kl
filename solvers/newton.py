from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion,NoCriterion
import scipy.sparse as sp
from benchmark_utils.sparse_op import VoverWH,VoverWH2
from benchmark_utils.scaling import sinkhorn
import torch

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """
    Minimizing the second order Taylor expansion
    """
    name = "newton"

    parameters = {
        'n_inner_iter': [5],
        'solving': ['hals'],
        'iter_HALS': [10,20,50],
        'sinkhorn_init': [True],
        'sinkhorn_freq': [None],
        'method': ["full","random","einsum","svd"],
        'S': [None],
        'svd_tol': [0.9,0.95,0.99],
        'balancing': [False]
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

    def compute_hessians(self,W,VoverWHs):
        """
        options are:
        - basic full computation
        - sparse basic
        - random rows dense
        - random rows sparse
        """
        N,M = VoverWHs.shape
        if self.method=="random" and self.S is None:
            self.S = N//self.rank

        if sp.issparse(VoverWHs):
            # return np.einsum("ia,ik,ij->akj",W,W,VoverWHs.toarray())
            WW = W[:, :, None] * W[:, None, :]   # shape (n, K, K)
            # T = (VoverWHs.T @ WW.reshape(W.shape[0], -1)).reshape(VoverWHs.shape[1], W.shape[1], W.shape[1])
            T = (WW.reshape(W.shape[0], -1).T @ VoverWHs).reshape(W.shape[1], W.shape[1],VoverWHs.shape[1])
            return T

        else:
            match self.method:
                case "einsum":
                    return np.einsum("ia,ik,ij->akj",W,W,VoverWHs)
                case "full":#works
                    # WW = (W[:,:,None]*W[:,None,:]).reshape(N,-1)
                    # print("norm VoverWH2", np.linalg.norm(VoverWHs))
                    # print("max VoverWH2",np.max(VoverWHs))
                    T = self.WWT @ VoverWHs #shape (R*R,M)
                    T = T.reshape(self.rank,self.rank,M)
                    # print("norm T",np.linalg.norm(T))
                    return T
                case "random":#sketching for each column independantly
                    # P = np.random((m, k))
                    sumWs = np.sum(W**2,axis=1)
                    P = (VoverWHs*sumWs[:,None])
                    P = (P/np.sum(P,axis=0)).T#probability matrix, each row is a distribution, shape (M,N)
                    # print(P)

                    # WW = (W[:,:,None]*W[:,None,:]).reshape(N,-1) #shape (N,R*R)
                    samples = torch.multinomial(torch.Tensor(P),num_samples=self.S,replacement=True).numpy() #shape (M,S)
                    # print(samples)

                    rows = np.repeat(np.arange(M), self.S)
                    cols = samples.ravel()
                    
                    # print(P.shape,np.max(cols),np.max(rows))
                    Mdata = VoverWHs[cols,rows]/(P[rows,cols]*self.S)

                    sparseMat = sp.csr_matrix((Mdata,(rows,cols)),shape=(M,N))

                    T = sparseMat@self.WW
                    # print(T.shape,type(T))
                    # print(M.shape,type(M))
                    # print(WW.shape,type(WW))
                    T = T.reshape(M, self.rank, self.rank).transpose(1, 2, 0) # M@WW has shape (M,R*R)

                    Ttrue = np.einsum("ia,ik,ij->akj",W,W,VoverWHs)

                    print(np.linalg.norm(Ttrue-T)/np.linalg.norm(Ttrue))
                    

                    return T
                case "svd":
                    T0 = self.factorR@VoverWHs
                    T = self.factorL@T0
                    T = T.reshape(self.rank,self.rank,M)
                    Ttrue = (self.WWT @ VoverWHs).reshape(self.rank,self.rank,M)
                    print("error T",np.linalg.norm(Ttrue-T)/np.linalg.norm(Ttrue))
                    print("norm True",np.linalg.norm(Ttrue))
                    return T

    def update_HALS_H(self,V,W,H): 
        R,_ = H.shape
        eps=np.finfo(float).eps

        #tensor of hessians need to have shape RxRxN
        N,M = V.shape
        Hwork = H.copy()
        

        """
        if sp.issparse(self.X):
            Bsparse = VoverWH(V,W,H,type="csr",eps=eps)
            B2sparse = VoverWH2(V,W,H,type="csr",eps=eps)
            Gsparse = W.T@(2*Bsparse)-sum_W #,eps=0

        V = V.toarray()
        WH = W@H
        # B2 = B2sparse.toarray()
        # B = Bsparse.toarray()
        B2 = V/(np.maximum(WH**2,eps)) #sparse
        B = V/(np.maximum(WH,eps))

        # np.set_printoptions(threshold=np.inf)

        # print(B2)
        # print(B2sparse.toarray())
        
        # print(np.abs(B-Bsparse.toarray()))

        G = W.T@(2*B) - sum_W #sparse
        
        Dsparse = (W**2).T @B2sparse
        D = (W**2).T @B2
        Tsparse = self.compute_hessians(W,B2sparse)
        T = self.compute_hessians(W,B2)

        # print(B2.dtype,B2sparse.dtype)
        print(np.max(B2),np.max(B2sparse))
        # print("B2:",np.allclose(B2,B2sparse.toarray()))
        # print("G:",np.allclose(G,Gsparse))
        # print("D:",np.allclose(D,Dsparse))
        # print("T:",np.allclose(T,Tsparse))
        print("B:",np.max(np.abs(B-Bsparse.toarray())))
        print("B2:",np.max(np.abs(B2-B2sparse.toarray())))
        print("G:",np.max(np.abs(G-Gsparse)))
        print("D:",np.max(np.abs(D-Dsparse)))
        print("T:",np.max(np.abs(T-Tsparse)))


        for it in range(self.iter_HALS):
            for a in range(R):
                num = G[a,:]- (T[a,:,:] * Hwork).sum(axis=0)+T[a,a,:]*Hwork[a,:]
                den = D[a,:]
                Hwork[a,:] = np.maximum(num/(den),eps)
                #putting 0 in max and eps in all fractions gives me a better fidelity but points won't draw idk why that's annoying
            

        return Hwork 
        """

        if sp.issparse(self.X):
            # sum_W = np.sum(self.W, axis = 0)[:, None]
            T = self.WWT @ VoverWH2(V,W,H,type="csr",eps=0)
            T = T.reshape(self.rank,self.rank,M)
            G = W.T@(2*VoverWH(V,W,H,type="csr",eps=0))-self.sum_W

        else:
            WH = W@H
            B2 = V/(WH**2) #sparse
            G = W.T@(2*V/(WH))-self.sum_W #sparse
            T = self.compute_hessians(W,B2)

        for it in range(self.iter_HALS):
            for a in range(R):
                deltaH = np.maximum((G[a,:]- (T[a,:,:] * Hwork).sum(axis=0))/T[a,a,:], eps-Hwork[a,:])
                Hwork[a,:] = Hwork[a,:]+deltaH

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

        eps = np.finfo(float).eps

        it=0

        while callback():
            
            

            #update H
            self.sum_W = np.sum(self.W, axis = 0)[:, None]
            #from here we can renormalize W if we want, to improve balancing
            # print("norm W", np.linalg.norm(self.W))
            # print("median W",np.median(self.W))
            # print("norm H", np.linalg.norm(self.H))
            # print("norm V", np.linalg.norm(self.X))
            # print("mean W", np.mean(self.W))
            #balancing 
            if self.balancing:
                medW = np.median(self.W)
                medH = np.median(self.H)
                medV = np.median(self.X)
                factor = (medV/(R**2 *medH**2))**(1/4)/medW
                print(factor)
                self.W *=factor

            # self.W *= 1e10
            self.WWT = ((self.W[:,:,None]*self.W[:,None,:]).reshape(N,-1)).T#shape (N,R*R)
            # print("norm WWT", np.linalg.norm(self.WWT))
            if self.method=="svd":
                U,S,Vh = np.linalg.svd(self.WWT,full_matrices=False)
                E = S**2
                relE = E.cumsum()/E.sum()
                # rank_svd =  self.rank
                # rank_svd =  self.rank**2
                rank_svd = (relE<self.svd_tol).sum()+1
                print(relE.tolist())
                print(self.rank,rank_svd)
                # self.factorL = U[:,:rank_svd]*S[:rank_svd]
                # self.factorR = Vh[:rank_svd,:]
                self.factorL = U[:,:rank_svd]*np.sqrt(S[:rank_svd])
                self.factorR = Vh[:rank_svd,:]*(np.sqrt(S[:rank_svd]))[:,None]
                print("error svd",np.linalg.norm(self.factorL@self.factorR-self.WWT)/np.linalg.norm(self.WWT))
            # print(S.tolist())
            for _ in range(D):
                self.H = self.update_HALS_H(self.X,self.W,self.H)
                # print("median H",np.median(self.H))
                # print(self.H)

            if self.balancing:
                self.W /= factor

            #update W
            self.sum_W = np.sum(self.H.T, axis = 0)[:, None]
            
            if self.balancing:
                medW = np.median(self.H)
                medH = np.median(self.W)
                medV = np.median(self.X)
                factor = (medV/(R**2 *medH**2))**(1/4)/medW
                print(factor)
                self.H *=factor

            # print("norm W", np.linalg.norm(self.H.T))
            # print("median W",np.median(self.H))
            # print("norm H", np.linalg.norm(self.W))
            # print("norm V", np.linalg.norm(self.X))
            # print("mean W", np.mean(self.H.T))
            self.WWT = (((self.H.T[:,:,None])*(self.H.T[:,None,:])).reshape(M,-1)).T
            # print("norm WWT", np.linalg.norm(self.WWT))
            if self.method=="svd":
                U,S,Vh = np.linalg.svd(self.WWT,full_matrices=False)
                E = S**2
                relE = E.cumsum()/E.sum()
                # rank_svd =  self.rank
                # rank_svd =  self.rank**2
                rank_svd = (relE<self.svd_tol).sum()+1
                print(relE.tolist())
                print(self.rank,rank_svd)
                # self.factorL = U[:,:rank_svd]*S[:rank_svd]
                # self.factorR = Vh[:rank_svd,:]
                self.factorL = U[:,:rank_svd]*np.sqrt(S[:rank_svd])
                self.factorR = Vh[:rank_svd,:]*(np.sqrt(S[:rank_svd]))[:,None]
                print("error svd",np.linalg.norm(self.factorL@self.factorR-self.WWT)/np.linalg.norm(self.WWT))
            # print(S.tolist())
            for _ in range(D):
                self.W = self.update_HALS_H(self.X.T,self.H.T,self.W.T).T
                # print("median H",np.median(self.W))
                # print(self.W)
            
            if self.balancing:
                self.H /= factor

            it+=1
            if self.sinkhorn_freq is not None and it%self.sinkhorn_freq==0:
                self.W, self.H = sinkhorn(self.X,self.W,self.H)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return dict(W=self.W, H=self.H)
    

"""
Debugging comments

no matter what the order between W and H is

first factor update works well
then first iteration of second one if fine
second iteration of second factor does not work

something that explains this issue, but for both:
at first no zeros in W and H
first serie of update puts 0 in W
first update for H is fine because still no zero in H
but then THERE ARE ZEROS IN BOTH W AND H
and I guess there are not handled the same depending on whever we are in the sparse or dense case

the issue is really in G and D, so the two matrices that do Wtruc.T@Bsparse

Other algorithms do not have this issue because they never do a max with 0
Am kind of does but apparently they do it cleanly enough that it does not matter

we HAVE TO set to eps to be able to compute the loss

insights from gpt:
the blas is better at handling numerical errors, einsum is not ideal apparently

"""

"""
We are essentially doing a matrix product, 
(R*R,N)*(N,M) -> NMR^2
and
(R*R,M)*(M,N) -> NMR^2

we cannot really touch to NM

SVD on 

SVD is not very stable, as even if we are full rank it is not exact

norm of T gets really big at some point
somehow the full algo still works but doing any op on one of the factor breaks everything
we need better conditionning

increase W so that the factors are evened out
what could be the more stable
it works be cause we are multiplying W with V/W@H and W^T V/(W@H)**2 W

can I also scale H for stability ?

I think we need to scale both

we compute the sum before any scaling, so that we are good to go

I don't care that the epsilons are not zero, that's the point I think, they are causing the problems

I need to focus to understand dynamics, but I think it is somewhat dommed because there might be huge dynamics in V/(WH)**2
I think these dynamics are essential for second order codes

I should test the sparse code again with that in mind, compare it to AmSOM, because AmSOM works (check), so it also should work

I think the main issue is that there can be huge dynamics in gradient and second order terms, especialy when we set to zero stuff a lot
(AmSOM might not set a lot of stuff to eps btw)

The main issue might also be that there are a lot of dynamics in T, so relative error does not mean a lot
should measure the relative error on each hessian if they have quite different magnitudes

what is important is not the total relative error, but the relative error BY HESSIAN (I think, to verify)


if sp.issparse(self.X):
        # sum_W = np.sum(self.W, axis = 0)[:, None]
        B2 = VoverWH2(V,W,H,type="csr",eps=0)
        G = W.T@(2*VoverWH(V,W,H,type="csr",eps=0))-self.sum_W

    else:
        WH = W@H
        B2 = V/(WH**2) #sparse
        G = W.T@(2*V/(WH))-self.sum_W #sparse
    
    D = (W**2).T @B2
    T = self.compute_hessians(W,B2)

    for it in range(self.iter_HALS):
        for a in range(R):
            num = G[a,:]- (T[a,:,:] * Hwork).sum(axis=0)+T[a,a,:]*Hwork[a,:]
            den = D[a,:]
            Hwork[a,:] = np.maximum(num/(den+eps),eps)
        

    return Hwork


"""