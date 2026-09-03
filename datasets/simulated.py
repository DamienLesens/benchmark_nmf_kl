import numpy as np

from benchopt import BaseDataset
from benchmark_utils.data_utils import sparsify,generate_mask


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'm_dim, n_dim, true_rank, estimated_rank': [
            (200,200,10,10),(500,500,20,20)],#(40, 40, 3, 3),(200,200,10,10),(200,200,30,30)
        'snr': [100],
        'rep': [1,2,3,4,5,6,7,8,9,10], #artificial repetitions to make multiple generation of random data
        'low_rank': [True,False], #whever the data is defined based on underlying factors
        'sparsity_factors': [1,0.9,0.3], #sparsity of the factors
        'noisy': [True,False], #whever we add noise to the model
        'noise_type': ['gaussian','poisson'] #which type of noise to use 
    }

    def get_data(self):
        """
        The generated factors are uniform on [0, 1], elementwise Gaussian iid
        noise is added to the data matrix.
        The Signal to Noise ratio is specified by the user.
        """
        seed = self.get_seed(
            use_objective=True,
            use_dataset=True,
            use_solver=False, #same data amongst solvers
            use_repetition=False #same data for all initializations
        )

        rng = np.random.RandomState(seed)

        if self.low_rank:

            W = rng.rand(self.m_dim, self.true_rank)
            H = rng.rand(self.true_rank, self.n_dim)

            if self.sparsity_factors<1:
                W = sparsify(W, s=self.sparsity_factors, epsilon=np.finfo(float).eps)
                H = sparsify(H, s=self.sparsity_factors, epsilon=np.finfo(float).eps)

            X = np.dot(W, H)

            if self.noisy:

                if self.noise_type=="gaussian":
                    noise = rng.randn(*X.shape)
                    sigma = 10**(-self.snr/20) * (
                        np.linalg.norm(X, ord="fro") / np.linalg.norm(noise, ord="fro")
                    )
                    X += sigma*noise
                    X = np.maximum(X,np.finfo(float).eps)
                    return dict(X=X, rank=self.estimated_rank, true_factors=[W, H])

                elif self.noise_type=="poisson":
                    sigma = 0.5*10**(self.snr/10)
                    X = np.maximum(rng.poisson(sigma*X), np.finfo(float).eps)
                    return dict(X=X, rank=self.estimated_rank, true_factors=[W, H])
                

            else:
                return dict(X=X, rank=self.estimated_rank, true_factors=[W, H])

        else:
            X = rng.rand(self.m_dim,self.n_dim)
            return dict(X=X, rank=self.estimated_rank)

        
