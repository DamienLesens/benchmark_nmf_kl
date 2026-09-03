from benchopt import BasePlot
import numpy as np
import pandas as pd
from benchmark_utils.style import get_style

class Plot(BasePlot):

    """
    Simple plot for synthetic datasets, averaging other matrix generation and initial point generation
    """

    name = "avg_synthetic_lr"
    type = "scatter"  # or "bar_chart", "boxplot" or "table"
    options = { # Will generate one plot per option entry !!!
        "dataset_param": [{'estimated_rank':10,'low_rank':True,'m_dim':200,'n_dim':200,'noise_type':'poisson','noisy':True,'snr':100,'true_rank':10,'sparsity_factors':1},
                          {'estimated_rank':10,'low_rank':True,'m_dim':200,'n_dim':200,'noise_type':'poisson','noisy':True,'snr':100,'true_rank':10,'sparsity_factors':0.9},
                          {'estimated_rank':10,'low_rank':True,'m_dim':200,'n_dim':200,'noise_type':'poisson','noisy':True,'snr':100,'true_rank':10,'sparsity_factors':0.3},
                          {'estimated_rank':10,'low_rank':True,'m_dim':200,'n_dim':200,'noise_type':'poisson','noisy':False,'snr':100,'true_rank':10,'sparsity_factors':1},
                          {'estimated_rank':10,'low_rank':True,'m_dim':200,'n_dim':200,'noise_type':'poisson','noisy':False,'snr':100,'true_rank':10,'sparsity_factors':0.9},
                          {'estimated_rank':10,'low_rank':True,'m_dim':200,'n_dim':200,'noise_type':'poisson','noisy':False,'snr':100,'true_rank':10,'sparsity_factors':0.3},
                          {'estimated_rank':20,'low_rank':True,'m_dim':500,'n_dim':500,'noise_type':'poisson','noisy':True,'snr':100,'true_rank':20,'sparsity_factors':1},
                          {'estimated_rank':20,'low_rank':True,'m_dim':500,'n_dim':500,'noise_type':'poisson','noisy':True,'snr':100,'true_rank':20,'sparsity_factors':0.9},
                          {'estimated_rank':20,'low_rank':True,'m_dim':500,'n_dim':500,'noise_type':'poisson','noisy':True,'snr':100,'true_rank':20,'sparsity_factors':0.3},
                          {'estimated_rank':20,'low_rank':True,'m_dim':500,'n_dim':500,'noise_type':'poisson','noisy':False,'snr':100,'true_rank':20,'sparsity_factors':1},
                          {'estimated_rank':20,'low_rank':True,'m_dim':500,'n_dim':500,'noise_type':'poisson','noisy':False,'snr':100,'true_rank':20,'sparsity_factors':0.9},
                          {'estimated_rank':20,'low_rank':True,'m_dim':500,'n_dim':500,'noise_type':'poisson','noisy':False,'snr':100,'true_rank':20,'sparsity_factors':0.3},],
        "objective": ...,
        "objective_column": ...,
    }

    def plot(self, df, objective, objective_column,dataset_param):

        plots = []

        mask = (df["objective_name"] == objective)

        for k, v in dataset_param.items():
            mask &= df[f"p_dataset_{k}"] == v

        df = df[mask]

        for solver, df_filtered in df.groupby('solver_name'):

            df_fill = (
                df_filtered.pivot_table(
                    index='stop_val',
                    columns=['p_dataset_rep', 'idx_rep'],
                    values=objective_column,
                    aggfunc='last'
                )
                .ffill()
            )

            y_rep = df_fill.T.groupby(level='p_dataset_rep').median().T
            y = y_rep.median(axis=1)

            # Time aggregation
            x = (
                df_filtered.groupby("stop_val")["time"]
                .median()
                .reindex(y.index)
            )

            curve_data = {
                "x": x.tolist(),
                "y": y.tolist(),
                **get_style(solver)
            }

            plots.append(curve_data)

        return plots
    
    def get_metadata(self, df, objective, objective_column, dataset_param):
        l = dataset_param["sparsity_factors"]
        noisy = dataset_param["noisy"]
        if noisy:
            return {
                "title": f"With noise and l={l}",
                "xlabel": "Time",
                "ylabel": "KL loss",
            }
        else:
            return {
                "title": f"Without noise and l={l}",
                "xlabel": "Time",
                "ylabel": "KL loss",
            }