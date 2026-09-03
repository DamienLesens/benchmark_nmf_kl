from benchopt import run_benchmark
from benchopt.plotting import plot_benchmark
from benchopt.benchmark import Benchmark
from benchopt.cli.process_results import merge
from pathlib import Path

benchmark_path = Path(__file__).resolve().parent.parent
benchmark = Benchmark(benchmark_path,no_cache=False)

exp_name = "full_simulated_full_rank_f"

solver_list = [
        "MU",
        # "MU_Burg",
        "FPA",
        # "admm[rho=[0.1,1,10,100,1000]]",
        "som",
        "scalar_newton",
        "newton[iter_HALS=[5]]",
        "HALS[iter_HALS=[5]]",
    ]

n_rep = 10

n_jobs = 1

max_runs = 100 #arbitrarly big

#in this dictionary put for each runtime the datasets that should be ran that long
time_groups = {'1': ["simulated[estimated_rank=10,low_rank=False,m_dim=200,n_dim=200,noise_type=poisson,noisy=True,snr=100,true_rank=10,sparsity_factors=1]"],
                '4': ["simulated[estimated_rank=20,low_rank=False,m_dim=500,n_dim=500,noise_type=poisson,noisy=True,snr=100,true_rank=20,sparsity_factors=1]",]}

output_file_list = []

for time,dataset_list in time_groups.items():
    output_file_list.append(run_benchmark(
        benchmark_path=benchmark_path,
        solver_names=solver_list,
        dataset_names=dataset_list,
        n_repetitions=n_rep,
        timeout=int(time)*n_rep,
        max_runs=max_runs,
        plot_result=False,
        n_jobs=n_jobs,
    ))

merge.callback(benchmark_path,filenames=output_file_list,output=exp_name)

# plot_benchmark(
#     fname=Path("../outputs/"+exp_name+".parquet"),
#     benchmark=benchmark,
# )