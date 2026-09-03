from benchopt import run_benchmark
from benchopt.plotting import plot_benchmark
from benchopt.benchmark import Benchmark
from benchopt.cli.process_results import merge
from pathlib import Path

benchmark_path = Path(__file__).resolve().parent.parent
benchmark = Benchmark(benchmark_path,no_cache=False)

exp_name = "full_audio_f2"

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
time_groups = {'5': ["MAPS_MUS-bach_846_AkPnBcht[estimated_rank=[2,11]]","MyHeart"],
                '10': ["MAPS_MUS-bach_846_AkPnBcht[estimated_rank=[23,45]]",]}

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
        n_jobs=n_jobs
    ))

merge.callback(benchmark_path,filenames=output_file_list,output=exp_name)

# plot_benchmark(
#     fname=benchmark_path / "outputs" / f"{exp_name}.parquet",
#     benchmark=benchmark,
# )