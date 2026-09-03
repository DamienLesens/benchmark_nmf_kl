from benchopt import BasePlot
import numpy as np

class Plot(BasePlot):

    """
    Performance profile with respect to the last point in the plot, one performance plot per dataset
    """

    name = "perf_end_time_per_dataset"
    type = "scatter"  # or "bar_chart", "boxplot" or "table"
    options = {
        "dataset": ...,         # Automatic options from DataFrame columns
        "objective": ...,
        "objective_column": ...,
    }

    def plot(self, df, dataset, objective, objective_column):

        taus = np.logspace(-6, 0, 200)  # error margins from 1e-6 to 1 (relative)

        plots = []

        df = df.query("dataset_name == @dataset and objective_name == @objective")

        # Keep only the last value of each (solver, run) pair
        df_last = (
            df.groupby(['solver_name', 'idx_rep'])
            .last()
            .reset_index()[['solver_name', 'idx_rep', objective_column]]
        )

        # Best performance across all solvers and all runs on this dataset
        best = df_last[objective_column].min()  # assumes minimization

        for solver, df_solver in df_last.groupby('solver_name'):
            n_runs = len(df_solver)

            # For each tau: fraction of runs within tau relative error of best
            final_values = df_solver[objective_column].values
            relative_errors = (final_values - best) / (abs(best) + 1e-10)

            profile = [
                (relative_errors <= tau).sum() / n_runs * 100
                for tau in taus
            ]

            plots.append({
                "x": taus.tolist(),
                "y": profile,
                "label": solver,
                **self.get_style(solver)
            })

        return plots

    def get_metadata(self, df, dataset, objective, objective_column):
        return {
            "title": f"Performance profile for {dataset}",
            "xlabel": "Error",
            "ylabel": "Performance",
        }