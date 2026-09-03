from benchopt import BasePlot
import numpy as np
from benchmark_utils.style import get_style

class Plot(BasePlot):

    """
    Performance profile with respect to the last point in the plot, aggregated across all datasets
    """

    name = "perf_end_time_global"
    type = "scatter"  # or "bar_chart", "boxplot" or "table"
    options = {
        "objective": ...,
        "objective_column": ...,
    }

    def plot(self, df, objective, objective_column):

        taus = np.logspace(-6, 0, 100)

        plots = []

        df = df.query("objective_name == @objective")

        # Keep only the last value of each (dataset, solver, run) triplet
        df_last = (
            df.groupby(['dataset_name', 'solver_name', 'idx_rep'])
            .last()
            .reset_index()[['dataset_name', 'solver_name', 'idx_rep', objective_column]]
        )

        # Best performance per dataset across all solvers and runs
        best_per_dataset = (
            df_last.groupby('dataset_name')[objective_column]
            .min()  # assumes minimization
            .rename('best')
        )

        # Attach best to each row and compute relative error
        df_last = df_last.join(best_per_dataset, on='dataset_name')
        df_last['relative_error'] = (
            (df_last[objective_column] - df_last['best'])
            / (df_last['best'].abs() + 1e-10)
        )

        # Total number of (dataset, run) pairs — same denominator for all solvers
        n_total = df_last[['dataset_name', 'idx_rep']].drop_duplicates().shape[0]

        for solver, df_solver in df_last.groupby('solver_name'):
            errors = df_solver['relative_error'].values

            profile = [
                (errors <= tau).sum() / n_total * 100
                for tau in taus
            ]

            plots.append({
                "x": taus.tolist(),
                "y": profile,
                **get_style(solver)
                # "color": self.get_style(solver)["color"],
                # "marker": " "
            })

        return plots

    def get_metadata(self, df, objective, objective_column):
        return {
            "title": f"Performance profiles",
            "xlabel": "Relative Error",
            "ylabel": "Performance (%)",
        }