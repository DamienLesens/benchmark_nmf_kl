from benchopt import BasePlot
import numpy as np

class Plot(BasePlot):

    """
    Puts all datasets on a shared relative time scale (x axis) and plots for a given relative error tau the percentage of runs that achieved
    error tau at given relative time
    """

    name = "when_error_achieved"
    type = "scatter"  # or "bar_chart", "boxplot" or "table"
    options = {       # Automatic options from DataFrame columns
        "objective": ...,
        "objective_column": ...,
        "tau": [0.1,0.01,0.001],
        "n_points": [200],
    }

    def plot(self, df, objective, objective_column, tau, n_points):

        # Computing reference values
        # Best per dataset from last observations across all solvers/runs
        df_last = (
            df.groupby(['dataset_name', 'solver_name', 'idx_rep'])
            .last()
            .reset_index()[['dataset_name', 'solver_name', 'idx_rep', objective_column]]
        )
        best_per_dataset = (
            df_last.groupby('dataset_name')[objective_column]
            .min()
            .rename('best')
        )

        # Max time per dataset to normalize the time axis
        max_time_per_dataset = (
            df.groupby('dataset_name')['time']
            .max()
            .rename('max_time')
        )

        # --- Attach references and compute relative quantities ---
        df = df.join(best_per_dataset, on='dataset_name')
        df = df.join(max_time_per_dataset, on='dataset_name')

        df['relative_time'] = df['time'] / df['max_time']
        df['relative_error'] = (
            (df[objective_column] - df['best'])
            / (df['best'].abs() + 1e-10)
        )

        # --- For each (dataset, solver, run): first relative time achieving error <= tau ---
        first_achievement = (
            df[df['relative_error'] <= tau]
            .groupby(['dataset_name', 'solver_name', 'idx_rep'])['relative_time']
            .min()
            .rename('first_time')
            .reset_index()
        )

        # Shared denominator: all (dataset, run) pairs regardless of solver
        n_total = df[['dataset_name', 'idx_rep']].drop_duplicates().shape[0]

        t_grid = np.linspace(0, 1, n_points)

        plots = []
        for solver, df_solver in first_achievement.groupby('solver_name'):
            times = df_solver['first_time'].values

            # Empirical CDF of first-achievement times
            profile = [(times <= t).sum() / n_total * 100 for t in t_grid]

            plots.append({
                "x": t_grid.tolist(),
                "y": profile,
                "label": solver,
                "color": self.get_style(solver)["color"],
                "marker": " "
            })

        return plots
    
    def get_metadata(self, df, objective, objective_column, tau, n_points):
        return {
            "title": f"Percentage of runs that achieved relative error {tau} at a given relative time",
            "xlabel": "Relative Time",
            "ylabel": "Percentage",
        }