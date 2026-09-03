from benchopt import BasePlot
import numpy as np
import pandas as pd

class Plot(BasePlot):

    """
    Average relative error wrt to relative time. Here relative means per dataset, but the average is on all datasets
    """

    name = "avgrelerr_reltime"
    type = "scatter"  # or "bar_chart", "boxplot" or "table"
    options = {       # Automatic options from DataFrame columns
        "objective": ...,
        "objective_column": ...,
        "n_points": [200],
    }

    def plot(self, df, objective, objective_column, n_points):

        t_grid = np.linspace(0, 1, n_points)

        #Refenrece values
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
        max_time_per_dataset = (
            df.groupby('dataset_name')['time']
            .max()
            .rename('max_time')
        )

        df = df.join(best_per_dataset, on='dataset_name')
        df = df.join(max_time_per_dataset, on='dataset_name')
        df['relative_time'] = df['time'] / df['max_time']
        df['relative_error'] = (
            (df[objective_column] - df['best'])
            / (df['best'].abs() + 1e-10)
        )

        # --- Interpolate each (dataset, solver, run) onto the common time grid ---
        # For each run, build a step function (ffill) of relative_error vs relative_time
        all_profiles = []  # will collect one row per (dataset, solver, run)

        for (dataset, solver, idx_rep), df_run in df.groupby(
            ['dataset_name', 'solver_name', 'idx_rep']
        ):

            run_series = (
                df_run.set_index('relative_time')['relative_error']
                .sort_index()
                .reindex(t_grid, method='ffill')
            )

            all_profiles.append({
                'dataset': dataset,
                'solver': solver,
                'idx_rep': idx_rep,
                'profile': run_series.values  # shape (n_points,)
            })

        df_profiles = pd.DataFrame(all_profiles)

        # --- Average across all (dataset, run) pairs per solver ---
        plots = []
        for solver, df_solver in df_profiles.groupby('solver'):
            # Stack all run profiles: shape (n_runs_total, n_points)
            matrix = np.stack(df_solver['profile'].values)

            # NaN at the start (before the run's first observation) are excluded
            mean_profile = np.nanmean(matrix, axis=0)

            plots.append({
                "x": t_grid.tolist(),
                "y": mean_profile.tolist(),
                "label": solver,
                "color": self.get_style(solver)["color"],
                "marker": " "
            })

        return plots
    
    def get_metadata(self, df, objective, objective_column, n_points):
        return {
            "title": f"Average relative error wrt relative time",
            "xlabel": "Relative Time",
            "ylabel": "Relative Error",
        }