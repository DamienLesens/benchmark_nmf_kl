from benchopt import BasePlot

class Plot(BasePlot):

    """
    Plotting all the repetitions for each dataset
    """

    name = "all_rep"
    type = "scatter" # or "bar_chart", "boxplot" or "table"
    options = {
        "dataset": ..., # Automatic options from DataFrame columns
        "objective": ...,
        "objective_column": ...,
    }

    def plot(self, df, dataset, objective, objective_column):
        plots = []

        df = df.query(
            "dataset_name == @dataset and objective_name == @objective"
        )

        for solver, df_solver in df.groupby('solver_name'):
            for idx_rep, df_filtered in df_solver.groupby('idx_rep'):

                # One value per stop_val for this specific run
                df_run = df_filtered.groupby('stop_val').last()

                y = df_run[objective_column]
                x = df_run['time']

                curve_data = {
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "label": f"{solver} (run {idx_rep})",
                    **self.get_style(solver)
                }
                plots.append(curve_data)

        return plots

    def get_metadata(self, df, dataset, objective, objective_column):
        return {
            "title": f"All repetitions for {dataset}",
            "xlabel": "Time [sec]",
            "ylabel": "Objective value",
        }