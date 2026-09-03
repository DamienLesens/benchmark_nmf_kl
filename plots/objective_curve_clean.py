from benchopt import BasePlot
import numpy as np
import pandas as pd
from benchmark_utils.style import get_style


class Plot(BasePlot):
    name = "objective_clean"
    type = "scatter"
    options = {
        "dataset": ...,
        "objective": ...,
        "objective_column": ...,
        "X_axis": ["Time", "Iteration"],
    }

    def plot(self, df, dataset, objective, objective_column, X_axis):
        plots = []

        df = df.query(
            "dataset_name == @dataset and objective_name == @objective"
        )

        for solver, df_filtered in df.groupby('solver_name'):

            # Make sure that median value is computed with all runs.
            df_fill = (
                df_filtered.pivot_table(
                    index='stop_val',
                    columns='idx_rep',
                    values=[objective_column],
                    # should only have one value of each config
                    aggfunc="last"
                ).ffill()
                .unstack()
            )

            y = df_fill.groupby('stop_val').median(numeric_only=True)
            if X_axis == "Iteration":
                x = y.index
            else:
                x = df_filtered.groupby('stop_val')['time'].median()
            
            curve_data = {
                "x": x.tolist(),
                "y": y.tolist(),
                **get_style(solver)

            }

            plots.append(curve_data)

        return plots

    def get_metadata(self, df, dataset, objective, objective_column, X_axis):
        df = df[df["dataset_name"] == dataset]
        df = df[df['objective_name'] == objective]
        title = f"{objective}\nData: {dataset} "
        return {
            "title": title,
            "xlabel": X_axis,
            "ylabel": "KL loss",
        }
