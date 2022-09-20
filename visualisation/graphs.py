import typing as t
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def compare_hist(y_true: np.ndarray, y_pred: np.ndarray, columns: t.List[str]):
    gt_df = pd.DataFrame(y_true, columns=columns)
    pred_df = pd.DataFrame(y_pred, columns=columns)

    # TODO: Use bars instead and precompute the histograms to avoid logging underlying histogram data in mlflow
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=gt_df[columns[0]], opacity=0.5, name=f"{columns[0]}_true", nbinsx=40))
    fig.add_trace(go.Histogram(x=pred_df[columns[0]], opacity=0.5, name=f"{columns[0]}_pred", nbinsx=40))

    buttons = []
    for column in columns:
        buttons.append(dict(
            args=[{
                "x": [gt_df[column], pred_df[column]],
                "name": [f"{column}_true", f"{column}_pred"],
            },
                {
                "title": f"Distribution Comparisons: {column}",
            },
            ],
            method="update",
            label=column,
            visible=True,
        ))

    update_menu = []
    menu = dict()
    update_menu.append(menu)
    update_menu[0]["buttons"] = buttons
    update_menu[0]["direction"] = "down"
    update_menu[0]["showactive"] = True
    fig.update_layout(updatemenus=update_menu)

    return fig
