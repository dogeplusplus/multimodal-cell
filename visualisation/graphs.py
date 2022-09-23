import typing as t
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def compare_hist(y_true: np.ndarray, y_pred: np.ndarray, columns: t.List[str]):
    gt_df = pd.DataFrame(y_true, columns=columns)
    pred_df = pd.DataFrame(y_pred, columns=columns)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=gt_df[columns[0]], opacity=0.5, name=f"{columns[0]}_true"))
    fig.add_trace(go.Bar(x=pred_df[columns[0]], opacity=0.5, name=f"{columns[0]}_pred"))

    buttons = []
    for column in sorted(columns):
        v_min = min(gt_df[column].min(), pred_df[column].min())
        v_max = max(gt_df[column].max(), pred_df[column].max())
        gt_data, gt_bins = np.histogram(gt_df[column], range=(v_min, v_max), bins=50)
        pred_data, pred_bins = np.histogram(pred_df[column], range=(v_min, v_max), bins=50)
        buttons.append(dict(
            args=[{
                "x": [gt_bins, pred_bins],
                "y": [gt_data, pred_data],
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
