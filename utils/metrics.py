import numpy as np
import pandas as pd


def correlation_score(y_true, y_pred):
    if type(y_true) == pd.DataFrame:
        y_true = y_true.values
    if type(y_pred) == pd.DataFrame:
        y_pred = y_pred.values

    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)
