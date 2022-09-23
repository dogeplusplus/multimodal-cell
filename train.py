import json
import scipy
import mlflow
import logging
# import lightgbm
import typing as t
import numpy as np
import pandas as pd


from tqdm import tqdm
from pathlib import Path
from typing import Callable
from dataclasses import dataclass
from xgboost import XGBRegressor
from scipy.sparse import csr_matrix
from tempfile import TemporaryDirectory
from pretty_html_table import build_table
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from data.svd import run_svd, timer
from visualisation.graphs import compare_hist


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConstructor:
    constructor: Callable
    parameters: t.Dict[str, t.Any]

    def instantiate(self):
        return self.constructor(**self.parameters)


def correlation_score(y_true, y_pred):
    if type(y_true) == pd.DataFrame:
        y_true = y_true.values
    if type(y_pred) == pd.DataFrame:
        y_pred = y_pred.values

    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


def mlflow_run(func):
    def wrapper(*args, **kwargs):
        mlflow.start_run()
        func(*args, **kwargs)
        mlflow.end_run()

    return wrapper


@timer
def fit_predict(
    model: t.Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
) -> np.ndarray:
    y_val_pred = []
    y_cols = y_train.shape[1]
    for i in tqdm(range(y_cols), ncols=100, desc="Model Training"):
        model.fit(X_train, y_train[:, i].copy())
        y_val_pred.append(model.predict(X_val))
    y_val_pred = np.column_stack(y_val_pred)

    return y_val_pred


@mlflow_run
@timer
def cross_validation(model_constructor: ModelConstructor):
    DATA_DIR = Path("open_problems_multimodal")
    FP_CELL_METADATA = DATA_DIR / "metadata.csv"
    FP_CITE_TRAIN_INPUTS = DATA_DIR / "train_cite_inputs.h5"
    FP_CITE_TRAIN_TARGETS = DATA_DIR / "train_cite_targets.h5"
    FP_CITE_TEST_INPUTS = DATA_DIR / "test_cite_inputs.h5"
    COLUMNS_PATH = DATA_DIR / "multimodal_columns.json"

    with open(COLUMNS_PATH) as f:
        columns = json.load(f)

    important_cols = columns["important"]
    constant_cols = columns["constant"]

    metadata_df = pd.read_csv(FP_CELL_METADATA, index_col="cell_id")
    metadata_df = metadata_df[metadata_df.technology == "citeseq"]
    metadata_df.shape

    X = pd.read_hdf(FP_CITE_TRAIN_INPUTS).drop(columns=constant_cols)
    cell_index = X.index
    meta = metadata_df.reindex(cell_index)
    X0 = X[important_cols].values

    # SVD performed on the combined train and test set
    X = csr_matrix(X.values)
    X_test = pd.read_hdf(FP_CITE_TEST_INPUTS).drop(columns=constant_cols)
    X_test = csr_matrix(X_test.values)

    logger.info(f"Original X shape: {X.shape} {X.size * 4 / (1024 ** 3):2.3f} GByte")
    logger.info(f"Original X_test shape: {X_test.shape} {X_test.size * (4 / 1024 ** 3):2.3f} GByte")

    combined = scipy.sparse.vstack([X, X_test])
    assert combined.shape[0] == 119651
    reduced, run_id = run_svd(combined)

    X = reduced[:70988]
    X_test = reduced[70988:]
    X = np.hstack([X, X0])

    logger.info(f"Reduced X shape: {X.shape} {X.size * 4 / (1024 ** 3):2.3f} GByte")
    Y = pd.read_hdf(FP_CITE_TRAIN_TARGETS)
    y_col_names = Y.columns
    Y = Y.values
    y_cols = Y.shape[1]
    n_splits = 3
    kf = GroupKFold(n_splits=n_splits)

    mses = []
    corrscores = []
    mses_column = []
    corrscores_column = []

    for train_idx, val_idx in kf.split(X, groups=meta.donor):
        model = model_constructor.instantiate()

        X_train = X[train_idx]
        y_train = Y[:, :y_cols][train_idx]
        X_val = X[val_idx]
        y_val = Y[:, :y_cols][val_idx]
        y_val_pred = fit_predict(model, X_train, y_train, X_val)

        col_mse = np.array([mean_squared_error(y_val[:, i], y_val_pred[:, i]) for i in range(y_cols)])
        col_corr = np.array([np.corrcoef(y_val[:, i], y_val_pred[:, i])[1, 0] for i in range(y_cols)])
        mse = np.mean(col_mse)
        corrscore = np.mean(col_corr)

        mses_column.append(col_mse)
        corrscores_column.append(col_corr)
        mses.append(mse)
        corrscores.append(corrscore)
        logger.info(f"shape = {X.shape[1]:4}: mse = {mse:.5f}, corr = {corrscore:.5f}")

    # Record scores for each column and display in descending order
    mse_df = pd.DataFrame(
        list(zip(y_col_names, np.mean(mses_column, axis=0))),
        columns=["column", "mse"],
    ).sort_values(by="mse", ascending=False)
    corr_df = pd.DataFrame(
        list(zip(y_col_names, np.mean(corrscores_column, axis=0))),
        columns=["column", "corrscore"],
    ).sort_values(by="corrscore", ascending=True)

    with TemporaryDirectory() as temp_dir:
        mse_table = build_table(mse_df, "blue_light")
        with open(Path(temp_dir, "mse.html"), "w") as f:
            f.write(mse_table)

        corr_table = build_table(corr_df, "blue_light")
        with open(Path(temp_dir, "corrscore.html"), "w") as f:
            f.write(corr_table)

        mlflow.log_artifact(Path(temp_dir, "mse.html"))
        mlflow.log_artifact(Path(temp_dir, "corrscore.html"))

    # Log distribution of predictions
    fig = compare_hist(y_val, y_val_pred, y_col_names)
    mlflow.log_figure(fig, "prediction_distributions.html")

    mlflow.log_params(model_constructor.parameters)
    mlflow.log_param("splits", n_splits)
    mlflow.log_param("run_id", run_id)
    mlflow.log_metric("mse", np.mean(mses))
    mlflow.log_metric("corrscore", np.mean(corrscores))


@mlflow_run
@timer
def create_submission(model_constructor: ModelConstructor):
    DATA_DIR = Path("open_problems_multimodal")
    FP_CITE_TRAIN_INPUTS = DATA_DIR / "train_cite_inputs.h5"
    FP_CITE_TRAIN_TARGETS = DATA_DIR / "train_cite_targets.h5"
    FP_CITE_TEST_INPUTS = DATA_DIR / "test_cite_inputs.h5"
    COLUMNS_PATH = DATA_DIR / "multimodal_columns.json"

    with open(COLUMNS_PATH) as f:
        columns = json.load(f)

    important_cols = columns["important"]
    constant_cols = columns["constant"]

    X = pd.read_hdf(FP_CITE_TRAIN_INPUTS).drop(columns=constant_cols)
    X0 = X[important_cols].values

    # SVD performed on the combined train and test set
    X = csr_matrix(X.values)
    X_test = pd.read_hdf(FP_CITE_TEST_INPUTS).drop(columns=constant_cols)
    X0_test = X_test[important_cols].values
    X_test = csr_matrix(X_test.values)

    combined = scipy.sparse.vstack([X, X_test])
    assert combined.shape[0] == 119651
    reduced, _ = run_svd(combined)

    X = reduced[:70988]
    X_test = reduced[70988:]
    X = np.hstack([X, X0])
    X_test = np.hstack([X_test, X0_test])

    logger.info(f"Original X shape: {X.shape} {X.size * 4 / (1024 ** 3):2.3f} GByte")
    logger.info(f"Original X_test shape: {X_test.shape} {X_test.size * (4 / 1024 ** 3):2.3f} GByte")
    logger.info(f"Reduced X shape: {X.shape} {X.size * 4 / (1024 ** 3):2.3f} GByte")
    Y = pd.read_hdf(FP_CITE_TRAIN_TARGETS)
    Y = Y.values

    test_predictions = []
    y_cols = Y.shape[1]

    for i in tqdm(range(y_cols), ncols=100, desc="Model Training"):
        model = model_constructor.instantiate()
        model.fit(X, Y[:, i].copy())
        test_predictions.append(model.predict(X_test))

    test_predictions = np.column_stack(test_predictions)
    # TODO: Replace with multiome pipeline later.
    submission = pd.read_csv(
        Path("open_problems_multimodal", "multiome-sparse-submission.csv"),
        index_col="row_id",
        squeeze=True,
    )
    submission.iloc[:len(test_predictions.ravel())] = test_predictions.ravel()
    assert not submission.isna().any()
    mlflow.set_experiment("submissions")

    with TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "submission.csv"
        submission.to_csv(temp_file)
        mlflow.log_artifact(temp_file)

    mlflow.log_params(model_constructor.parameters)


if __name__ == "__main__":
    # lightgbm_params = {
    #     "learning_rate": 0.1,
    #     "max_depth": 2,
    #     "num_leaves": 200,
    #     "min_child_samples": 250,
    #     "colsample_bytree": 0.8,
    #     "subsample": 0.6,
    #     "seed": 1,
    #     "device": "gpu",
    #     "verbosity": -1,
    #     "n_estimators": 100,
    # }
    xgb_params = {
        "eta": 0.3,
        "min_child_weight": 1,
        "max_depth": 6,
        "alpha": 0,
        "lambda": 1,
        "tree_method": "gpu_hist",
    }

    # lightgbm_builder = ModelConstructor(
    #     lightgbm.LGBMRegressor,
    #     lightgbm_params,
    # )
    xgb_builder = ModelConstructor(
        XGBRegressor,
        xgb_params,

    )
    # create_submission(lightgbm_params)
    cross_validation(xgb_builder)
