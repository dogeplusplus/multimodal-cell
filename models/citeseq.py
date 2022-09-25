import json
import scipy
import mlflow
import logging
import typing as t
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from typing import Callable
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from tempfile import TemporaryDirectory
from pretty_html_table import build_table
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from data.svd import run_svd
from utils.decorators import mlflow_run, timer
from utils.metrics import correlation_score
from visualisation.graphs import compare_hist


logger = logging.getLogger(__name__)


@dataclass
class ModelConstructor:
    constructor: Callable
    parameters: t.Dict[str, t.Any]

    def instantiate(self):
        return self.constructor(**self.parameters)


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


@mlflow_run("citeseq")
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

    for train_idx, val_idx in kf.split(X, groups=meta.donor):
        model = model_constructor.instantiate()

        X_train = X[train_idx]
        y_train = Y[:, :y_cols][train_idx]
        X_val = X[val_idx]
        y_val = Y[:, :y_cols][val_idx]
        y_val_pred = fit_predict(model, X_train, y_train, X_val)

        col_mse = np.array([mean_squared_error(y_val[:, i], y_val_pred[:, i]) for i in range(y_cols)])
        mses_column.append(col_mse)
        mse = np.mean(col_mse)
        mses.append(mse)
        corrscore = correlation_score(y_val, y_val_pred)
        corrscores.append(corrscore)
        logger.info(f"shape = {X.shape[1]:4}: mse = {mse:.5f}, corr = {corrscore:.5f}")

    # Record scores for each column and display in descending order
    mse_df = pd.DataFrame(
        list(zip(y_col_names, np.mean(mses_column, axis=0))),
        columns=["column", "mse"],
    ).sort_values(by="mse", ascending=False)

    with TemporaryDirectory() as temp_dir:
        mse_table = build_table(mse_df, "blue_light")
        with open(Path(temp_dir, "mse.html"), "w") as f:
            f.write(mse_table)

        mlflow.log_artifact(Path(temp_dir, "mse.html"))

    # Log distribution of predictions
    fig = compare_hist(y_val, y_val_pred, y_col_names)
    mlflow.log_figure(fig, "prediction_distributions.html")

    mlflow.log_params(model_constructor.parameters)
    mlflow.log_param("splits", n_splits)
    mlflow.log_param("run_id", run_id)
    mlflow.log_metric("mse", np.mean(mses))
    mlflow.log_metric("corrscore", np.mean(corrscores))


@timer
def citeseq_submission(model_constructor: ModelConstructor, citeseq_path: Path):
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
    np.save(citeseq_path, test_predictions)
