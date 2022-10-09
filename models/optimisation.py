import gc
import json
import optuna
import logging
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz, csr_matrix
from catboost import CatBoostRegressor
from optuna.integration.mlflow import MLflowCallback
from sklearn.model_selection import KFold, GroupKFold

from data.svd import run_svd
from utils.metrics import correlation_score


def multiome_objective(trial: optuna.Trial) -> float:
    seed = 1
    catboost_params = {
        "loss_function": "MultiRMSE",
        "eval_metric": "MultiRMSE",
        "task_type": "CPU",
        "devices": "0",
        "iterations": 200,
        "od_type": "Iter",
        "boosting_type": "Plain",
        "bootstrap_type": "Bayesian",
        "allow_const_label": True,
        "random_state": 1,
        "metric_period": 100,
        "silent": True,
    }

    catboost_params["depth"] = trial.suggest_int("depth", 6, 10)
    catboost_params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1, 10)
    catboost_params["learning_rate"] = trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2])

    DATA_DIR = Path("multimodal")
    FP_MULTIOME_TRAIN_INPUTS = DATA_DIR / "train_multiome_input_sparse.npz"
    FP_MULTIOME_TRAIN_TARGETS = DATA_DIR / "train_multiome_target_sparse.npz"

    # Truncated SVD
    X = load_npz(FP_MULTIOME_TRAIN_INPUTS)
    y = load_npz(FP_MULTIOME_TRAIN_TARGETS)

    # Run SVD on entire train/validation sets to ensure same features are used
    X, _, _ = run_svd(X, n_components=128, random_state=seed)
    X = X.astype(np.float16, copy=False)
    y_reduced, _, target_svd = run_svd(y, n_components=128, random_state=seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    scores = []
    for (train_index, val_index) in kf.split(X, y):
        X_train = X[train_index]

        # Fitting is done on SVD reduced training data
        y_train_reduced = y_reduced[train_index]
        model = CatBoostRegressor(**catboost_params)
        model.fit(X_train, y_train_reduced)

        del X_train, y_train_reduced
        gc.collect()

        X_val = X[val_index]
        y_val = y[val_index].toarray().astype(np.float16, copy=False)
        y_val_pred_reduced = model.predict(X_val)
        del X_val
        gc.collect()

        y_val_pred = y_val_pred_reduced @ target_svd.components_
        del y_val_pred_reduced
        gc.collect()
        val_corr = correlation_score(y_val, y_val_pred)
        scores.append(val_corr)
        del y_val, y_val_pred
        gc.collect()

    return np.mean(scores)


def citeseq_objective(trial: optuna.Trial) -> float:
    DATA_DIR = Path("multimodal")
    FP_CELL_METADATA = DATA_DIR / "metadata.csv"
    FP_CITE_TRAIN_INPUTS = DATA_DIR / "train_cite_inputs.h5"
    FP_CITE_TRAIN_TARGETS = DATA_DIR / "train_cite_targets.h5"
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

    seed = 1
    catboost_params = {
        "loss_function": "MultiRMSE",
        "eval_metric": "MultiRMSE",
        "task_type": "CPU",
        "devices": "0",
        "iterations": 200,
        "od_type": "Iter",
        "boosting_type": "Plain",
        "bootstrap_type": "Bayesian",
        "allow_const_label": True,
        "random_state": 1,
        "metric_period": 100,
        "silent": True,
    }

    catboost_params["depth"] = trial.suggest_int("depth", 6, 10)
    catboost_params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1, 10)
    catboost_params["learning_rate"] = trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2])

    # SVD performed on the combined train and test set
    X = csr_matrix(X.values)
    X, _, _ = run_svd(X, random_state=seed)
    X = np.hstack([X, X0])

    Y = pd.read_hdf(FP_CITE_TRAIN_TARGETS)
    Y = Y.values
    y_cols = Y.shape[1]
    n_splits = 3
    kf = GroupKFold(n_splits=n_splits)

    corrscores = []
    for train_idx, val_idx in kf.split(X, groups=meta.donor):
        model = CatBoostRegressor(**catboost_params)
        X_train = X[train_idx]
        y_train = Y[:, :y_cols][train_idx]
        X_val = X[val_idx]
        y_val = Y[:, :y_cols][val_idx]

        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)

        corrscore = correlation_score(y_val, y_val_pred)
        corrscores.append(corrscore)

    return np.mean(corrscores)


def optuna_multiome():
    logging.basicConfig(level=logging.INFO)
    multiome_study = optuna.create_study()
    mlflc = MLflowCallback(
        metric_name="correlation_score",
        mlflow_kwargs={"experiment": "multiome"},
    )
    multiome_study.optimize(multiome_objective, n_trials=5, callbacks=[mlflc])
    logging.info(f"Multiome Best parameters: {multiome_study.best_params}")
    logging.info(f"Multiome Best trial: {multiome_study.best_trial}")
    logging.info(f"Multiome Best CV correlation score: {multiome_study.best_value}")


def optuna_citeseq():
    citeseq_study = optuna.create_study()
    mlflc = MLflowCallback(
        metric_name="correlation_score",
        mlflow_kwargs={"experiment": "citeseq"},
    )
    citeseq_study.optimize(multiome_objective, n_trials=5, callbacks=[mlflc])
    logging.info(f"Citeseq Best parameters: {citeseq_study.best_params}")
    logging.info(f"Citeseq Best trial: {citeseq_study.best_trial}")
    logging.info(f"Citeseq Best CV correlation score: {citeseq_study.best_value}")
