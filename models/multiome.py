import gc
import time
import mlflow
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import dump
from pathlib import Path
from functools import reduce
from scipy.sparse import load_npz
from tempfile import TemporaryDirectory
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from data.svd import run_svd
from utils.decorators import mlflow_run, timer
from utils.metrics import correlation_score
from models.citeseq import ModelConstructor


logger = logging.getLogger(__name__)


@timer
@mlflow_run("multiome")
def cross_validation(model_constructor: ModelConstructor, seed: int = 1):
    DATA_DIR = Path("multimodal")
    FP_MULTIOME_TRAIN_INPUTS = DATA_DIR / "train_multiome_input_sparse.npz"
    FP_MULTIOME_TRAIN_TARGETS = DATA_DIR / "train_multiome_target_sparse.npz"

    # Truncated SVD
    X = load_npz(FP_MULTIOME_TRAIN_INPUTS)
    y = load_npz(FP_MULTIOME_TRAIN_TARGETS)

    # Run SVD on entire train/validation sets to ensure same features are used
    X, run_id, _ = run_svd(X, n_components=128)
    X = X.astype(np.float16, copy=False)
    y_reduced, target_run_id, target_svd = run_svd(y, n_components=128)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    train_mses = []
    val_mses = []
    train_correlations = []
    val_correlations = []

    pbar = tqdm(desc="Cross Validation")
    for (train_index, val_index) in kf.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index].toarray().astype(np.float16, copy=False)

        # Fitting is done on SVD reduced training data
        y_train_reduced = y_reduced[train_index]

        model = model_constructor.instantiate()
        model.fit(X_train, y_train_reduced)
        pbar.set_postfix_str("Cross validation models fitted.")

        y_train_pred_reduced = model.predict(X_train)
        del X_train, y_train_reduced
        gc.collect()

        y_train_pred = y_train_pred_reduced @ target_svd.components_
        del y_train_pred_reduced
        gc.collect()
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mses.append(train_mse)
        train_corr = correlation_score(y_train, y_train_pred)
        train_correlations.append(train_corr)
        del y_train, y_train_pred
        gc.collect()

        X_val = X[val_index]
        y_val = y[val_index].toarray().astype(np.float16, copy=False)
        y_val_pred_reduced = model.predict(X_val)
        del X_val
        gc.collect()

        y_val_pred = y_val_pred_reduced @ target_svd.components_
        del y_val_pred_reduced
        gc.collect()
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_mses.append(val_mse)
        val_corr = correlation_score(y_val, y_val_pred)
        val_correlations.append(val_corr)
        del y_val, y_val_pred
        gc.collect()

        metrics = {
            "train_mse": np.mean(train_mses),
            "train_corrscore": np.mean(train_correlations),
            "val_mse": np.mean(val_mses),
            "val_corrscore": np.mean(val_correlations),
        }
        pbar.set_postfix(metrics)

    # Save model from last cross validation
    with TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir, "models.joblib")
        dump(model, model_path)
        mlflow.log_artifact(model_path)

    # Log metrics from final fold
    mlflow.log_metrics(metrics)
    mlflow.log_params(model_constructor.parameters)
    mlflow.log_param("run_id", run_id)
    mlflow.log_param("target_run_id", target_run_id)
    mlflow.log_param("seed", seed)


def extract_index(hdf_path: Path) -> pd.Index:
    total_rows = 0
    chunk_size = 5000
    start = 0
    indices = []
    while True:
        df = None
        gc.collect()
        df = pd.read_hdf(hdf_path, start=start, stop=start+chunk_size)
        rows_read = len(df)
        index = df.index
        indices.append(index)

        total_rows += rows_read
        if rows_read < chunk_size:
            break
        start += chunk_size

    combined_index = reduce(lambda x, y: x.union(y), indices[1:], indices[0])
    return combined_index


@timer
def multiome_submission(model_constructor: ModelConstructor, multiome_path: Path, seed: int = 1):
    DATA_DIR = Path("multimodal")
    FP_MULTIOME_TRAIN_INPUTS = DATA_DIR / "train_multiome_input_sparse.npz"
    FP_MULTIOME_TEST_INPUTS = DATA_DIR / "test_multiome_input_sparse.npz"
    FP_MULTIOME_TRAIN_TARGETS = DATA_DIR / "train_multiome_target_sparse.npz"
    FP_EVALUATION_IDS = DATA_DIR / "evaluation_ids.csv"
    FP_MULTIOME_TEST_INPUTS_RAW = DATA_DIR / "test_multi_inputs.h5"

    pbar = tqdm(desc="Multiome Submission")

    start = time.time()
    X_train = load_npz(FP_MULTIOME_TRAIN_INPUTS)
    X_train, _, input_svd = run_svd(X_train)
    y_train = load_npz(FP_MULTIOME_TRAIN_TARGETS)
    y_train, _, target_svd = run_svd(y_train, n_components=128)

    end_svd = time.time()
    pbar.set_postfix_str(f"Ran SVD on inputs and targets after: {end_svd - start:.3f}s")

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    indices = np.arange(X_train.shape[0])
    models = []

    # Train ensemble with cross validation
    for train_fold, _ in kf.split(indices):
        X_fold = X_train[train_fold]
        y_fold = y_train[train_fold]

        model = model_constructor.instantiate()
        model.fit(X_fold, y_fold)
        del X_fold, y_fold
        gc.collect()
        models.append(model)

    end_training = time.time()
    pbar.set_postfix_str(f"Model fitted after: {end_training - end_svd:.3f}s")

    del X_train, y_train
    gc.collect()

    # Load raw multiome targets in order to extract column names
    FP_MULTIOME_TRAIN_TARGETS_RAW = DATA_DIR / "train_multi_targets.h5"
    df_target = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS_RAW, start=0, stop=1)
    test_index = extract_index(FP_MULTIOME_TEST_INPUTS_RAW)

    y_columns = df_target.columns
    del df_target
    gc.collect()

    X_test = load_npz(FP_MULTIOME_TEST_INPUTS)
    X_test = input_svd.transform(X_test)

    test_predictions = np.zeros((X_test.shape[0], 23418), dtype="float16")
    for idx, model in enumerate(models, 1):
        test_predictions += (model.predict(X_test) @ target_svd.components_) / idx
        gc.collect()

    del X_test
    gc.collect()
    end_inference = time.time()
    pbar.set_postfix_str(f"Test predictions obtained after: {end_inference - end_training:.3f}s")

    eval_ids = pd.read_csv(FP_EVALUATION_IDS, index_col="row_id")
    eval_ids.cell_id = eval_ids.cell_id.astype(pd.CategoricalDtype())
    eval_ids.gene_id = eval_ids.gene_id.astype(pd.CategoricalDtype())

    y_columns = pd.CategoricalIndex(y_columns, dtype=eval_ids.gene_id.dtype, name="gene_id")
    submission = pd.Series(name="target", index=pd.MultiIndex.from_frame(eval_ids), dtype=np.float16)

    cell_dict = dict((k, v) for v, k in enumerate(test_index))
    gene_dict = dict((k, v) for v, k in enumerate(y_columns))

    eval_ids_cell_num = eval_ids.cell_id.apply(lambda x: cell_dict.get(x, -1))
    eval_ids_gene_num = eval_ids.gene_id.apply(lambda x: gene_dict.get(x, -1))
    valid_multi_rows = (eval_ids_gene_num != -1) & (eval_ids_cell_num != -1)
    submission.iloc[valid_multi_rows] = test_predictions[
        eval_ids_cell_num[valid_multi_rows].to_numpy(),
        eval_ids_gene_num[valid_multi_rows].to_numpy(),
    ]

    submission.reset_index(drop=True, inplace=True)
    submission.index.name = "row_id"
    submission.to_csv(multiome_path)
