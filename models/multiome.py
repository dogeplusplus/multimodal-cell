import gc
# import pickle
import mlflow
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import dump
from pathlib import Path
from scipy.sparse import load_npz
from tempfile import TemporaryDirectory
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

from data.svd import run_svd
from utils.decorators import mlflow_run, timer
from utils.metrics import correlation_score
from models.citeseq import ModelConstructor


logger = logging.getLogger(__name__)


class PreprocessMultiome(BaseEstimator, TransformerMixin):
    columns_to_use = slice(10000, 14000)

    @staticmethod
    def take_column_subset(X):
        return X[:, PreprocessMultiome.columns_to_use]

    def transform(self, X):
        X = X[:, ~self.all_zero_columns]
        X = PreprocessMultiome.take_column_subset(X)

        X = self.pca.transform(X)
        return X

    def fit_transform(self, X):
        self.all_zero_columns = (X == 0).all(axis=0)
        X = X[:, ~self.all_zero_columns]
        X = PreprocessMultiome.take_column_subset(X)

        self.pca = PCA(n_components=4, copy=False, random_state=1)
        X = self.pca.fit_transform(X)
        return X


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


@timer
def multiome_submission(model_constructor: ModelConstructor, multiome_path: Path):
    # TODO: Update this function once cross validation is working again

    DATA_DIR = Path("multimodal")
    FP_MULTIOME_TRAIN_INPUTS = DATA_DIR / "train_multiome_input_sparse.npz"
    FP_MULTIOME_TEST_INPUTS = DATA_DIR / "test_multiome_input_sparse.npz"
    FP_MULTIOME_TRAIN_TARGETS = DATA_DIR / "train_multiome_target_sparse.npz"
    FP_EVALUATION_IDS = DATA_DIR / "evaluation_ids.csv"
    # Load raw multiome targets in order to extract column names
    FP_MULTIOME_TRAIN_TARGETS_RAW = DATA_DIR / "train_multi_target.h5"

    df_target = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS_RAW, start=0, stop=1)
    y_columns = df_target.columns

    X_train = load_npz(FP_MULTIOME_TRAIN_INPUTS)
    y_train = load_npz(FP_MULTIOME_TRAIN_TARGETS)
    X_test = load_npz(FP_MULTIOME_TEST_INPUTS)

    X_train, _, _ = run_svd(X_train)
    model = model_constructor.instantiate()
    y_train, _, target_svd = run_svd(y_train, n_components=128)
    model.fit(X_train, y_train)

    del X_train, y_train
    gc.collect()

    test_predictions = model.predict(X_test)
    del X_test
    gc.collect()
    test_predictions = test_predictions @ target_svd.components_

    # TODO: figure out how to add the submissions
    eval_ids = pd.read_csv(FP_EVALUATION_IDS, index_col="row_id")
    eval_ids.cell_id = eval_ids.cell_id.astype(pd.CategoricalDtype())
    eval_ids.gene_id = eval_ids.gene_id.astype(pd.CategoricalDtype())

    # cell_id_set = set(eval_ids.cell_id)
    y_columns = pd.CategoricalIndex(y_columns, dtype=eval_ids.gene_id.dtype, name="gene_id")
    # submission = pd.Series(name="target", index=pd.MultiIndex.from_frame(eval_ids), dtype=np.float32)

    # while True:
    #     rows_read = len(multi_test_x)
    #     needed_row_mask = multi_test_x.index.isin(cell_id_set)
    #     multi_test_x = multi_test_x.loc[needed_row_mask]
    #     multi_test_index = multi_test_x.index
    #     test_pred = model.predict(multi_test_x)

    #     test_pred = pd.DataFrame(
    #         test_pred,
    #         index=pd.CategoricalIndex(multi_test_index, dtype=eval_ids.cell_id.dtype, name="cell_id"),
    #         columns=y_columns,
    #     )

    #     for (index, row) in test_pred.iterrows():
    #         row = row.reindex(eval_ids.gene_id[eval_ids.cell_id == index])
    #         submission.loc[index] = row.values

    #     total_rows += len(multi_test_x)
    #     if rows_read < chunksize:
    #         break
    #     start += chunksize

    # submission.reset_index(drop=True, inplace=True)
    # submission.index.name = "row_id"
    # with open(multiome_path, "wb") as f:
    #     pickle.dump(submission, f)
