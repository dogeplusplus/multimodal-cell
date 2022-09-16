import uuid
import time
import json
import scipy
import pickle
import mlflow
import hashlib
import logging
import lightgbm
import functools
import typing as t
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_svd(svd_dir: Path, array: np.array, new_config: t.Dict[str, t.Any]) -> t.Union[str, bool]:
    for run in svd_dir.iterdir():
        run_config_path = run / "config.json"
        with open(run_config_path) as f:
            run_config = json.load(f)

        md5_array = hashlib.md5(str(array).encode("utf-8")).hexdigest()
        if md5_array != run_config["md5"]:
            continue

        for param, value in new_config.items():
            if run_config[param] != value:
                continue

        # Return folder if data and config are the same
        return run

    # No existing run matches this configuration
    return False


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed = toc - tic
        logger.info(f"Function Elapsed time {func.__name__}: {elapsed:0.4f} seconds")
        return value

    return wrapper_timer


@timer
def run_svd(
    array: np.array,
    n_components: int = 512,
    random_state: int = 1,
    n_iter: int = 5,
    algorithm: str = "randomized",
    n_oversamples: int = 10,
    power_iteration_normalizer: str = "auto",
    tol: float = 0.0,
    folder: Path = Path("precomputed_svd"),
) -> t.Tuple[np.array, TruncatedSVD]:
    # Store MD5 of array before SVD to avoid repeating work
    configuration = dict(
        md5=hashlib.md5(str(array).encode("utf-8")).hexdigest(),
        random_state=random_state,
        n_components=n_components,
        algorithm=algorithm,
        n_iter=n_iter,
        n_oversamples=n_oversamples,
        power_iteration_normalizer=power_iteration_normalizer,
        tol=tol,
    )

    run_id = check_svd(folder, array, configuration)
    # Load existing SVD if it exists already
    if run_id:
        logger.info("SVD with this configuration computed already")
        reduced_path = run_id / "reduced.npy"
        algorithm_path = run_id / "svd.pkl"

        with open(algorithm_path, "rb") as f:
            svd = pickle.load(f)

        reduced = np.load(reduced_path)
    # Otherwise create a new one
    else:
        logger.info("SVD with this configuration not cached. Creating new SVD run")
        logger.info(f"Shape before SVD: {array.shape}")
        svd = TruncatedSVD(
            n_components,
            random_state=random_state,
            algorithm=algorithm,
            n_iter=n_iter,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer,
            tol=tol,
        )
        run_id = str(uuid.uuid4())

        results_path = folder / run_id
        results_path.mkdir()

        array_path = results_path / "reduced.npy"
        algorithm_path = results_path / "svd.pkl"
        config_path = results_path / "config.json"

        reduced = svd.fit_transform(array)
        logger.info(f"Shape after SVD: {array.shape}")

        np.save(array_path, reduced)
        with open(algorithm_path, "wb") as f:
            pickle.dump(svd, f)

        with open(config_path, "w") as f:
            json.dump(configuration, f, indent=4)
        logger.info(f"SVD completed and stored in {str(results_path)}")

    return reduced, svd, run_id


def correlation_score(y_true, y_pred):
    if type(y_true) == pd.DataFrame:
        y_true = y_true.values
    if type(y_pred) == pd.DataFrame:
        y_pred = y_pred.values

    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


@timer
def train(X_train, y_train, lightgbm_params):
    model = lightgbm.LGBMRegressor(**lightgbm_params)
    model.fit(X_train, y_train)
    return model


def train_full(X_train, Y_train, X_test, lightgbm_params):
    test_predictions = []
    for i in range(Y_train.shape[1]):
        model = lightgbm.LGBMRegressor(**lightgbm_params)
        model.fit(X_train, Y_train[:i].copy())

        preds = model.predict(X_test)
        test_predictions.append(preds)

    test_pred = np.column_stack(test_predictions)
    del test_predictions

    return test_pred


@timer
def main():
    DATA_DIR = Path("open_problems_multimodal")
    FP_CELL_METADATA = DATA_DIR / "metadata.csv"

    FP_CITE_TRAIN_INPUTS = DATA_DIR / "train_cite_inputs.h5"
    FP_CITE_TRAIN_TARGETS = DATA_DIR / "train_cite_targets.h5"
    FP_CITE_TEST_INPUTS = DATA_DIR / "test_cite_inputs.h5"

    # FP_MULTIOME_TRAIN_INPUTS = DATA_DIR / "train_multi_inputs.h5"
    # FP_MULTIOME_TRAIN_TARGETS = DATA_DIR / "train_multi_targets.h5"
    # FP_MULTIOME_TRAIN_INPUTS = DATA_DIR / "test_multi_inputs.h5"

    # FP_SUBMISSION = DATA_DIR / "sample_submission.csv"
    # FP_EVALUATION_IDS = DATA_DIR / "evaluation_ids.csv"

    COLUMNS_PATH = Path("multimodal_columns.json")
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
    reduced, _, run_id = run_svd(combined)

    X = reduced[:70988]
    X_test = reduced[70988:]
    X = np.hstack([X, X0])

    logger.info(f"Reduced X shape: {X.shape} {X.size * 4 / (1024 ** 3):2.3f} GByte")
    Y = pd.read_hdf(FP_CITE_TRAIN_TARGETS)
    Y = Y.values

    lightgbm_params = {
        "learning_rate": 0.1,
        "max_depth": 10,
        "num_leaves": 200,
        "min_child_samples": 250,
        "colsample_bytree": 0.8,
        "subsample": 0.6,
        "seed": 1,
        "device": "gpu",
        "verbosity": -1,
        "n_estimators": 300,
    }

    y_cols = Y.shape[1]
    n_splits = 3
    kf = GroupKFold(n_splits=n_splits)
    mses = []
    corrscores = []

    for train_idx, val_idx in kf.split(X, groups=meta.donor):
        X_train = X[train_idx]
        y_train = Y[:, :y_cols][train_idx]
        X_val = X[val_idx]
        y_val = Y[:, :y_cols][val_idx]

        val_pred = []
        for i in range(y_cols):
            model = train(X_train, y_train[:, i].copy(), lightgbm_params)
            validation_pred = model.predict(X_val)
            val_pred.append(validation_pred)

        y_val_pred = np.column_stack(val_pred)

        mse = mean_squared_error(y_val, y_val_pred)
        corrscore = correlation_score(y_val, y_val_pred)
        mses.append(mse)
        corrscores.append(corrscore)
        logger.info(f"shape = {X.shape[1]:4}: mse = {mse:.5f}, corr = {corrscore:.5f}")

    mlflow.log_params(lightgbm_params)
    mlflow.log_param("splits", n_splits)
    mlflow.log_param("run_id", run_id)
    mlflow.log_metric("mse", np.mean(mses))
    mlflow.log_metric("corrscore", np.mean(corrscores))


if __name__ == "__main__":
    main()
