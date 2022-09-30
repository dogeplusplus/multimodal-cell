import pickle
import mlflow
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, TransformerMixin

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
def cross_validation(model_constructor: ModelConstructor):
    FP_MULTIOME_TRAIN_INPUTS = Path("open_problems_multimodal", "train_multi_inputs.h5")
    FP_MULTIOME_TRAIN_TARGETS = Path("open_problems_multimodal", "train_multi_targets.h5")

    chunksize = 5000
    TOTAL_SIZE = 105942
    chunks = int(np.ceil(TOTAL_SIZE / chunksize))
    columns_to_use = slice(10000, 14000)

    all_zero_columns = None

    # # Fit PCA by chunks of the data
    # pca = IncrementalPCA(n_components=4)
    # for idx in trange(0, chunks, desc="Fitting incremental PCA"):
    #     df = pd.read_hdf(
    #         FP_MULTIOME_TRAIN_INPUTS,
    #         start=idx*chunksize,
    #         stop=min((idx+1) * chunksize, TOTAL_SIZE)
    #     )
    #     X = df.values
    #     if all_zero_columns is None:
    #         all_zero_columns = (X == 0).all(axis=0)
    #     X = X[:, ~all_zero_columns]
    #     X = X[:, columns_to_use]
    #     pca.partial_fit(X)

    # TODO: Decide on suitable dimensionality reduction scheme for multiome
    # Truncated SVD
    df = pd.read_hdf(
        FP_MULTIOME_TRAIN_INPUTS,
        start=0,
        stop=chunksize,
    )
    X = df.values
    if all_zero_columns is None:
        all_zero_columns = (X == 0).all(axis=0)
    X = X[:, ~all_zero_columns]
    X = X[:, columns_to_use]
    svd = TruncatedSVD(n_components=128)
    svd.fit(X)

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    start_points = [idx * chunksize for idx in range(chunks)]

    fold_mses = []
    fold_correlations = []

    for fold, (train_starts, val_starts) in enumerate(kf.split(start_points)):
        model = model_constructor.instantiate()
        model = MultiOutputRegressor(model, n_jobs=-1)

        # Train SGD regressor using data chunks as minibatch
        for start in tqdm(train_starts, desc=f"Training Fold {fold}"):
            train_batch = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS, start=start, stop=start+chunksize)
            train_batch = train_batch.values
            train_batch = train_batch[:, ~all_zero_columns]
            train_batch = train_batch[:, columns_to_use]

            X_train = svd.transform(train_batch)
            y_train = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=start, stop=start+chunksize)
            y_train = y_train.values
            model.partial_fit(X_train, y_train)

        # For a given fold, build up the MSEs for each chunk in validation
        mses = []
        correlations = []

        val_bar = tqdm(val_starts, desc=f"Validation Fold {fold}:")
        for start in val_bar:
            val_batch = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS, start=start, stop=start+chunksize)
            val_batch = val_batch.values
            val_batch = val_batch[:, ~all_zero_columns]
            val_batch = val_batch[:, columns_to_use]

            X_val = svd.transform(val_batch)
            y_val = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=start, stop=start+chunksize)
            y_val = y_val.values
            y_val_pred = model.predict(X_val)

            mse = mean_squared_error(y_val, y_val_pred)
            mses.append(mse)
            corrscore = correlation_score(y_val, y_val_pred)
            correlations.append(corrscore)

            val_bar.set_postfix({"mse": np.mean(mses), "corrscore": np.mean(correlations)})

        fold_mses.append(np.mean(mses))
        fold_correlations.append(np.mean(correlations))

    logger.info(f"Mean MSE: {np.mean(fold_mses)}")
    logger.info(f"Mean Correlation: {np.mean(fold_correlations)}")
    mlflow.log_params(model_constructor.parameters)
    mlflow.log_metric("mse", np.mean(fold_mses))
    mlflow.log_metric("corrscore", np.mean(fold_correlations))


@timer
def multiome_submission(multiome_path: Path):
    # TODO: tidy up the hard coding, try to abstract away some of the model bits
    # TODO: Incorporate bits from the cross validation
    FP_MULTIOME_TRAIN_INPUTS = Path("open_problems_multimodal", "train_multi_inputs.h5")
    FP_MULTIOME_TEST_INPUTS = Path("open_problems_multimodal", "test_multi_inputs.h5")
    FP_MULTIOME_TRAIN_TARGETS = Path("open_problems_multimodal", "train_multi_targets.h5")
    FP_EVALUATION_IDS = Path("open_problems_multimodal", "evaluation_ids.csv")

    preprocessor = PreprocessMultiome()
    start = 0
    stop = 10000
    multi_train_x = preprocessor.fit_transform(
        pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS, start=start, stop=stop).values
    )

    multi_train_y = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=start, stop=stop)
    y_columns = multi_train_y.columns
    multi_train_y = multi_train_y.values

    model = Ridge(copy_X=False)
    model.fit(multi_train_x, multi_train_y)

    eval_ids = pd.read_csv(FP_EVALUATION_IDS, index_col="row_id")
    eval_ids.cell_id = eval_ids.cell_id.astype(pd.CategoricalDtype())
    eval_ids.gene_id = eval_ids.gene_id.astype(pd.CategoricalDtype())

    cell_id_set = set(eval_ids.cell_id)
    y_columns = pd.CategoricalIndex(y_columns, dtype=eval_ids.gene_id.dtype, name="gene_id")
    submission = pd.Series(name="target", index=pd.MultiIndex.from_frame(eval_ids), dtype=np.float32)

    start = 0
    chunksize = 5000
    total_rows = 0

    while True:
        multi_test_x = pd.read_hdf(FP_MULTIOME_TEST_INPUTS, start=start, stop=start+chunksize)
        rows_read = len(multi_test_x)
        needed_row_mask = multi_test_x.index.isin(cell_id_set)
        multi_test_x = multi_test_x.loc[needed_row_mask]
        multi_test_index = multi_test_x.index
        multi_test_x = multi_test_x.values
        multi_test_x = preprocessor.transform(multi_test_x)
        test_pred = model.predict(multi_test_x)

        test_pred = pd.DataFrame(
            test_pred,
            index=pd.CategoricalIndex(multi_test_index, dtype=eval_ids.cell_id.dtype, name="cell_id"),
            columns=y_columns,
        )

        for (index, row) in test_pred.iterrows():
            row = row.reindex(eval_ids.gene_id[eval_ids.cell_id == index])
            submission.loc[index] = row.values

        total_rows += len(multi_test_x)
        if rows_read < chunksize:
            break
        start += chunksize

    submission.reset_index(drop=True, inplace=True)
    submission.index.name = "row_id"
    with open(multiome_path, "wb") as f:
        pickle.dump(submission, f)
