import logging
import lightgbm
import numpy as np
import pandas as pd

from pathlib import Path
# from xgboost import XGBRegressor

from utils.decorators import timer
from models.multiome import multiome_submission
from models.citeseq import citeseq_submission, ModelConstructor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_combined_submission(citeseq_path: Path, multiome_path: Path, combined_path: Path):
    citeseq = np.load(citeseq_path)
    submission = pd.read_csv(
        multiome_path,
        index_col="row_id",
        squeeze=True,
    )
    submission.iloc[:len(citeseq.ravel())] = citeseq.ravel()
    assert not submission.isna().any()
    submission.to_csv(combined_path)


@timer
def main():
    lightgbm_params = {
        "learning_rate": 0.1,
        "max_depth": 5,
        "num_leaves": 200,
        "min_child_samples": 250,
        "colsample_bytree": 0.8,
        "subsample": 0.6,
        "seed": 1,
        "device": "gpu",
        "verbosity": -1,
        "n_estimators": 300,
    }

    lightgbm_builder = ModelConstructor(
        lightgbm.LGBMRegressor,
        lightgbm_params,
    )
    # xgb_params = {
    #     "eta": 0.3,
    #     "min_child_weight": 1,
    #     "max_depth": 6,
    #     "alpha": 0,
    #     "lambda": 1,
    #     "tree_method": "gpu_hist",
    # }

    # xgb_builder = ModelConstructor(
    #     XGBRegressor,
    #     xgb_params,

    # )

    submission_dir = Path("submissions")
    citeseq_dir = submission_dir / "citeseq"
    multiome_dir = submission_dir / "multiome"
    combined_dir = submission_dir / "combined"

    citeseq_dir.mkdir(exist_ok=True, parents=True)
    multiome_dir.mkdir(exist_ok=True, parents=True)
    combined_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Starting creation of Citeseq submission")
    citeseq_path = citeseq_dir / "citeseq.csv"
    citeseq_submission(lightgbm_builder, citeseq_path)
    logger.info(f"CiteSeq submission created at: {str(citeseq_path)}")

    logger.info("Starting creation of Multiome submission")
    multiome_path = multiome_dir / "multiome.pickle"
    multiome_submission(multiome_path)
    logger.info(f"Multiome submission created at: {str(multiome_path)}")

    logger.info(
        f"Starting creation of combined submission from CiteSeq: {str(citeseq_path)} Multiome: {str(multiome_path)}")
    combined_path = combined_dir / "submission.csv"
    create_combined_submission(citeseq_path, multiome_path, combined_path)
    logger.info(f"Combined submission created at {str(combined_path)}")


if __name__ == "__main__":
    main()
