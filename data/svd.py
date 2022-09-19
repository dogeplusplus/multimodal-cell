import uuid
import time
import json
import pickle
import logging
import hashlib
import functools
import typing as t
import numpy as np

from pathlib import Path
from sklearn.decomposition import TruncatedSVD


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

    return reduced, run_id
