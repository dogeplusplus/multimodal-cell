import time
import logging
import mlflow
import functools

from functools import wraps


logger = logging.getLogger(__name__)


def mlflow_run(experiment_name):
    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
            fn(*args, **kwargs)
            mlflow.end_run()

        return wrapper
    return decorate


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
