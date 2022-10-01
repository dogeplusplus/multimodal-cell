import logging
import numpy as np
import pandas as pd

from pathlib import Path

from data.create_indptr import create_indptr


logger = logging.getLogger(__name__)


def check_size(xs: np.ndarray, ys: np.ndarray, datas: np.ndarray) -> float:
    return (xs.nbytes + ys.nbytes + datas.nbytes) * 1e-9


def create_csr_arrays(h5_file_path: Path):
    # Initialize Variables
    chunksize = 1000
    loaded_rows = chunksize
    start = 0
    start_pos = 0
    file_pointer = 0

    # Initialize CSR arrays
    indptr = np.array([], dtype=np.int64)
    indices = np.array([], dtype=np.int32)
    data_s = np.array([], dtype=np.float32)

    prefix_filename = h5_file_path.name

    while chunksize == loaded_rows:
        size_gb = check_size(indptr, indices, data_s)
        if size_gb > 7.0:
            logger.info(f"Total size is {size_gb}. Saving...")
            np.save(f"{prefix_filename}_indptr_{file_pointer}.npy", indptr)
            np.save(f"{prefix_filename}_indices_{file_pointer}.npy", indices)
            np.save(f"{prefix_filename}_data_{file_pointer}.npy", data_s)

            # Re-initialize
            indptr = np.array([], dtype=np.int64)
            indices = np.array([], dtype=np.int32)
            data_s = np.array([], dtype=np.float32)
            file_pointer += 1

        logger.info("Reading .h5 chunk")
        df = pd.read_hdf(h5_file_path, start=start, stop=start+chunksize)
        logger.info("Extracting non-zero values")
        x_coords, y_coords = df.values.nonzero()
        tmp_data = df.values[df.values != 0.0]

        loaded_rows = df.shape[0]

        # Convert types
        y_coords = y_coords.astype(np.int32, copy=False)
        tmp_data = tmp_data.astype(np.float32, copy=False)

        logger.info("Compressing row values")
        x_coords = create_indptr(x_coords, start_pos=start_pos, nrows=loaded_rows)

        logger.info("Update variables")
        start_pos += y_coords.shape[0]
        start += chunksize
        # Append data at the end of each array
        indptr = np.hstack((indptr, x_coords))
        indices = np.hstack((indices, y_coords))
        data_s = np.hstack((data_s, tmp_data))

    logger.info("Done. Save last files")
    np.save(f"{prefix_filename}_indptr_{file_pointer.npy}", indptr)
    np.save(f"{prefix_filename}_indices_{file_pointer.npy}", indices)
    np.save(f"{prefix_filename}_data_{file_pointer.npy}", data_s)
