import numpy as np
import typing as t
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from tempfile import TemporaryDirectory
from scipy.sparse import hstack, csr_matrix, save_npz

from create_indptr import create_indptr


def check_size(xs: np.ndarray, ys: np.ndarray, datas: np.ndarray) -> float:
    return (xs.nbytes + ys.nbytes + datas.nbytes) * 1e-9


def create_csr_arrays(h5_file_path: Path, destination: Path) -> t.Tuple[t.List[Path], t.List[Path], t.List[Path]]:
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

    indptr_paths = []
    indice_paths = []
    data_paths = []

    pbar = tqdm()
    while chunksize == loaded_rows:
        size_gb = check_size(indptr, indices, data_s)
        if size_gb > 7.0:
            pbar.set_postfix_str(f"Total size is {size_gb}. Saving...")
            indptr_path = destination / f"indptr_{file_pointer}.npy"
            indice_path = destination / f"indice_{file_pointer}.npy"
            data_path = destination / f"data_{file_pointer}.npy"

            np.save(indptr_path, indptr)
            np.save(indice_path, indices)
            np.save(data_path, data_s)

            indptr_paths.append(indptr_path)
            indice_paths.append(indice_path)
            data_paths.append(data_path)

            # Re-initialize
            indptr = np.array([], dtype=np.int64)
            indices = np.array([], dtype=np.int32)
            data_s = np.array([], dtype=np.float32)
            file_pointer += 1

        pbar.set_postfix_str("Reading .h5 chunk")
        df = pd.read_hdf(h5_file_path, start=start, stop=start+chunksize)
        pbar.set_postfix_str("Extracting non-zero values")
        x_coords, y_coords = df.values.nonzero()
        tmp_data = df.values[df.values != 0.0]

        loaded_rows = df.shape[0]

        # Convert types
        y_coords = y_coords.astype(np.int32, copy=False)
        tmp_data = tmp_data.astype(np.float32, copy=False)

        pbar.set_postfix_str("Compressing row values")
        x_coords = create_indptr(x_coords, start_pos=start_pos, nrows=loaded_rows)

        pbar.set_postfix_str("Update variables")
        start_pos += y_coords.shape[0]
        start += chunksize

        # Append data at the end of each array
        indptr = np.hstack((indptr, x_coords))
        indices = np.hstack((indices, y_coords))
        data_s = np.hstack((data_s, tmp_data))

        pbar.update(loaded_rows)

    pbar.set_postfix_str("Done. Save last files")
    indptr_path = destination / f"indptr_{file_pointer}.npy"
    indice_path = destination / f"indice_{file_pointer}.npy"
    data_path = destination / f"data_{file_pointer}.npy"

    np.save(indptr_path, indptr)
    np.save(indice_path, indices)
    np.save(data_path, data_s)

    indptr_paths.append(indptr_path)
    indice_paths.append(indice_path)
    data_paths.append(data_path)

    return indptr_paths, indice_paths, data_paths


def create_sparse_matrix(
    indptr_paths: t.List[Path],
    indice_paths: t.List[Path],
    data_paths: t.List[Path],
) -> csr_matrix:
    # Known shapes of the multiome inputs apriori
    N_ROWS = 105942
    N_COLS = 228942

    csr_arrays = []
    for ptr, ind, dat in zip(indptr_paths, indice_paths, data_paths):
        indptr = np.load(ptr)
        indices = np.load(ind)
        data = np.load(dat)

        # Indptr has shape nrows instead of nrows + 1, can add last elemenent
        # corresponding to the length of indices or data arrays
        indptr = np.append(indptr, indptr[-1] + indices[indptr[-1]:].shape)
        csr = csr_matrix((data, indices, indptr), shape=(N_ROWS, N_COLS))
        csr_arrays.append(csr)

    return hstack(csr_arrays)


def main():
    DATA_DIR = Path("multimodal")
    TRAIN_MULTI_INPUTS = DATA_DIR / "train_multi_inputs.h5"
    TEST_MULTI_INPUTS = DATA_DIR / "test_multi_inputs.h5"

    TRAIN_OUTPUT_PATH = DATA_DIR / "train_multiome_input_sparse.npz"
    TEST_OUTPUT_PATH = DATA_DIR / "test_multiome_input_sparse.npz"

    for source, destination in [
        (TRAIN_MULTI_INPUTS, TRAIN_OUTPUT_PATH),
        (TEST_MULTI_INPUTS, TEST_OUTPUT_PATH),
    ]:
        with TemporaryDirectory() as temp_dir:
            indptr_paths, indice_paths, data_paths = create_csr_arrays(source, Path(temp_dir))
            stacked_csr = create_sparse_matrix(indptr_paths, indice_paths, data_paths)
        save_npz(destination, stacked_csr)


if __name__ == "__main__":
    main()
